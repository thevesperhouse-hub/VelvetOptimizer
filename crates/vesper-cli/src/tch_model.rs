//! VesperLM Model — tch-rs (LibTorch/PyTorch) backend
//!
//! Identical architecture to the Candle version but using PyTorch's autograd.
//! PyTorch frees intermediate activations during backward pass automatically,
//! enabling batch 64+ on 96GB VRAM (vs ~70GB for batch 10 with Candle).

use tch::{nn, nn::Module, Kind, Tensor};
use vesper_core::VesperConfig;

// ======================== ERA Activation ========================
// ERA(x) = GELU(x) + γ * softplus(x)

pub struct ERAActivation {
    gamma: f64,
}

impl ERAActivation {
    pub fn new(gamma: f64) -> Self {
        Self { gamma }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gelu = x.gelu("tanh");
        if self.gamma > 0.0 {
            // softplus(x) = log(1 + exp(x))
            let sp = (x.exp() + 1.0).log();
            &gelu + sp * self.gamma
        } else {
            gelu
        }
    }
}

// ======================== FlyLoRA Linear ========================
// Base linear + learnable low-rank adaptation (sparse projection omitted — unused in forward)

pub struct FlyLoRALinear {
    base: nn::Linear,
    lora_b_down: nn::Linear,
    lora_b_up: nn::Linear,
    scaling: f64,
}

impl FlyLoRALinear {
    pub fn new(vs: &nn::Path, in_features: i64, out_features: i64, rank: i64, alpha: f64) -> Self {
        let base = nn::linear(vs / "base", in_features, out_features, Default::default());
        let lora_b_down = nn::linear(vs / "lora_b_down", in_features, rank, Default::default());
        let lora_b_up = nn::linear(vs / "lora_b_up", rank, out_features, Default::default());
        let scaling = alpha / rank as f64;
        Self { base, lora_b_down, lora_b_up, scaling }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let base_out = self.base.forward(x);
        let lora_down = self.lora_b_down.forward(x);
        let lora_up = self.lora_b_up.forward(&lora_down);
        base_out + lora_up * self.scaling
    }
}

// ======================== Feed Forward ========================
// GLU-style: gate(FlyLoRA) → ERA → mul(up(FlyLoRA)) → down(Linear)

pub struct FeedForward {
    gate_proj: FlyLoRALinear,
    up_proj: FlyLoRALinear,
    down_proj: nn::Linear,
    era: ERAActivation,
}

impl FeedForward {
    pub fn new(vs: &nn::Path, config: &VesperConfig) -> Self {
        let h = config.hidden_size as i64;
        let inter = config.intermediate_size as i64;
        let rank = config.flylora_rank as i64;
        let alpha = config.flylora_alpha;

        Self {
            gate_proj: FlyLoRALinear::new(&(vs / "gate_proj"), h, inter, rank, alpha),
            up_proj: FlyLoRALinear::new(&(vs / "up_proj"), h, inter, rank, alpha),
            down_proj: nn::linear(vs / "down_proj", inter, h, Default::default()),
            era: ERAActivation::new(config.era_gamma),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward(x);
        let gate_activated = self.era.forward(&gate);
        let up = self.up_proj.forward(x);
        let gated = &gate_activated * &up;
        self.down_proj.forward(&gated)
    }
}

// ======================== Expert FFN (for MoE) ========================

pub struct ExpertFFN {
    gate_proj: FlyLoRALinear,
    up_proj: FlyLoRALinear,
    down_proj: nn::Linear,
    era: ERAActivation,
}

impl ExpertFFN {
    pub fn new(vs: &nn::Path, config: &VesperConfig) -> Self {
        let h = config.hidden_size as i64;
        let inter = config.intermediate_size as i64;
        let rank = config.flylora_rank as i64;
        let alpha = config.flylora_alpha;

        Self {
            gate_proj: FlyLoRALinear::new(&(vs / "gate_proj"), h, inter, rank, alpha),
            up_proj: FlyLoRALinear::new(&(vs / "up_proj"), h, inter, rank, alpha),
            down_proj: nn::linear(vs / "down_proj", inter, h, Default::default()),
            era: ERAActivation::new(config.era_gamma),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.gate_proj.forward(x);
        let gate_activated = self.era.forward(&gate);
        let up = self.up_proj.forward(x);
        let gated = &gate_activated * &up;
        self.down_proj.forward(&gated)
    }
}

// ======================== MoE Layer ========================

pub struct MoELayer {
    router: nn::Linear,
    experts: Vec<ExpertFFN>,
    num_experts: usize,
    top_k: usize,
}

impl MoELayer {
    pub fn new(vs: &nn::Path, config: &VesperConfig) -> Self {
        let h = config.hidden_size as i64;
        let n = config.moe_num_experts as i64;

        let router = nn::linear(vs / "router", h, n, Default::default());
        let experts: Vec<_> = (0..config.moe_num_experts)
            .map(|i| ExpertFFN::new(&(vs / format!("experts.{}", i)), config))
            .collect();

        Self { router, experts, num_experts: config.moe_num_experts, top_k: config.moe_top_k }
    }

    pub fn forward(&self, x: &Tensor) -> (Tensor, Tensor) {
        let dims = x.size();
        let (batch, seq, hidden) = (dims[0], dims[1], dims[2]);
        let num_tokens = batch * seq;
        let device = x.device();
        let kind = x.kind();

        // Router: softmax over expert logits
        let router_logits = self.router.forward(x);
        let router_probs = router_logits.softmax(-1, Kind::Float);

        // Top-K routing
        let (top_k_probs, top_k_indices) = router_probs.topk(self.top_k as i64, -1, true, true);
        let normalizer = top_k_probs.sum_dim_intlist(-1, true, Kind::Float);
        let top_k_weights = &top_k_probs / &normalizer;

        // Flatten to token-level
        let x_flat = x.reshape(&[num_tokens, hidden]);
        let tk_idx_flat = top_k_indices.reshape(&[num_tokens, self.top_k as i64]);
        let tk_w_flat = top_k_weights.reshape(&[num_tokens, self.top_k as i64]);

        // CPU sync for routing decisions
        let mut expert_token_ids: Vec<Vec<i64>> = vec![Vec::new(); self.num_experts];
        let mut expert_token_weights: Vec<Vec<f64>> = vec![Vec::new(); self.num_experts];

        for t in 0..num_tokens {
            for k in 0..self.top_k as i64 {
                let eid = tk_idx_flat.int64_value(&[t, k]) as usize;
                let w = tk_w_flat.double_value(&[t, k]);
                expert_token_ids[eid].push(t);
                expert_token_weights[eid].push(w);
            }
        }

        // Run each expert on its assigned tokens, scatter results back
        let mut output_flat = Tensor::zeros(&[num_tokens, hidden], (kind, device));

        for expert_id in 0..self.num_experts {
            let n_sel = expert_token_ids[expert_id].len();
            if n_sel == 0 { continue; }

            let idx = Tensor::from_slice(&expert_token_ids[expert_id]).to_device(device);
            let x_expert = x_flat.index_select(0, &idx);
            let expert_out = self.experts[expert_id].forward(&x_expert);

            let w = Tensor::from_slice(&expert_token_weights[expert_id])
                .to_device(device)
                .to_kind(kind)
                .unsqueeze(1)
                .expand_as(&expert_out);
            let weighted = &expert_out * &w;

            // scatter_add: differentiable accumulation
            let scatter_idx = idx.unsqueeze(1).expand_as(&weighted);
            output_flat = output_flat.scatter_add(0, &scatter_idx, &weighted);
        }

        let output = output_flat.reshape(&[batch, seq, hidden]);

        // Load balancing auxiliary loss
        let aux_loss = self.load_balancing_loss(&router_probs, &top_k_indices);

        (output, aux_loss)
    }

    fn load_balancing_loss(&self, router_probs: &Tensor, top_k_indices: &Tensor) -> Tensor {
        let dims = router_probs.size();
        let (batch, seq) = (dims[0], dims[1]);
        let total_assignments = (batch * seq * self.top_k as i64) as f64;

        // P_i: mean router probability per expert
        let p_mean = router_probs
            .mean_dim(&[0i64, 1][..], false, Kind::Float);

        // f_i: fraction of token-slots per expert
        let mut expert_fractions: Vec<Tensor> = Vec::with_capacity(self.num_experts);
        for expert_id in 0..self.num_experts {
            let mask = top_k_indices.eq(expert_id as i64).to_kind(Kind::Float);
            let count = mask.sum(Kind::Float);
            expert_fractions.push(count / total_assignments);
        }

        let f_tensor = Tensor::stack(&expert_fractions, 0);
        let products = &f_tensor * &p_mean;
        products.sum(Kind::Float) * (self.num_experts as f64)
    }
}

// ======================== Multi-Head Attention with RoPE ========================

pub struct MultiHeadAttention {
    num_heads: i64,
    head_dim: i64,
    hidden_size: i64,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    o_proj: nn::Linear,
    rope_theta: f64,
}

impl MultiHeadAttention {
    pub fn new(vs: &nn::Path, config: &VesperConfig) -> Self {
        let h = config.hidden_size as i64;
        Self {
            num_heads: config.num_heads as i64,
            head_dim: h / config.num_heads as i64,
            hidden_size: h,
            q_proj: nn::linear(vs / "q_proj", h, h, Default::default()),
            k_proj: nn::linear(vs / "k_proj", h, h, Default::default()),
            v_proj: nn::linear(vs / "v_proj", h, h, Default::default()),
            o_proj: nn::linear(vs / "o_proj", h, h, Default::default()),
            rope_theta: config.rope_theta,
        }
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Tensor {
        let dims = hidden_states.size();
        let (batch, seq_len) = (dims[0], dims[1]);

        let q = self.q_proj.forward(hidden_states);
        let k = self.k_proj.forward(hidden_states);
        let v = self.v_proj.forward(hidden_states);

        // Reshape: [B, S, H] -> [B, num_heads, S, head_dim]
        let q = q.view([batch, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);
        let k = k.view([batch, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);
        let v = v.view([batch, seq_len, self.num_heads, self.head_dim]).transpose(1, 2);

        // Apply RoPE
        let q = self.apply_rope(&q, seq_len);
        let k = self.apply_rope(&k, seq_len);

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(-2, -1)) / scale;

        // Apply causal + padding mask
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights + mask,
            None => {
                let causal = self.create_causal_mask(seq_len, hidden_states);
                attn_weights + causal
            }
        };

        let attn_probs = attn_weights.softmax(-1, Kind::Float).to_kind(v.kind());

        // Apply attention to values
        let context = attn_probs.matmul(&v);

        // Reshape back: [B, num_heads, S, head_dim] -> [B, S, H]
        let context = context.transpose(1, 2).contiguous().view([batch, seq_len, self.hidden_size]);

        self.o_proj.forward(&context)
    }

    /// LLaMA-style split-half RoPE
    fn apply_rope(&self, x: &Tensor, seq_len: i64) -> Tensor {
        let half_dim = self.head_dim / 2;
        let device = x.device();

        // inv_freq[i] = 1 / theta^(2i/d)
        let i = Tensor::arange(half_dim, (Kind::Float, device));
        let inv_freq = (&i * (-2.0 * self.rope_theta.ln() / self.head_dim as f64)).exp();

        // Positions: [0, 1, ..., seq_len-1]
        let positions = Tensor::arange(seq_len, (Kind::Float, device));

        // Angles: [seq_len, half_dim] = outer product
        let angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0);

        // cos/sin: [1, 1, seq_len, half_dim]
        let cos = angles.cos().unsqueeze(0).unsqueeze(0).to_kind(x.kind());
        let sin = angles.sin().unsqueeze(0).unsqueeze(0).to_kind(x.kind());

        // Split and rotate
        let x1 = x.narrow(-1, 0, half_dim);
        let x2 = x.narrow(-1, half_dim, half_dim);
        let rotated_x1 = &x1 * &cos - &x2 * &sin;
        let rotated_x2 = &x2 * &cos + &x1 * &sin;

        Tensor::cat(&[rotated_x1, rotated_x2], -1)
    }

    fn create_causal_mask(&self, seq_len: i64, reference: &Tensor) -> Tensor {
        // Upper triangular → -1e9 (causal: can't attend to future positions)
        let mask = Tensor::ones(&[seq_len, seq_len], (Kind::Float, reference.device()))
            .triu(1) * (-1e9);
        mask.view([1, 1, seq_len, seq_len]).to_kind(reference.kind())
    }
}

// ======================== Transformer Layer ========================

enum FFNLayer {
    Standard(FeedForward),
    MoE(MoELayer),
}

pub struct TransformerLayer {
    attention: MultiHeadAttention,
    ffn: FFNLayer,
    attention_norm: nn::LayerNorm,
    ffn_norm: nn::LayerNorm,
}

impl TransformerLayer {
    pub fn new(vs: &nn::Path, config: &VesperConfig) -> Self {
        let h = config.hidden_size as i64;
        let ln_cfg = nn::LayerNormConfig { eps: config.layer_norm_eps, ..Default::default() };

        let attention = MultiHeadAttention::new(&(vs / "attention"), config);

        let ffn = if config.moe_enabled {
            FFNLayer::MoE(MoELayer::new(&(vs / "ffn"), config))
        } else {
            FFNLayer::Standard(FeedForward::new(&(vs / "ffn"), config))
        };

        let attention_norm = nn::layer_norm(vs / "attention_norm", vec![h], ln_cfg);
        let ffn_norm = nn::layer_norm(vs / "ffn_norm", vec![h], ln_cfg);

        Self { attention, ffn, attention_norm, ffn_norm }
    }

    pub fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> (Tensor, Option<Tensor>) {
        // Pre-norm attention + residual
        let normed = hidden_states.apply(&self.attention_norm);
        let attn_out = self.attention.forward(&normed, attention_mask);
        let h = hidden_states + attn_out;

        // Pre-norm FFN + residual
        let normed = h.apply(&self.ffn_norm);
        let (ffn_out, aux_loss) = match &self.ffn {
            FFNLayer::Standard(ffn) => (ffn.forward(&normed), None),
            FFNLayer::MoE(moe) => {
                let (out, aux) = moe.forward(&normed);
                (out, Some(aux))
            }
        };

        (&h + ffn_out, aux_loss)
    }
}

// ======================== VesperLM ========================

pub struct TchVesperLM {
    config: VesperConfig,
    embeddings: nn::Embedding,
    layers: Vec<TransformerLayer>,
    final_norm: nn::LayerNorm,
    lm_head: nn::Linear,
}

impl TchVesperLM {
    pub fn new(vs: &nn::Path, config: &VesperConfig) -> Self {
        let h = config.hidden_size as i64;
        let v = config.vocab_size as i64;
        let ln_cfg = nn::LayerNormConfig { eps: config.layer_norm_eps, ..Default::default() };

        let embeddings = nn::embedding(vs / "embeddings", v, h, Default::default());

        let layers: Vec<_> = (0..config.num_layers)
            .map(|i| TransformerLayer::new(&(vs / format!("layers.{}", i)), config))
            .collect();

        let final_norm = nn::layer_norm(vs / "final_norm", vec![h], ln_cfg);
        let lm_head = nn::linear(vs / "lm_head", h, v, Default::default());

        Self { config: config.clone(), embeddings, layers, final_norm, lm_head }
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> (Tensor, Option<Tensor>) {
        let mut hidden = self.embeddings.forward(input_ids);
        let mask_4d = self.compute_mask_4d(attention_mask, &hidden);

        let mut total_aux: Option<Tensor> = None;

        for layer in &self.layers {
            let (new_h, aux) = layer.forward(&hidden, mask_4d.as_ref());
            hidden = new_h;
            if let Some(a) = aux {
                total_aux = Some(match total_aux {
                    Some(acc) => acc + a,
                    None => a,
                });
            }
        }

        let normed = hidden.apply(&self.final_norm);
        let logits = self.lm_head.forward(&normed);

        (logits, total_aux)
    }

    pub fn config(&self) -> &VesperConfig {
        &self.config
    }

    /// Compute 4D causal+padding attention mask from 2D mask
    fn compute_mask_4d(&self, attention_mask: Option<&Tensor>, hidden: &Tensor) -> Option<Tensor> {
        let mask = attention_mask?;
        let dims = mask.size();
        if dims.len() != 2 { return Some(mask.shallow_clone()); }
        let seq_len = dims[1];

        // Causal mask: upper triangular = -1e9
        let causal = Tensor::ones(&[seq_len, seq_len], (Kind::Float, hidden.device()))
            .triu(1) * (-1e9);
        let causal = causal.view([1, 1, seq_len, seq_len]).to_kind(hidden.kind());

        // Padding mask: [B, S] -> [B, 1, 1, S]
        let padding = (Tensor::ones_like(mask).to_kind(Kind::Float)
            - mask.to_kind(Kind::Float)) * (-1e9);
        let padding = padding.unsqueeze(1).unsqueeze(1).to_kind(hidden.kind());

        Some(causal + padding)
    }
}
