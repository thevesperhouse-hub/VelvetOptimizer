"""VesperLM — Full Transformer with Flash Attention 2, FlyLoRA, ERA, MoE, RoPE.

Architecture mirrors the Rust vesper-core model exactly:
  Embedding → N × TransformerLayer (MultiHeadAttention + FFN/MoE) → LayerNorm → LM Head

Features:
  - Flash Attention 2 via torch.nn.functional.scaled_dot_product_attention
  - FlyLoRA (sparse low-rank adaptation) on attention and FFN projections
  - ERA activation (GELU + γ * softplus)
  - Mixture of Experts (MoE) with top-K routing and load balancing loss
  - RoPE (LLaMA-style split-half rotation)
  - Gradient checkpointing via torch.utils.checkpoint
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .config import VesperConfig

import warnings

try:
    from .kernels import era_kernel, HAS_TRITON
except ImportError:
    era_kernel = None
    HAS_TRITON = False

# Flash Attention 2 detection
HAS_FLASH_ATTN_2 = False
try:
    from flash_attn import flash_attn_func  # noqa: F401
    HAS_FLASH_ATTN_2 = True
except ImportError:
    pass

# PyTorch SDPA backend check (FlashAttention or Memory-Efficient)
_SDPA_FLASH_AVAILABLE = hasattr(torch.nn.functional, "scaled_dot_product_attention")
if _SDPA_FLASH_AVAILABLE and torch.cuda.is_available():
    try:
        # PyTorch 2.x includes FlashAttention via SDPA when flash-attn is installed
        from torch.backends.cuda import (
            flash_sdp_enabled,
            mem_efficient_sdp_enabled,
        )
        if not flash_sdp_enabled():
            warnings.warn(
                "Flash Attention not available via PyTorch SDPA. "
                "Install flash-attn>=2.8.3 for 2-4x faster attention. "
                "Falling back to memory-efficient or math kernel."
            )
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# ERA Activation
# ---------------------------------------------------------------------------

class ERAActivation(nn.Module):
    """Entropy-Regularized Activation: ERA(x) = GELU(x) + γ * softplus(x)"""

    def __init__(self, gamma: float = 0.1):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if HAS_TRITON and era_kernel is not None and x.is_cuda:
            return era_kernel(x, self.gamma)
        gelu_x = F.gelu(x, approximate="tanh")
        if self.gamma > 0.0:
            sp = F.softplus(x)
            return gelu_x + self.gamma * sp
        return gelu_x


# ---------------------------------------------------------------------------
# FlyLoRA Linear
# ---------------------------------------------------------------------------

class FlyLoRALinear(nn.Module):
    """Sparse Low-Rank Adaptation inspired by fruit fly mushroom body coding.

    base(x) + scaling * lora_up(lora_down(x))
    where scaling = alpha / rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        sparsity: float = 0.25,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Base linear (will be frozen during LoRA-only training)
        self.base = nn.Linear(in_features, out_features)

        # Frozen sparse projection (not used in forward — kept for compatibility)
        proj_a = torch.randn(in_features, rank)
        mask = torch.rand_like(proj_a) > sparsity
        proj_a = proj_a * mask.float()
        col_norms = proj_a.norm(dim=0, keepdim=True).clamp(min=1e-8)
        proj_a = proj_a / col_norms
        self.register_buffer("proj_a", proj_a)

        # Learnable low-rank matrices
        self.lora_down = nn.Linear(in_features, rank, bias=True)
        self.lora_up = nn.Linear(rank, out_features, bias=True)

        # Initialize LoRA: small init for stable training start
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(x))
        return base_out + self.scaling * lora_out


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Embedding)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """LLaMA-style split-half RoPE."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        half_dim = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half_dim, dtype=torch.float32) * 2.0 / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos/sin for max_seq_len
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [max_seq, half_dim]
        self.register_buffer("cos_cached", angles.cos())  # [max_seq, half_dim]
        self.register_buffer("sin_cached", angles.sin())  # [max_seq, half_dim]

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply RoPE to tensor of shape [batch, heads, seq, head_dim]."""
        cos = self.cos_cached[:seq_len].to(x.dtype)  # [seq, half_dim]
        sin = self.sin_cached[:seq_len].to(x.dtype)

        # Broadcast: [1, 1, seq, half_dim]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        half = self.head_dim // 2
        x1 = x[..., :half]
        x2 = x[..., half:]

        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x2 * cos + x1 * sin
        return torch.cat([rotated_x1, rotated_x2], dim=-1)


# ---------------------------------------------------------------------------
# Multi-Head Attention (with Flash Attention 2)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, config: VesperConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.hidden_size = config.hidden_size
        self.use_flash_attn = config.use_flash_attn

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.rope = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape: [B, S, H] → [B, num_heads, S, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = self.rope(q, seq_len)
        k = self.rope(k, seq_len)

        # Attention
        if self.use_flash_attn and q.is_cuda:
            # Flash Attention 2 via PyTorch's SDPA (uses FlashAttention backend when available)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                is_causal=(attention_mask is None),
                dropout_p=0.0,
            )
        else:
            # Manual attention (CPU fallback)
            scale = math.sqrt(self.head_dim)
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            else:
                causal = torch.triu(
                    torch.full((seq_len, seq_len), -1e9, device=q.device, dtype=q.dtype),
                    diagonal=1,
                ).unsqueeze(0).unsqueeze(0)
                attn_weights = attn_weights + causal

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [B, num_heads, S, head_dim] → [B, S, H]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        return self.o_proj(attn_output)


# ---------------------------------------------------------------------------
# Feed-Forward (GLU-style with FlyLoRA + ERA)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """GLU-style FFN: gate_proj(FlyLoRA) → ERA → mul(up_proj(FlyLoRA)) → down_proj."""

    def __init__(self, config: VesperConfig):
        super().__init__()
        self.gate_proj = FlyLoRALinear(
            config.hidden_size, config.intermediate_size,
            rank=config.flylora_rank, alpha=config.flylora_alpha,
            sparsity=config.flylora_sparsity,
        )
        self.up_proj = FlyLoRALinear(
            config.hidden_size, config.intermediate_size,
            rank=config.flylora_rank, alpha=config.flylora_alpha,
            sparsity=config.flylora_sparsity,
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.era = ERAActivation(gamma=config.era_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.era(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# Expert FFN (for MoE)
# ---------------------------------------------------------------------------

class ExpertFFN(nn.Module):
    """Single expert — identical architecture to FeedForward."""

    def __init__(self, config: VesperConfig):
        super().__init__()
        self.gate_proj = FlyLoRALinear(
            config.hidden_size, config.intermediate_size,
            rank=config.flylora_rank, alpha=config.flylora_alpha,
            sparsity=config.flylora_sparsity,
        )
        self.up_proj = FlyLoRALinear(
            config.hidden_size, config.intermediate_size,
            rank=config.flylora_rank, alpha=config.flylora_alpha,
            sparsity=config.flylora_sparsity,
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.era = ERAActivation(gamma=config.era_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.era(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ---------------------------------------------------------------------------
# MoE Layer
# ---------------------------------------------------------------------------

class MoELayer(nn.Module):
    """Mixture of Experts with top-K routing and Switch Transformer load balancing loss."""

    def __init__(self, config: VesperConfig):
        super().__init__()
        self.num_experts = config.moe_num_experts
        self.top_k = config.moe_top_k
        self.aux_loss_weight = config.moe_aux_loss_weight

        self.router = nn.Linear(config.hidden_size, config.moe_num_experts)
        self.experts = nn.ModuleList([ExpertFFN(config) for _ in range(config.moe_num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden = x.shape
        num_tokens = batch_size * seq_len

        # 1. Router: softmax probabilities
        router_logits = self.router(x)  # [B, S, N]
        router_probs = F.softmax(router_logits, dim=-1)  # [B, S, N]

        # 2. Top-K routing
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)  # [B, S, K]

        # Normalize top-K weights
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # 3. Flatten to token-level
        x_flat = x.reshape(num_tokens, hidden)  # [T, H]
        tk_idx_flat = top_k_indices.reshape(num_tokens, self.top_k)  # [T, K]
        tk_w_flat = top_k_weights.reshape(num_tokens, self.top_k)  # [T, K]

        # 4. Token dispatch: each expert processes only its assigned tokens
        output_flat = torch.zeros_like(x_flat)

        for expert_id in range(self.num_experts):
            # Find tokens routed to this expert (across all K slots)
            expert_mask = (tk_idx_flat == expert_id)  # [T, K] bool
            if not expert_mask.any():
                continue

            # Get unique token indices routed to this expert
            token_mask = expert_mask.any(dim=-1)  # [T] bool
            token_indices = token_mask.nonzero(as_tuple=True)[0]  # [n_sel]

            x_expert = x_flat[token_indices]  # [n_sel, H]
            expert_out = self.experts[expert_id](x_expert)  # [n_sel, H]

            # Compute weight for each token-expert pair
            # Sum weights across K slots where this expert was selected
            weights = (expert_mask[token_indices].float() * tk_w_flat[token_indices]).sum(dim=-1)  # [n_sel]
            weighted_out = expert_out * weights.unsqueeze(-1)  # [n_sel, H]

            output_flat.index_add_(0, token_indices, weighted_out)

        output = output_flat.reshape(batch_size, seq_len, hidden)

        # 5. Load balancing loss
        aux_loss = self._load_balancing_loss(router_probs, top_k_indices)

        return output, aux_loss

    def _load_balancing_loss(
        self,
        router_probs: torch.Tensor,  # [B, S, N]
        top_k_indices: torch.Tensor,  # [B, S, K]
    ) -> torch.Tensor:
        """Switch Transformer auxiliary loss: N * Σ(f_i * P_i)."""
        batch_size, seq_len, _ = router_probs.shape
        total_assignments = batch_size * seq_len * self.top_k

        # P_i: mean router probability per expert
        p_mean = router_probs.mean(dim=(0, 1))  # [N]

        # f_i: fraction of assignments per expert
        fractions = []
        for expert_id in range(self.num_experts):
            count = (top_k_indices == expert_id).float().sum()
            fractions.append(count / total_assignments)
        f_tensor = torch.stack(fractions)  # [N]

        return self.num_experts * (f_tensor * p_mean).sum()


# ---------------------------------------------------------------------------
# Transformer Layer
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    """Pre-norm Transformer layer with residual connections."""

    def __init__(self, config: VesperConfig, use_moe: bool = False):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.use_moe = use_moe

        if use_moe:
            self.ffn = MoELayer(config)
        else:
            self.ffn = FeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Pre-norm attention + residual
        normed = self.attention_norm(hidden_states)
        attn_out = self.attention(normed, attention_mask)
        hidden_states = hidden_states + attn_out

        # Pre-norm FFN + residual
        normed = self.ffn_norm(hidden_states)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(normed)
        else:
            ffn_out = self.ffn(normed)
            aux_loss = None

        return hidden_states + ffn_out, aux_loss


# ---------------------------------------------------------------------------
# VesperLM
# ---------------------------------------------------------------------------

class VesperLM(nn.Module):
    """VesperLM Language Model.

    Architecture: Embedding → N × TransformerLayer → LayerNorm → LM Head
    """

    def __init__(self, config: VesperConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerLayer(config, use_moe=config.moe_enabled)
            for _ in range(config.num_layers)
        ])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embedding weights
        self.lm_head.weight = self.embeddings.weight

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Returns:
            (logits, aux_loss): logits [B, S, V], aux_loss scalar or None.
        """
        hidden_states = self.embeddings(input_ids)
        mask_4d = self._compute_mask_4d(attention_mask, hidden_states)

        total_aux_loss = None

        for layer in self.layers:
            if self.config.gradient_checkpointing and self.training:
                hidden_states, aux_loss = torch_checkpoint(
                    layer, hidden_states, mask_4d,
                    use_reentrant=False,
                )
            else:
                hidden_states, aux_loss = layer(hidden_states, mask_4d)

            if aux_loss is not None:
                total_aux_loss = aux_loss if total_aux_loss is None else total_aux_loss + aux_loss

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, total_aux_loss

    def _compute_mask_4d(
        self,
        attention_mask: torch.Tensor | None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Convert 2D padding mask [B, S] to 4D causal+padding mask [B, 1, S, S]."""
        if attention_mask is None:
            return None

        if attention_mask.dim() != 2:
            return attention_mask

        batch_size, seq_len = attention_mask.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        # Causal mask: [1, 1, S, S]
        causal = torch.triu(
            torch.full((seq_len, seq_len), -1e9, dtype=dtype, device=device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        # Padding mask: [B, 1, 1, S]
        padding = (1.0 - attention_mask.to(dtype)).unsqueeze(1).unsqueeze(2) * (-1e9)

        return causal + padding

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
