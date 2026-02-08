"""VesperLM Configuration — mirrors Rust vesper-core config exactly."""

from dataclasses import dataclass, field


@dataclass
class VesperConfig:
    # Architecture
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 4096

    # FlyLoRA
    flylora_rank: int = 16
    flylora_alpha: float = 32.0
    flylora_sparsity: float = 0.25

    # ERA activation
    era_gamma: float = 0.1

    # Training
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6

    # Attention
    use_flash_attn: bool = True
    rope_theta: float = 10000.0

    # MoE
    moe_enabled: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_aux_loss_weight: float = 0.01

    # Gradient checkpointing
    gradient_checkpointing: bool = False

    @staticmethod
    def tiny():
        return VesperConfig(hidden_size=256, num_layers=6, num_heads=4, intermediate_size=1024)

    @staticmethod
    def small():
        return VesperConfig(hidden_size=512, num_layers=8, num_heads=8, intermediate_size=2048)

    @staticmethod
    def medium():
        return VesperConfig(hidden_size=768, num_layers=12, num_heads=12, intermediate_size=3072)

    @staticmethod
    def large():
        return VesperConfig(hidden_size=1024, num_layers=24, num_heads=16, intermediate_size=4096)

    @staticmethod
    def xlarge():
        return VesperConfig(
            hidden_size=2048, num_layers=24, num_heads=16,
            intermediate_size=5504, max_position_embeddings=4096,
        )

    def with_moe(self, num_experts: int = 8, top_k: int = 2):
        self.moe_enabled = True
        self.moe_num_experts = num_experts
        self.moe_top_k = top_k
        return self

    def total_params(self) -> int:
        H, I, V, L = self.hidden_size, self.intermediate_size, self.vocab_size, self.num_layers
        R = self.flylora_rank

        emb = V * H  # embeddings (tied with lm_head)
        # Attention: Q, K, V, O — plain nn.Linear (no FlyLoRA)
        attn = L * 4 * (H * H + H)  # weight + bias
        # FFN: gate_proj + up_proj (FlyLoRA) + down_proj (plain)
        flylora_per = (H * I + I) + (H * R + R) + (R * I + I)  # base + lora_down + lora_up
        ffn = L * (2 * flylora_per + (I * H + H))  # gate + up (FlyLoRA) + down (plain)

        if self.moe_enabled:
            router = L * (H * self.moe_num_experts + self.moe_num_experts)
            expert_ffn = L * self.moe_num_experts * (2 * flylora_per + (I * H + H))
            ffn = router + expert_ffn

        norms = L * 2 * 2 * H + 2 * H  # layer norms (weight + bias) + final norm
        return emb + attn + ffn + norms

    def validate(self):
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        if self.moe_enabled:
            assert 1 <= self.moe_top_k <= self.moe_num_experts
            assert self.moe_num_experts >= 2
