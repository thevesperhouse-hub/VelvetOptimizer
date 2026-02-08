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
        emb = self.vocab_size * self.hidden_size
        attn = self.num_layers * (3 * self.hidden_size * (self.flylora_rank * 2) + self.hidden_size ** 2)
        ffn_per_expert = 2 * self.hidden_size * self.intermediate_size
        if self.moe_enabled:
            router = self.hidden_size * self.moe_num_experts
            ffn = self.num_layers * (router + self.moe_num_experts * ffn_per_expert)
        else:
            ffn = self.num_layers * ffn_per_expert
        norms = self.num_layers * 2 * self.hidden_size
        return emb + attn + ffn + norms

    def validate(self):
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"
        if self.moe_enabled:
            assert 1 <= self.moe_top_k <= self.moe_num_experts
            assert self.moe_num_experts >= 2
