"""Velvet Optimizer — AdamW + entropy-adaptive LR + perplexity-guided momentum.

Three backends (auto-detected, priority order):
  1. Triton fused kernel (default on CUDA) — single kernel per param
  2. Native CUDA kernel (fallback) — compiled via cpp_extension
  3. Pure PyTorch (CPU or no Triton/CUDA)
"""

import math
import torch
from torch.optim import Optimizer

from .kernels import velvet_update_kernel, HAS_TRITON, HAS_CUDA_EXT


class VelvetOptimizer(Optimizer):
    """Velvet: AdamW with entropy-adaptive LR and perplexity-guided momentum.

    Args:
        params: model parameters
        lr: learning rate (default: 5e-4)
        betas: coefficients for moment estimation (default: (0.9, 0.999))
        eps: term for numerical stability (default: 1e-8)
        weight_decay: decoupled weight decay (default: 1e-3)
        max_grad_norm: global gradient clipping (0 = disabled, default: 1.0)
        entropy_adaptive: scale LR by entropy estimate (default: True)
        perplexity_guided: scale beta1 by perplexity (default: True)
        sparse_aware: skip near-zero weights (default: True)
    """

    def __init__(
        self,
        params,
        lr: float = 5e-4,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-3,
        max_grad_norm: float = 1.0,
        entropy_adaptive: bool = True,
        perplexity_guided: bool = True,
        sparse_aware: bool = True,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            entropy_adaptive=entropy_adaptive,
            perplexity_guided=perplexity_guided,
            sparse_aware=sparse_aware,
        )
        super().__init__(params, defaults)

        self._entropy_scale = 1.0
        self._perplexity_scale = 1.0
        self._last_grad_norm = 0.0
        self._global_step = 0

        # Auto-detect best kernel backend
        if HAS_TRITON and torch.cuda.is_available():
            self._kernel_backend = "triton"
        elif HAS_CUDA_EXT and torch.cuda.is_available():
            self._kernel_backend = "cuda"
        else:
            self._kernel_backend = "pytorch"

    # ---- Adaptive signals (set by training loop) ----

    def set_loss_metrics(self, loss_val: float, vocab_size: int):
        """Update entropy/perplexity scales from current loss."""
        max_entropy = math.log(vocab_size)
        approx_entropy = min(loss_val, max_entropy) / max_entropy
        self._entropy_scale = max(0.5, min(2.0, approx_entropy / 0.5))

        ppl = math.exp(min(loss_val, 20.0))  # clamp to avoid overflow
        self._perplexity_scale = max(0.5, min(2.0, 40.0 / max(ppl, 1.0)))

    @property
    def effective_lr(self) -> float:
        lr = self.defaults["lr"]
        if self.defaults["entropy_adaptive"]:
            lr *= self._entropy_scale
        return lr

    @property
    def effective_beta1(self) -> float:
        beta1 = self.defaults["betas"][0]
        if self.defaults["perplexity_guided"]:
            beta1 = max(0.5, min(0.999, beta1 * self._perplexity_scale))
        return beta1

    @property
    def last_grad_norm(self) -> float:
        return self._last_grad_norm

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def kernel_backend(self) -> str:
        return self._kernel_backend

    # ---- Gradient clipping ----

    def clip_grad_norm_(self) -> float:
        """Global gradient clipping. Returns the global norm."""
        max_norm = self.defaults["max_grad_norm"]
        if max_norm <= 0:
            return 0.0

        all_params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    all_params.append(p)

        if not all_params:
            return 0.0

        total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm)
        self._last_grad_norm = total_norm.item()
        return self._last_grad_norm

    # ---- Step ----

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._global_step += 1

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]

            # Adaptive overrides
            if group["entropy_adaptive"]:
                lr *= self._entropy_scale
            eff_beta1 = beta1
            if group["perplexity_guided"]:
                eff_beta1 = max(0.5, min(0.999, beta1 * self._perplexity_scale))

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Velvet does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p, dtype=torch.float32)
                    state["v"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] += 1

                bc1 = 1.0 - eff_beta1 ** state["step"]
                bc2 = 1.0 - beta2 ** state["step"]

                # GPU kernel path (Triton or CUDA)
                if self._kernel_backend != "pytorch" and p.is_cuda and velvet_update_kernel is not None:
                    velvet_update_kernel(
                        param=p.data,
                        grad=grad,
                        m=state["m"],
                        v=state["v"],
                        lr=lr,
                        beta1=eff_beta1,
                        beta2=beta2,
                        eps=eps,
                        wd=wd,
                        bias_correction1=bc1,
                        bias_correction2=bc2,
                        entropy_adaptive=group["entropy_adaptive"],
                        entropy_lr_scale=self._entropy_scale,
                        perplexity_guided=group["perplexity_guided"],
                        ppl_momentum_scale=self._perplexity_scale,
                        sparse_aware=group["sparse_aware"],
                    )
                else:
                    # Pure PyTorch fallback
                    p_f32 = p.data.float()
                    g_f32 = grad.float()

                    # Decoupled weight decay
                    p_f32.mul_(1.0 - lr * wd)

                    # Moments
                    state["m"].mul_(eff_beta1).add_(g_f32, alpha=1.0 - eff_beta1)
                    state["v"].mul_(beta2).addcmul_(g_f32, g_f32, value=1.0 - beta2)

                    # Bias-corrected
                    m_hat = state["m"] / bc1
                    v_hat = state["v"] / bc2

                    # Update
                    update = m_hat / (v_hat.sqrt() + eps)
                    p_f32.add_(update, alpha=-lr)
                    p.data.copy_(p_f32.to(p.dtype))

        return loss
