#!/usr/bin/env python3
"""VesperLM Training — Complete training pipeline with Velvet optimizer.

Usage:
    # Basic training
    python train.py --dataset data.txt --model-size medium

    # Large-scale with Flash Attn + gradient checkpointing
    python train.py --dataset data.txt --model-size xlarge --dtype bf16 \
        --gradient-checkpointing --batch-size 16 --grad-accum 4

    # MoE training
    python train.py --dataset data.txt --model-size large --moe --num-experts 8 --top-k 2

    # Streaming mode for large datasets
    python train.py --dataset data.txt --model-size medium --data-mode streaming

    # Resume from checkpoint with wandb
    python train.py --dataset data.txt --model-size medium --resume checkpoints/step_1000.pt --wandb

    # With validation
    python train.py --dataset data.txt --val-dataset val.txt --model-size medium --eval-every 500
"""

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from vesper.config import VesperConfig
from vesper.model import VesperLM
from vesper.optimizer import VelvetOptimizer
from vesper.data import load_tokenizer, create_dataloader
from vesper.kernels import get_backend

try:
    from vesper.kernels import cross_entropy_kernel
except ImportError:
    cross_entropy_kernel = None

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Signal handling for graceful shutdown
# ---------------------------------------------------------------------------

_SHOULD_STOP = False

def _signal_handler(signum, frame):
    global _SHOULD_STOP
    if _SHOULD_STOP:
        print("\nForce quit.")
        sys.exit(1)
    print("\nGraceful shutdown requested. Saving checkpoint...")
    _SHOULD_STOP = True

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    aux_loss: torch.Tensor | None = None,
    aux_loss_weight: float = 0.01,
) -> torch.Tensor:
    """Compute cross-entropy loss, optionally with fused Triton kernel."""
    # Use fused cross-entropy for large vocab on CUDA
    if cross_entropy_kernel is not None and logits.is_cuda and vocab_size > 8192:
        loss = cross_entropy_kernel(logits, labels)
    else:
        loss = F.cross_entropy(
            logits.view(-1, vocab_size).float(),
            labels.view(-1),
            reduction="mean",
        )

    if aux_loss is not None:
        loss = loss + aux_loss_weight * aux_loss

    return loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: VesperLM,
    dataloader,
    config: VesperConfig,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 50,
) -> dict:
    """Run evaluation, return metrics dict."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for input_ids, labels in dataloader:
        if num_batches >= max_batches:
            break

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32 and device.type == "cuda")):
            logits, aux_loss = model(input_ids)
            loss = compute_loss(logits, labels, config.vocab_size)

        batch_tokens = input_ids.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        num_batches += 1

    model.train()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return {"val_loss": avg_loss, "val_ppl": ppl, "val_batches": num_batches}


# ---------------------------------------------------------------------------
# Checkpoint saving / loading
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: VesperLM,
    optimizer: VelvetOptimizer,
    step: int,
    loss: float,
    config: VesperConfig,
    path: str,
    lightweight: bool = True,
):
    """Save training checkpoint.
    lightweight=True: model weights only (~2.6GB for 1.3B). Default.
    lightweight=False: includes optimizer state (~13GB). For exact resume.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "loss": loss,
        "config": {
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
            "flylora_rank": config.flylora_rank,
            "flylora_alpha": config.flylora_alpha,
            "flylora_sparsity": config.flylora_sparsity,
            "era_gamma": config.era_gamma,
            "moe_enabled": config.moe_enabled,
            "moe_num_experts": config.moe_num_experts,
            "moe_top_k": config.moe_top_k,
        },
    }
    if not lightweight:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: VesperLM,
    optimizer: VelvetOptimizer | None = None,
    device: torch.device | None = None,
) -> dict:
    """Load a training checkpoint. Returns metadata dict."""
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


# ---------------------------------------------------------------------------
# Learning rate scheduler (cosine with warmup)
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine LR schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_lr(optimizer: VelvetOptimizer, lr: float):
    """Set learning rate on all param groups."""
    for group in optimizer.param_groups:
        group["lr"] = lr


# ---------------------------------------------------------------------------
# Safe data iterator (catches worker timeouts)
# ---------------------------------------------------------------------------

def _safe_data_iter(data_iter, dataloader, pbar):
    """Wrap data iterator to catch worker timeouts and recover."""
    while True:
        try:
            batch = next(data_iter)
            yield batch
        except StopIteration:
            return
        except Exception as e:
            err_msg = str(e).lower()
            if "timeout" in err_msg or "dataloader" in err_msg or "worker" in err_msg:
                from tqdm import tqdm as _tqdm
                _tqdm.write(f"  [WARN] DataLoader worker timeout — restarting data pipeline")
                try:
                    data_iter = iter(dataloader)
                    continue
                except Exception:
                    _tqdm.write(f"  [ERROR] DataLoader restart failed — ending epoch")
                    return
            raise


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    global _SHOULD_STOP

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")

    # Mixed precision dtype
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(args.dtype, torch.float32)

    # Config
    config_map = {
        "tiny": VesperConfig.tiny,
        "small": VesperConfig.small,
        "medium": VesperConfig.medium,
        "large": VesperConfig.large,
        "xlarge": VesperConfig.xlarge,
    }
    config = config_map[args.model_size]()

    if args.moe:
        config.with_moe(num_experts=args.num_experts, top_k=args.top_k)
    if args.gradient_checkpointing:
        config.gradient_checkpointing = True
    if args.vocab_size:
        config.vocab_size = args.vocab_size
    if args.seq_len:
        config.max_position_embeddings = args.seq_len

    config.use_flash_attn = not args.no_flash_attn
    config.validate()

    # Effective batch size = batch_size * grad_accum
    effective_batch = args.batch_size * args.grad_accum

    # Print config
    print(f"{'='*60}")
    print(f"VesperLM Training")
    print(f"{'='*60}")
    print(f"  Model size:      {args.model_size} ({config.total_params()/1e6:.1f}M params)")
    print(f"  Hidden:          {config.hidden_size}")
    print(f"  Layers:          {config.num_layers}")
    print(f"  Heads:           {config.num_heads}")
    print(f"  FFN:             {config.intermediate_size}")
    print(f"  Vocab:           {config.vocab_size}")
    print(f"  Seq len:         {config.max_position_embeddings}")
    if config.moe_enabled:
        print(f"  MoE:             {config.moe_num_experts} experts, top-{config.moe_top_k}")
    else:
        print(f"  MoE:             disabled")
    print(f"  FlyLoRA:         rank={config.flylora_rank}, alpha={config.flylora_alpha}")
    print(f"  ERA gamma:       {config.era_gamma}")
    print(f"  Dtype:           {args.dtype}")
    print(f"  Device:          {device}")
    print(f"  Flash Attn:      {config.use_flash_attn}")
    print(f"  Grad checkpoint: {config.gradient_checkpointing}")
    print(f"  Kernel backend:  {get_backend()}")
    print(f"  Batch size:      {args.batch_size} (effective: {effective_batch} with {args.grad_accum}x accum)")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Max steps:       {args.max_steps}")
    if args.wandb:
        print(f"  W&B:             enabled (project={args.wandb_project})")
    print(f"{'='*60}")

    # Tokenizer
    tokenizer = load_tokenizer(args.tokenizer)
    actual_vocab = tokenizer.get_vocab_size()
    if config.vocab_size != actual_vocab:
        print(f"  Adjusting vocab_size from {config.vocab_size} to {actual_vocab}")
        config.vocab_size = actual_vocab

    # Model
    model = VesperLM(config)
    model = model.to(device=device, dtype=dtype)
    trainable = model.num_parameters()
    print(f"  Trainable params: {trainable/1e6:.1f}M")

    # Compile model for speed (PyTorch 2.x)
    # NOTE: torch.compile conflicts with gradient checkpointing (bypasses recompute)
    if not args.no_compile and hasattr(torch, "compile") and not config.gradient_checkpointing:
        print("  Compiling model with torch.compile()...")
        model = torch.compile(model)
    elif config.gradient_checkpointing and not args.no_compile:
        print("  Skipping torch.compile (incompatible with gradient checkpointing)")

    # Optimizer
    optimizer = VelvetOptimizer(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        entropy_adaptive=not args.no_entropy_adaptive,
        perplexity_guided=not args.no_perplexity_guided,
    )
    print(f"  Optimizer kernel: {optimizer.kernel_backend}")

    # AMP scaler for fp16
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16 and device.type == "cuda"))

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_step = ckpt.get("step", 0)
        print(f"  Resumed from step {start_step}, loss={ckpt.get('loss', '?'):.4f}")

    # DataLoader
    seq_len = args.seq_len or config.max_position_embeddings
    dataloader = create_dataloader(
        dataset_path=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=seq_len,
        mode=args.data_mode,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )

    # Validation DataLoader
    val_dataloader = None
    if args.val_dataset:
        val_dataloader = create_dataloader(
            dataset_path=args.val_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            seq_len=seq_len,
            mode="inmemory",
            shuffle=False,
        )

    # Wandb
    if args.wandb and HAS_WANDB:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or f"vesper-{args.model_size}-{args.dtype}",
            config={
                "model_size": args.model_size,
                "hidden_size": config.hidden_size,
                "num_layers": config.num_layers,
                "num_heads": config.num_heads,
                "vocab_size": config.vocab_size,
                "seq_len": seq_len,
                "batch_size": args.batch_size,
                "effective_batch_size": effective_batch,
                "lr": args.lr,
                "dtype": args.dtype,
                "moe": config.moe_enabled,
                "gradient_checkpointing": config.gradient_checkpointing,
                "trainable_params": trainable,
                "kernel_backend": get_backend(),
                "optimizer_kernel": optimizer.kernel_backend,
            },
        )
    elif args.wandb and not HAS_WANDB:
        print("  WARNING: --wandb requested but wandb not installed. pip install wandb")

    # Training
    model.train()
    step = start_step
    epoch = 0
    total_tokens = 0
    best_loss = float("inf")
    start_time = time.time()
    accum_loss = 0.0
    micro_step = 0

    print(f"\nStarting training from step {step}...\n")

    pbar = tqdm(
        total=args.max_steps,
        initial=step,
        desc="Training",
        unit="step",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    while step < args.max_steps and not _SHOULD_STOP:
        epoch += 1

        try:
            data_iter = iter(dataloader)
        except Exception as e:
            tqdm.write(f"  [WARN] DataLoader restart failed: {e}")
            break

        for input_ids, labels in _safe_data_iter(data_iter, dataloader, pbar):
            if step >= args.max_steps or _SHOULD_STOP:
                break

            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Forward + loss
            with torch.amp.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32 and device.type == "cuda")):
                logits, aux_loss = model(input_ids)
                loss = compute_loss(
                    logits, labels, config.vocab_size,
                    aux_loss=aux_loss,
                    aux_loss_weight=config.moe_aux_loss_weight,
                )
                # Scale loss for gradient accumulation
                loss = loss / args.grad_accum

            # Backward (accumulate gradients)
            scaler.scale(loss).backward()
            loss_item = loss.item() * args.grad_accum
            micro_step += 1
            total_tokens += input_ids.numel()

            # NaN detection — skip this micro-step entirely
            if not math.isfinite(loss_item):
                nan_count = getattr(train, '_nan_count', 0) + 1
                train._nan_count = nan_count
                optimizer.zero_grad()
                micro_step = 0  # reset accumulation
                accum_loss = 0.0
                if nan_count <= 3:
                    tqdm.write(f"  [WARN] NaN loss at micro-step — skipping (#{nan_count})")
                elif nan_count == 10:
                    tqdm.write(f"  [ERROR] 10 NaN losses — training is diverging. Try lower --lr")
                continue

            accum_loss += loss_item

            # Only step optimizer every grad_accum micro-steps
            if micro_step % args.grad_accum != 0:
                continue

            # LR schedule (applied at optimizer step, not micro-step)
            lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.lr * 0.1)
            set_lr(optimizer, lr)

            # Gradient clipping + step
            scaler.unscale_(optimizer)
            grad_norm = optimizer.clip_grad_norm_()

            # Skip step if gradients are NaN (scaler handles this for fp16)
            if not math.isfinite(grad_norm):
                optimizer.zero_grad()
                accum_loss = 0.0
                tqdm.write(f"  [WARN] NaN gradients at step {step} — skipping optimizer step")
                continue

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update adaptive signals
            loss_val = accum_loss / args.grad_accum if args.grad_accum > 1 else accum_loss
            optimizer.set_loss_metrics(loss_val, config.vocab_size)

            step += 1
            accum_loss = 0.0

            # Update tqdm every step
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            ppl = math.exp(min(loss_val, 20.0))
            postfix = {
                "loss": f"{loss_val:.3f}",
                "ppl": f"{ppl:.0f}",
                "tok/s": f"{tokens_per_sec:.0f}",
            }
            if device.type == "cuda":
                mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
                postfix["VRAM"] = f"{mem_gb:.0f}G"
            pbar.set_postfix(postfix)
            pbar.update(1)

            # Detailed logging at intervals
            if step % args.log_every == 0:
                aux_str = ""
                if aux_loss is not None:
                    aux_str = f"  aux={aux_loss.item():.4f}"

                log_line = (
                    f"step {step:>6d} | loss {loss_val:.4f} | ppl {ppl:.1f}{aux_str} "
                    f"| lr {lr:.2e} | eff_lr {optimizer.effective_lr:.2e} "
                    f"| gnorm {grad_norm:.3f} | tok/s {tokens_per_sec:.0f}"
                )

                if device.type == "cuda":
                    log_line += f" | VRAM {mem_gb:.1f}GB"

                tqdm.write(log_line)

                # Wandb logging
                if args.wandb and HAS_WANDB:
                    log_dict = {
                        "train/loss": loss_val,
                        "train/ppl": ppl,
                        "train/lr": lr,
                        "train/effective_lr": optimizer.effective_lr,
                        "train/grad_norm": grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/epoch": epoch,
                    }
                    if aux_loss is not None:
                        log_dict["train/aux_loss"] = aux_loss.item()
                    if device.type == "cuda":
                        log_dict["train/vram_gb"] = mem_gb
                    wandb.log(log_dict, step=step)

            # Evaluation
            if val_dataloader is not None and args.eval_every > 0 and step % args.eval_every == 0:
                val_metrics = evaluate(model, val_dataloader, config, device, dtype)
                tqdm.write(
                    f"  [EVAL] step {step} | val_loss {val_metrics['val_loss']:.4f} "
                    f"| val_ppl {val_metrics['val_ppl']:.1f} ({val_metrics['val_batches']} batches)"
                )
                if args.wandb and HAS_WANDB:
                    wandb.log(val_metrics, step=step)

            # Checkpoint
            if step % args.save_every == 0:
                if not math.isfinite(loss_val):
                    tqdm.write(f"  [WARN] Skipping checkpoint at step {step} — loss is NaN")
                else:
                    ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}.pt")
                    save_checkpoint(model, optimizer, step, loss_val, config, ckpt_path)
                    tqdm.write(f"  Saved checkpoint: {ckpt_path}")

                    # Rotate: delete checkpoint from 2 saves ago (keep last 2 + best)
                    old_step = step - 2 * args.save_every
                    if old_step > 0:
                        old_path = os.path.join(args.checkpoint_dir, f"step_{old_step}.pt")
                        if os.path.exists(old_path):
                            os.remove(old_path)

                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_path = os.path.join(args.checkpoint_dir, "best.pt")
                        save_checkpoint(model, optimizer, step, loss_val, config, best_path)
                        tqdm.write(f"  New best: loss={loss_val:.4f}")

    # Close progress bar
    pbar.close()

    # Final checkpoint
    elapsed = time.time() - start_time
    print(f"\nTraining complete: {step} steps in {elapsed:.1f}s")
    final_path = os.path.join(args.checkpoint_dir, f"final_step_{step}.pt")
    save_checkpoint(model, optimizer, step, loss_val, config, final_path, lightweight=False)
    print(f"Saved final checkpoint (full, with optimizer state): {final_path}")

    if device.type == "cuda":
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / (1024**3):.1f}GB")

    # Final eval
    if val_dataloader is not None:
        val_metrics = evaluate(model, val_dataloader, config, device, dtype)
        print(f"Final eval: val_loss={val_metrics['val_loss']:.4f}, val_ppl={val_metrics['val_ppl']:.1f}")

    if args.wandb and HAS_WANDB:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VesperLM Training")

    # Data
    parser.add_argument("--dataset", type=str, required=True, help="Path to training data (.txt or .jsonl)")
    parser.add_argument("--val-dataset", type=str, default=None, help="Path to validation data")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer path or HF name")
    parser.add_argument("--data-mode", choices=["inmemory", "streaming", "cached", "jsonl"], default="inmemory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for cached mode")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Model
    parser.add_argument("--model-size", choices=["tiny", "small", "medium", "large", "xlarge"], default="medium")
    parser.add_argument("--vocab-size", type=int, default=None, help="Override vocab size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length (default: from config)")

    # MoE
    parser.add_argument("--moe", action="store_true", help="Enable Mixture of Experts")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)

    # Training
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-flash-attn", action="store_true", help="Disable Flash Attention")
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile()")

    # Optimizer
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--no-entropy-adaptive", action="store_true")
    parser.add_argument("--no-perplexity-guided", action="store_true")

    # Logging / Checkpointing
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate every N steps (0=disabled)")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")

    # Wandb
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="vesper-training")
    parser.add_argument("--wandb-run", type=str, default=None, help="W&B run name")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
