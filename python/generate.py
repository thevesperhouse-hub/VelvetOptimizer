#!/usr/bin/env python3
"""VesperLM Text Generation — load a checkpoint and generate text.

Usage:
    python generate.py --checkpoint checkpoints/best.pt --prompt "The future of AI"
    python generate.py --checkpoint checkpoints/best.pt --prompt "Once upon a time" --max-tokens 200 --temperature 0.8
    python generate.py --checkpoint checkpoints/best.pt --interactive
"""

import argparse
import sys

import torch
import torch.nn.functional as F

from vesper.config import VesperConfig
from vesper.model import VesperLM
from vesper.data import load_tokenizer


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[VesperLM, dict]:
    """Load VesperLM from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = ckpt["config"]
    config = VesperConfig(
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        max_position_embeddings=cfg["max_position_embeddings"],
        flylora_rank=cfg.get("flylora_rank", 16),
        flylora_alpha=cfg.get("flylora_alpha", 32.0),
        flylora_sparsity=cfg.get("flylora_sparsity", 0.25),
        era_gamma=cfg.get("era_gamma", 0.1),
        moe_enabled=cfg.get("moe_enabled", False),
        moe_num_experts=cfg.get("moe_num_experts", 8),
        moe_top_k=cfg.get("moe_top_k", 2),
    )

    model = VesperLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device=device, dtype=dtype)
    model.eval()

    meta = {"step": ckpt.get("step", "?"), "loss": ckpt.get("loss", "?")}
    return model, meta


@torch.no_grad()
def generate(
    model: VesperLM,
    tokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    device: torch.device = torch.device("cpu"),
) -> str:
    """Autoregressive generation with top-k, top-p, temperature, and repetition penalty."""
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)

    max_seq = model.config.max_position_embeddings
    generated = input_ids[0].tolist()

    for _ in range(max_tokens):
        # Truncate to max seq len
        if len(generated) >= max_seq:
            context = generated[-max_seq:]
        else:
            context = generated

        ids = torch.tensor([context], dtype=torch.long, device=device)
        logits, _ = model(ids)
        next_logits = logits[0, -1, :].float()

        # Repetition penalty
        if repetition_penalty != 1.0:
            for token_id in set(generated):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

        # Temperature
        if temperature > 0:
            next_logits = next_logits / temperature
        else:
            # Greedy
            next_token = next_logits.argmax().item()
            generated.append(next_token)
            continue

        # Top-K filtering
        if top_k > 0:
            top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            threshold = top_k_vals[-1]
            next_logits[next_logits < threshold] = float("-inf")

        # Top-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            cutoff_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[cutoff_mask] = float("-inf")
            next_logits = sorted_logits.scatter(0, sorted_indices, sorted_logits)

        # Sample
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)

        # EOS check
        eos_id = tokenizer.token_to_id("</s>") or tokenizer.token_to_id("<|endoftext|>")
        if eos_id is not None and next_token == eos_id:
            break

    return tokenizer.decode(generated)


def main():
    parser = argparse.ArgumentParser(description="VesperLM Text Generation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer name or path")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0=greedy)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K filtering")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode — type prompts in a loop")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Load
    print(f"Loading checkpoint: {args.checkpoint}")
    model, meta = load_model_from_checkpoint(args.checkpoint, device, dtype)
    tokenizer = load_tokenizer(args.tokenizer)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: {params:.0f}M params | step {meta['step']} | loss {meta['loss']}")
    print(f"  Device: {device} | Dtype: {args.dtype}")
    print(f"  Temperature: {args.temperature} | Top-K: {args.top_k} | Top-P: {args.top_p}")
    print()

    if args.interactive:
        print("Interactive mode. Type a prompt, press Enter. Ctrl+C to quit.\n")
        while True:
            try:
                prompt = input(">>> ")
                if not prompt.strip():
                    continue
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break

            output = generate(
                model, tokenizer, prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                device=device,
            )
            print(f"\n{output}\n")

    elif args.prompt:
        output = generate(
            model, tokenizer, args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        print(output)

    else:
        parser.error("Provide --prompt or --interactive")


if __name__ == "__main__":
    main()
