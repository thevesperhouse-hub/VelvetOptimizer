#!/usr/bin/env python3
"""Download FineWeb-Edu subset and export as plain text for Vesper training.

Usage:
    # Download ~1B tokens (good for xlarge/1B model benchmark)
    python download_fineweb.py --tokens 1B --output /workspace/fineweb-1B.txt

    # Download ~100M tokens (good for small/medium model benchmark)
    python download_fineweb.py --tokens 100M --output /workspace/fineweb-100M.txt

    # Download ~10M tokens (quick test)
    python download_fineweb.py --tokens 10M --output /workspace/fineweb-10M.txt

Requires: pip install datasets
"""

import argparse
import sys

def parse_token_count(s: str) -> int:
    """Parse '100M', '1B', '10M' etc."""
    s = s.strip().upper()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith("M"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("K"):
        return int(float(s[:-1]) * 1_000)
    else:
        return int(s)

def main():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu for Vesper benchmarks")
    parser.add_argument("--tokens", type=str, default="100M",
                        help="Approximate token count to download: 10M, 100M, 1B (default: 100M)")
    parser.add_argument("--output", type=str, default="fineweb-edu.txt",
                        help="Output text file path")
    parser.add_argument("--config", type=str, default="sample-10BT",
                        help="FineWeb-Edu config: sample-10BT, sample-100BT, CC-MAIN-2024-10, etc.")
    args = parser.parse_args()

    target_tokens = parse_token_count(args.tokens)
    # Rough estimate: 1 token ~= 4 chars in English
    target_chars = target_tokens * 4

    print(f"Downloading FineWeb-Edu ({args.config})")
    print(f"Target: ~{target_tokens:,} tokens (~{target_chars / 1e9:.1f} GB text)")
    print(f"Output: {args.output}")
    print()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    # Stream the dataset (no full download needed)
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.config,
        split="train",
        streaming=True,
    )

    total_chars = 0
    doc_count = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for sample in ds:
            text = sample.get("text", "")
            if not text or len(text) < 50:
                continue

            f.write(text)
            f.write("\n")

            total_chars += len(text)
            doc_count += 1

            if doc_count % 10_000 == 0:
                approx_tokens = total_chars // 4
                pct = min(100.0, approx_tokens / target_tokens * 100)
                print(f"  {doc_count:,} docs | ~{approx_tokens:,} tokens ({pct:.1f}%)", end="\r")

            if total_chars >= target_chars:
                break

    approx_tokens = total_chars // 4
    file_size_mb = total_chars / (1024 * 1024)

    print(f"\nDone!")
    print(f"  Documents: {doc_count:,}")
    print(f"  Approx tokens: {approx_tokens:,}")
    print(f"  File size: {file_size_mb:.0f} MB")
    print(f"  Saved to: {args.output}")

if __name__ == "__main__":
    main()
