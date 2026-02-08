"""Dataset loading — text, JSONL, and streaming modes.

Mirrors the Rust vesper-cli data loading:
  - In-memory: load entire dataset, tokenize, chunk into seq_len windows
  - Streaming: read line-by-line, yield batches on the fly
  - Cached: tokenize once, save to disk, memory-map on subsequent runs
"""

import os
import json
import random
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from tokenizers import Tokenizer


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_tokenizer(path: str) -> Tokenizer:
    """Load a HuggingFace tokenizer from file or pretrained name."""
    if os.path.isfile(path):
        return Tokenizer.from_file(path)
    return Tokenizer.from_pretrained(path)


def tokenize_text(tokenizer: Tokenizer, text: str) -> list[int]:
    """Tokenize a string, returning token IDs."""
    return tokenizer.encode(text).ids


# ---------------------------------------------------------------------------
# In-Memory Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Load a text file, tokenize, and chunk into fixed-length windows.

    Each sample is (input_ids, labels) where labels = input_ids shifted by 1.
    """

    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        seq_len: int = 512,
        stride: int | None = None,
    ):
        self.seq_len = seq_len
        self.stride = stride or seq_len

        # Read and tokenize
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        self.token_ids = tokenize_text(tokenizer, text)

        # Build windows
        self.windows = []
        for start in range(0, len(self.token_ids) - seq_len, self.stride):
            self.windows.append(start)

        if len(self.windows) == 0 and len(self.token_ids) > seq_len:
            self.windows.append(0)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.windows[idx]
        chunk = self.token_ids[start : start + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, labels


class JSONLDataset(Dataset):
    """JSONL dataset where each line has a "text" field."""

    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        seq_len: int = 512,
        text_field: str = "text",
    ):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Read all lines, tokenize, concatenate
        all_ids = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(text_field, "")
                if text:
                    ids = tokenize_text(tokenizer, text)
                    all_ids.extend(ids)

        self.token_ids = all_ids

        # Build windows
        self.windows = []
        for start in range(0, len(self.token_ids) - seq_len, seq_len):
            self.windows.append(start)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.windows[idx]
        chunk = self.token_ids[start : start + self.seq_len + 1]
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        labels = torch.tensor(chunk[1:], dtype=torch.long)
        return input_ids, labels


# ---------------------------------------------------------------------------
# Streaming Dataset
# ---------------------------------------------------------------------------

class StreamingTextDataset(IterableDataset):
    """Stream a large text file line-by-line, yielding chunks without loading all into memory."""

    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        seq_len: int = 512,
        shuffle_buffer: int = 1000,
    ):
        self.path = path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        # Multi-worker sharding: each worker reads every Nth line
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        buffer = []
        token_buffer = []
        line_idx = 0

        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                # Shard lines across workers (round-robin)
                if line_idx % num_workers != worker_id:
                    line_idx += 1
                    continue
                line_idx += 1

                line = line.strip()
                if not line:
                    continue

                # Handle JSONL
                if line.startswith("{"):
                    try:
                        obj = json.loads(line)
                        line = obj.get("text", line)
                    except json.JSONDecodeError:
                        pass

                ids = tokenize_text(self.tokenizer, line)
                token_buffer.extend(ids)

                # Yield chunks from buffer
                while len(token_buffer) > self.seq_len:
                    chunk = token_buffer[: self.seq_len + 1]
                    token_buffer = token_buffer[self.seq_len :]

                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    labels = torch.tensor(chunk[1:], dtype=torch.long)
                    buffer.append((input_ids, labels))

                    # Shuffle buffer
                    if len(buffer) >= self.shuffle_buffer:
                        random.shuffle(buffer)
                        while buffer:
                            yield buffer.pop()

        # Flush remaining
        if buffer:
            random.shuffle(buffer)
            for item in buffer:
                yield item


# ---------------------------------------------------------------------------
# Cached Dataset (tokenize once, memory-map)
# ---------------------------------------------------------------------------

class CachedDataset(Dataset):
    """Tokenize once, save as .pt, memory-map on subsequent runs."""

    def __init__(
        self,
        path: str,
        tokenizer: Tokenizer,
        seq_len: int = 512,
        cache_dir: str | None = None,
    ):
        self.seq_len = seq_len

        # Determine cache path
        if cache_dir is None:
            cache_dir = os.path.dirname(path)
        cache_name = os.path.basename(path) + f".seq{seq_len}.pt"
        cache_path = os.path.join(cache_dir, cache_name)

        if os.path.exists(cache_path):
            self.token_ids = torch.load(cache_path, weights_only=True)
        else:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            ids = tokenize_text(tokenizer, text)
            self.token_ids = torch.tensor(ids, dtype=torch.long)
            torch.save(self.token_ids, cache_path)

        self.num_samples = max(0, (len(self.token_ids) - 1) // seq_len)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len + 1
        chunk = self.token_ids[start:end]
        return chunk[:-1], chunk[1:]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloader(
    dataset_path: str,
    tokenizer: Tokenizer,
    batch_size: int = 8,
    seq_len: int = 512,
    mode: str = "inmemory",
    num_workers: int = 0,
    shuffle: bool = True,
    cache_dir: str | None = None,
) -> DataLoader:
    """Create a DataLoader from a dataset file.

    Args:
        mode: "inmemory", "streaming", "cached", or "jsonl"
    """
    # timeout=60s on workers — raises instead of hanging forever on deadlock
    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs = dict(
            timeout=60,
            persistent_workers=True,
            prefetch_factor=2,
        )

    if mode == "streaming":
        dataset = StreamingTextDataset(dataset_path, tokenizer, seq_len)
        return DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers,
            pin_memory=True, **worker_kwargs,
        )
    elif mode == "jsonl":
        dataset = JSONLDataset(dataset_path, tokenizer, seq_len)
    elif mode == "cached":
        dataset = CachedDataset(dataset_path, tokenizer, seq_len, cache_dir)
    else:
        dataset = TextDataset(dataset_path, tokenizer, seq_len)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        **worker_kwargs,
    )
