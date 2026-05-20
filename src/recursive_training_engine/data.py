from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import os
from pathlib import Path

import torch

from recursive_training_engine.config import DataConfig, TrainingConfig
from recursive_training_engine.utils import batch_iterator


@dataclass(slots=True)
class TokenStreams:
    train: torch.Tensor
    eval: torch.Tensor
    tokenizer_name: str
    data_fingerprint: str

    def train_batches(self, training: TrainingConfig):
        return batch_iterator(self.train, training.batch_size, training.seq_len)

    def eval_batches(self, training: TrainingConfig):
        return batch_iterator(self.eval, training.batch_size, training.seq_len)


def _synthetic_tokens(vocab_size: int, n: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    base = torch.arange(n, dtype=torch.long)
    noise = torch.randint(0, max(vocab_size, 2), (n,), generator=gen)
    return (base * 17 + noise * 3 + (base // 11)) % vocab_size


def _encode_with_gpt2_bpe(texts: list[str], cache_dir: Path) -> tuple[torch.Tensor, str]:
    try:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_pretrained("gpt2")
        ids: list[int] = []
        for text in texts:
            ids.extend(tokenizer.encode(text).ids)
        return torch.tensor(ids, dtype=torch.long), "gpt2_bpe"
    except Exception:
        from tokenizers import ByteLevelBPETokenizer

        cache_dir.mkdir(parents=True, exist_ok=True)
        corpus = cache_dir / "fallback_corpus.txt"
        corpus.write_text("\n".join(texts))
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=[str(corpus)], vocab_size=2048, min_frequency=1)
        ids = tokenizer.encode("\n".join(texts)).ids
        return torch.tensor(ids, dtype=torch.long), "bytelevel_bpe_fallback"


def _load_texts(config: DataConfig) -> list[str]:
    if config.dataset == "local":
        if config.local_text_path is None:
            raise ValueError("local_text_path is required for local dataset")
        return [Path(config.local_text_path).read_text()]
    try:
        from datasets import load_dataset

        if config.dataset == "tinystories":
            ds = load_dataset("roneneldan/TinyStories", split=config.train_split, streaming=True)
            sample_count = max(512, min(100_000, config.max_tokens // 96 + 2_048))
            return [str(x["text"]) for x in islice(ds, sample_count)]
        if config.dataset == "wikitext103":
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            sample_count = max(1024, min(100_000, config.max_tokens // 80 + 2_048))
            return [str(x["text"]) for x in islice(ds, sample_count)]
    except Exception:
        if os.environ.get("RTE_STRICT_DATASET", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            raise
        return [
            "Once upon a time, a small model learned to predict the next token.",
            "The training stream is deterministic, packed, and shared by every run.",
            "Recursive exact execution remains the source of truth for audits.",
        ] * 200
    return [
        "Once upon a time, a small model learned to predict the next token.",
        "The training stream is deterministic, packed, and shared by every run.",
        "Recursive exact execution remains the source of truth for audits.",
    ] * 200


def load_token_streams(config: DataConfig, training: TrainingConfig, vocab_size: int) -> TokenStreams:
    if config.dataset == "synthetic":
        tokens = _synthetic_tokens(vocab_size, config.synthetic_tokens, training.seed)
        split = max(training.seq_len + 2, int(tokens.numel() * 0.9))
        return TokenStreams(
            train=tokens[:split],
            eval=tokens[split - training.seq_len - 1 :],
            tokenizer_name="synthetic",
            data_fingerprint=f"synthetic:{vocab_size}:{config.synthetic_tokens}:{training.seed}",
        )
    texts = _load_texts(config)
    tokens, tokenizer_name = _encode_with_gpt2_bpe(texts, Path(config.cache_dir))
    if config.vocab_projection == "modulo":
        tokens = tokens.remainder(vocab_size)
        tokenizer_name = f"{tokenizer_name}_mod{vocab_size}"
    else:
        tokens = tokens[tokens < vocab_size]
    if tokens.numel() < training.seq_len * 4:
        tokens = _synthetic_tokens(vocab_size, config.synthetic_tokens, training.seed)
        tokenizer_name = f"{tokenizer_name}+synthetic_fill"
    tokens = tokens[: config.max_tokens]
    split = max(training.seq_len + 2, min(tokens.numel() - training.seq_len - 2, tokens.numel() - config.eval_tokens))
    return TokenStreams(
        train=tokens[:split],
        eval=tokens[split - training.seq_len - 1 :],
        tokenizer_name=tokenizer_name,
        data_fingerprint=f"{config.dataset}:{tokenizer_name}:{tokens.numel()}",
    )
