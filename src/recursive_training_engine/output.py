from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from recursive_training_engine.config import OutputConfig
from recursive_training_engine.kernels import optimized as K


@dataclass(slots=True)
class ShortlistResult:
    loss_per_sample: torch.Tensor
    logits: torch.Tensor
    shortlist: torch.Tensor
    target_positions: torch.Tensor
    cluster_logits: torch.Tensor
    duplicate_count: torch.Tensor


class ShortlistHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, config: OutputConfig):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.config = config
        self.cluster_router = nn.Linear(d_model, config.num_clusters, bias=False)
        cluster_ids = torch.arange(vocab_size, dtype=torch.long) % config.num_clusters
        self.register_buffer("cluster_ids", cluster_ids, persistent=False)
        max_per_cluster = (vocab_size + config.num_clusters - 1) // config.num_clusters
        members = torch.full((config.num_clusters, max_per_cluster), 0, dtype=torch.long)
        counts = torch.zeros(config.num_clusters, dtype=torch.long)
        for token in range(vocab_size):
            c = int(cluster_ids[token])
            members[c, counts[c]] = token
            counts[c] += 1
        self.register_buffer("cluster_members", members, persistent=False)
        self.register_buffer("cluster_counts", counts, persistent=False)
        hard = torch.arange(config.hard_negatives, dtype=torch.long) % vocab_size
        self.register_buffer("hard_negative_cache", hard, persistent=False)
        filler = torch.arange(config.shortlist_max_tokens, dtype=torch.long) % vocab_size
        self.register_buffer("shortlist_filler", filler, persistent=False)

    def _dedupe_preserve_first(
        self,
        candidates: torch.Tensor,
        max_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = candidates.shape[-1]
        flat = candidates.reshape(-1, width).long()
        positions = torch.arange(width, device=candidates.device, dtype=torch.float32).view(1, -1)
        first_pos = torch.full(
            (flat.shape[0], self.vocab_size),
            float(width + max_tokens),
            dtype=torch.float32,
            device=candidates.device,
        )
        first_pos.scatter_reduce_(
            1,
            flat,
            positions.expand_as(flat),
            reduce="amin",
            include_self=True,
        )
        present = first_pos < width
        duplicate_count = (width - present.sum(dim=-1)).float().sum()
        _, shortlist = first_pos.topk(k=max_tokens, dim=-1, largest=False, sorted=True)
        return shortlist.view(*candidates.shape[:-1], max_tokens).contiguous(), duplicate_count

    def build_shortlist(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        *,
        seed: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, _ = hidden.shape
        device = hidden.device
        cluster_logits = self.cluster_router(hidden)
        top_clusters = cluster_logits.topk(
            k=min(self.config.shortlist_top_clusters, self.config.num_clusters), dim=-1
        ).indices
        max_tokens = self.config.shortlist_max_tokens
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        random_negs = torch.randint(
            0,
            self.vocab_size,
            (b, s, self.config.random_negatives),
            device=device,
            generator=generator,
        )
        top_members = self.cluster_members.to(device)[top_clusters]
        top_members = top_members.flatten(start_dim=-2)
        hard = self.hard_negative_cache.to(device).view(1, 1, -1).expand(b, s, -1)
        filler = self.shortlist_filler.to(device).view(1, 1, -1).expand(b, s, -1)
        target_filler_width = max_tokens * 2
        target_filler = (
            targets.unsqueeze(-1)
            + torch.arange(target_filler_width, device=device).view(1, 1, -1)
        ) % self.vocab_size
        candidates = torch.cat(
            [targets.unsqueeze(-1), top_members, hard, random_negs, filler, target_filler],
            dim=-1,
        )
        shortlist, duplicate_count = self._dedupe_preserve_first(candidates, max_tokens)
        target_pos = torch.zeros(b, s, dtype=torch.long, device=device)
        return shortlist, target_pos, cluster_logits, duplicate_count

    def loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        vocab_weight: torch.Tensor,
        *,
        seed: int,
    ) -> ShortlistResult:
        shortlist, target_pos, cluster_logits, duplicate_count = self.build_shortlist(
            hidden,
            targets,
            seed=seed,
        )
        logits = K.k_logits_shortlist(hidden, vocab_weight, shortlist)
        loss_tokens = F.cross_entropy(
            logits.flatten(0, -2), target_pos.flatten(), reduction="none"
        ).view_as(targets)
        return ShortlistResult(
            loss_per_sample=loss_tokens.sum(dim=-1),
            logits=logits,
            shortlist=shortlist,
            target_positions=target_pos,
            cluster_logits=cluster_logits,
            duplicate_count=duplicate_count,
        )
