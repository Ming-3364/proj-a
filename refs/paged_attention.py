"""Paged-KV decode attention：语义定义 + torch 参考实现。

这个文件是整个项目的「地基」：kernel 的正确性全部以它为准。
它刻意写成最直白的形式（fp32 显式 matmul + softmax），不做任何融合，
只为了保证「语义无歧义」：

  k_cache / v_cache : [num_pages, page_size, H_kv, D]
  block_table       : [B, max_pages]，第 b 行的前 ceil(len_b/page_size) 项是该序列的物理页号
  seq_lens          : [B]
  q                 : [B, H_q, D]（decode：每个序列 1 个 query token）
  out               : [B, H_q, D]

三个容易写错的点，这里都显式处理：
  1. 最后一个 page 通常只用一部分 -> gather 后必须按 seq_len 截断；
  2. block_table 里超出 seq_len 的项是垃圾值 -> 不能无条件 gather；
  3. GQA 的 Q head 是「分组连续」的（H_q / H_kv 个连续 Q head 共享一个 KV head），
     对应 repeat_interleave，而不是 tile。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "PagedKVCache",
    "build_random_case",
    "gather_kv",
    "paged_decode_reference",
    "sdpa_oracle",
]


@dataclass
class PagedKVCache:
    """最小 paged KV cache 表示（字段与 SGLang 的语义一致，形状做了简化）。"""

    k_cache: torch.Tensor  # [num_pages, page_size, H_kv, D]
    v_cache: torch.Tensor  # [num_pages, page_size, H_kv, D]
    block_table: torch.Tensor  # [B, max_pages]
    seq_lens: torch.Tensor  # [B]

    @property
    def page_size(self) -> int:
        return self.k_cache.shape[1]

    @property
    def num_kv_heads(self) -> int:
        return self.k_cache.shape[2]

    @property
    def head_dim(self) -> int:
        return self.k_cache.shape[3]

    def to(self, device, dtype=None) -> "PagedKVCache":
        kw = {} if dtype is None else {"dtype": dtype}
        return PagedKVCache(
            k_cache=self.k_cache.to(device, **kw),
            v_cache=self.v_cache.to(device, **kw),
            block_table=self.block_table.to(device),
            seq_lens=self.seq_lens.to(device),
        )


def _pages_needed(seq_len: int, page_size: int) -> int:
    return (seq_len + page_size - 1) // page_size


def gather_kv(cache: PagedKVCache, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    """按 block_table 把第 b 条序列的 KV 还原成连续布局。

    返回 k, v，形状均为 [seq_len, H_kv, D]。
    """
    seq_len = int(cache.seq_lens[b].item())
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    n_pages = _pages_needed(seq_len, cache.page_size)
    pages = cache.block_table[b, :n_pages].to(torch.long)

    k = cache.k_cache.index_select(0, pages)  # [n_pages, page_size, H_kv, D]
    v = cache.v_cache.index_select(0, pages)
    k = k.reshape(n_pages * cache.page_size, cache.num_kv_heads, cache.head_dim)
    v = v.reshape(n_pages * cache.page_size, cache.num_kv_heads, cache.head_dim)
    return k[:seq_len], v[:seq_len]


def paged_decode_reference(
    q: torch.Tensor,
    cache: PagedKVCache,
    scale: float | None = None,
) -> torch.Tensor:
    """逐序列的 decode attention 参考实现，返回 [B, H_q, D]。

    数学上等价于：softmax(q @ K^T / sqrt(D)) @ V，KV 来自 paged cache。
    """
    if q.dim() != 3:
        raise ValueError(f"q must be [B, H_q, D], got {tuple(q.shape)}")
    bs, h_q, d = q.shape
    h_kv = cache.num_kv_heads
    if d != cache.head_dim:
        raise ValueError(f"head_dim mismatch: q={d}, cache={cache.head_dim}")
    if h_q % h_kv != 0:
        raise ValueError(f"H_q ({h_q}) must be divisible by H_kv ({h_kv})")
    if bs != cache.block_table.shape[0]:
        raise ValueError("batch size mismatch between q and block_table")

    scale = (1.0 / math.sqrt(d)) if scale is None else scale
    group = h_q // h_kv
    # 参考实现内部统一用 fp32/fp64 算，避免 fp16 累加噪声干扰「语义」判定
    work_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32

    outs = []
    for b in range(bs):
        k, v = gather_kv(cache, b)  # [S, H_kv, D]
        kk = k.transpose(0, 1).to(work_dtype)  # [H_kv, S, D]
        vv = v.transpose(0, 1).to(work_dtype)
        if group > 1:
            # GQA：分组连续，与 SDPA(enable_gqa=True) 的约定一致
            kk = kk.repeat_interleave(group, dim=0)  # [H_q, S, D]
            vv = vv.repeat_interleave(group, dim=0)

        qb = q[b].to(work_dtype)  # [H_q, D]
        scores = torch.einsum("hd,hsd->hs", qb, kk) * scale  # [H_q, S]
        probs = torch.softmax(scores, dim=-1)
        out = torch.einsum("hs,hsd->hd", probs, vv)  # [H_q, D]
        outs.append(out)

    return torch.stack(outs, dim=0).to(q.dtype)


def sdpa_oracle(
    q: torch.Tensor,
    cache: PagedKVCache,
    scale: float | None = None,
) -> torch.Tensor:
    """用 PyTorch SDPA 作为「独立第三方」交叉验证参考实现。

    走的是完全不同的代码路径（fused kernel / flash 后端），
    用来防「我自己写的两个实现同时错」。
    """
    import torch.nn.functional as F

    scale = (1.0 / math.sqrt(cache.head_dim)) if scale is None else scale
    outs = []
    for b in range(q.shape[0]):
        k, v = gather_kv(cache, b)
        kk = k.transpose(0, 1).unsqueeze(0).to(q.dtype)  # [1, H_kv, S, D]
        vv = v.transpose(0, 1).unsqueeze(0).to(q.dtype)
        qq = q[b].unsqueeze(0).unsqueeze(2)  # [1, H_q, 1, D]
        out = F.scaled_dot_product_attention(qq, kk, vv, scale=scale, enable_gqa=True)
        outs.append(out.squeeze(0).squeeze(1))
    return torch.stack(outs, dim=0)


def build_random_case(
    bs: int,
    h_q: int,
    h_kv: int,
    d: int,
    seq_lens: list[int],
    page_size: int = 16,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, PagedKVCache]:
    """构造一个可复现的随机 paged 用例（q + cache）。

    页分配策略：每个序列独占连续的 max_pages 个页，便于调试时把页号对回序列。
    """
    assert len(seq_lens) == bs
    g = torch.Generator().manual_seed(seed)
    max_pages = max(_pages_needed(s, page_size) for s in seq_lens)
    num_pages = bs * max_pages

    block_table = torch.zeros(bs, max_pages, dtype=torch.int32)
    for b, s in enumerate(seq_lens):
        n = _pages_needed(s, page_size)
        block_table[b, :n] = torch.arange(b * max_pages, b * max_pages + n, dtype=torch.int32)

    k_cache = torch.randn(num_pages, page_size, h_kv, d, generator=g)
    v_cache = torch.randn(num_pages, page_size, h_kv, d, generator=g)
    q = torch.randn(bs, h_q, d, generator=g)

    cache = PagedKVCache(
        k_cache=k_cache.to(dtype),
        v_cache=v_cache.to(dtype),
        block_table=block_table,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
    )
    return q.to(dtype), cache.to(device)
