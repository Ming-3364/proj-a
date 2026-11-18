"""参考实现的正确性测试（纯 CPU，零依赖）。

这些测试是整条链路的「地基验收」：
kernel 之后所有的数值对比，都以 refs/paged_attention.py 为准，
所以先用 SDPA 这个完全独立的实现把它钉死。
"""

from __future__ import annotations

import unittest

import torch

from refs.paged_attention import (
    PagedKVCache,
    build_random_case,
    gather_kv,
    paged_decode_reference,
    sdpa_oracle,
)

ATOL = 1e-5
RTOL = 1e-4


def test_gather_restores_contiguous_layout():
    """gather 出来的 KV 必须与「逻辑连续布局」逐一相等。"""
    bs, h_kv, d, page_size = 2, 4, 8, 16
    seq_lens = [37, 5]
    _, cache = build_random_case(bs, 8, h_kv, d, seq_lens, page_size=page_size, seed=1)

    for b, s in enumerate(seq_lens):
        k, v = gather_kv(cache, b)
        assert k.shape == (s, h_kv, d), k.shape
        assert v.shape == (s, h_kv, d), v.shape

        # 逐页手工拼一遍，和 gather 的结果对拍
        n_pages = (s + page_size - 1) // page_size
        manual = []
        for p in range(n_pages):
            phys = int(cache.block_table[b, p].item())
            manual.append(cache.k_cache[phys])
        manual = torch.cat(manual, dim=0)[:s]
        assert torch.equal(k, manual), "gather 的页拼接顺序与 block_table 不一致"


def test_reference_matches_sdpa_gqa_partial_page():
    """GQA + 最后一页只用了 5/16：两种实现必须一致。"""
    q, cache = build_random_case(
        bs=3, h_q=32, h_kv=8, d=64, seq_lens=[37, 128, 5], page_size=16, seed=2
    )
    ref = paged_decode_reference(q, cache)
    oracle = sdpa_oracle(q, cache)
    assert ref.shape == (3, 32, 64)
    assert torch.allclose(ref, oracle, atol=ATOL, rtol=RTOL), (
        (ref - oracle).abs().max().item()
    )


def test_reference_matches_sdpa_mha_page_size_one():
    """退化情形：MHA（H_q == H_kv）+ page_size=1，最容易被写错的边界。"""
    q, cache = build_random_case(
        bs=2, h_q=4, h_kv=4, d=32, seq_lens=[9, 17], page_size=1, seed=3
    )
    ref = paged_decode_reference(q, cache)
    oracle = sdpa_oracle(q, cache)
    assert torch.allclose(ref, oracle, atol=ATOL, rtol=RTOL)


def test_block_table_padding_is_not_read():
    """block_table 里超出 seq_len 的项是垃圾值，不能影响结果。"""
    q, cache = build_random_case(
        bs=2, h_q=8, h_kv=2, d=16, seq_lens=[20, 33], page_size=8, seed=4
    )
    base = paged_decode_reference(q, cache)

    # 把未使用的表项改成越界/垃圾值
    polluted = PagedKVCache(
        k_cache=cache.k_cache, v_cache=cache.v_cache,
        block_table=cache.block_table.clone(), seq_lens=cache.seq_lens.clone(),
    )
    for b, s in enumerate([20, 33]):
        n_used = (s + 8 - 1) // 8
        polluted.block_table[b, n_used:] = 9999  # 越界页号

    after = paged_decode_reference(q, polluted)
    assert torch.equal(base, after), "参考实现读到了 block_table 的 padding 项"


def test_gqa_head_grouping_is_interleaved_not_tiled():
    """GQA 的分组必须是 repeat_interleave（连续分组），不是 tile。

    构造 K 每个 head 为常数、V 每个 head 为互不相同的单位向量：
    此时 softmax 均匀，输出必然等于对应 KV head 的 V。
    若把分组写成 tile（h % group），这个测试会立刻失败。
    """
    h_q, h_kv, d, page_size, s = 8, 2, 4, 4, 8
    group = h_q // h_kv

    k_cache = torch.zeros(1, page_size, h_kv, d)
    v_cache = torch.zeros(1, page_size, h_kv, d)
    for h in range(h_kv):
        v_cache[:, :, h, h] = 1.0  # 每个 KV head 一个不同的 one-hot
    block_table = torch.zeros(1, 2, dtype=torch.int32)
    block_table[0, :2] = torch.tensor([0, 0], dtype=torch.int32)
    cache = PagedKVCache(k_cache, v_cache, block_table, torch.tensor([s], dtype=torch.int32))

    q = torch.randn(1, h_q, d)  # q 取值无关，因为 K 全相等
    out = paged_decode_reference(q, cache)[0]  # [H_q, D]

    for h in range(h_q):
        expect = v_cache[0, 0, h // group]  # repeat_interleave 语义
        assert torch.allclose(out[h], expect, atol=1e-6), (
            f"head {h} 期望 kv head {h // group}，实际 {out[h].argmax().item()}"
        )


def test_fp16_path_close_to_fp32_reference():
    """fp16 输入在容差内应接近 fp32 参考（后面 kernel 也是 fp16 为主）。"""
    q32, cache32 = build_random_case(
        bs=2, h_q=8, h_kv=2, d=64, seq_lens=[64, 100], page_size=16, seed=5
    )
    ref32 = paged_decode_reference(q32, cache32)

    q16, cache16 = q32.half(), cache32.to("cpu", dtype=torch.float16)
    ref16 = paged_decode_reference(q16, cache16)
    assert ref16.dtype == torch.float16
    diff = (ref16.float() - ref32).abs().max().item()
    assert diff < 2e-2, f"fp16 与 fp32 参考偏差过大: {diff}"
