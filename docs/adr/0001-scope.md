# ADR 0001：范围与「不做什么」

日期：2026-11-11 ｜ 状态：已接受

## 背景

面试项目 / 学习项目：通过 Triton 理解 AI 编译器的几个核心机制
（L1 tiling、bank conflict、layout 与 IR），并把一个 kernel 接进 SGLang 做落地验证。

硬约束：本地开发机是 Apple Silicon（无 NVIDIA GPU，**且 macOS 没有 triton 轮子**）；
GPU 侧是 H100 PCIe 80GB（sm_90a），ncu 可用。

## 决策

**做**：
1. GEMM tiling ladder（v0 → v3），只作为「标定分析模型」的实验台；
2. paged-KV decode attention（k0 → k4，split-KV + GQA 复用 + autotune）为主线；
3. TTGIR / PTX / SASS 取证，`#ttg.swizzled_shared` → bank 映射模型；
4. CUDA baseline（`__shared__` padding vs XOR swizzle）作为 bank conflict 的 ground truth；
5. 无 GPU 分析模型（smem/occupancy、bank、roofline、split-KV 最优解）；
6. SGLang 最小 backend（decode only），**只作落地探针**，pin `sglang==0.5.23`。

**不做**（防止范围失控）：
- 自己写 DSL 前端、改 Triton 的 pass、自己实现 layout 推断；
- flash-attention backward、MLA / sparse attention / fp4；
- 多卡并行、PD 分离、投机解码集成；
- SGLang 端的性能调优与 CUDA graph 深度适配（graph 兼容性只作为 stretch）；
- 融合 MoE、量化训练。

## 后果

- 交付物可在 4 周 × 10~12h 内完成；
- 若时间不足，先砍 §5（SGLang 探针）与 TMA 深挖，保留 1/2/3/4/5；
- 报告里必须显式列出「未验证的部分」，不允许用估算数字冒充实测。
