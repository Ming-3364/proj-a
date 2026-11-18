> **状态：已降级（非目标）。** 主目标见 `TARGET.md`：构建 minitriton 以理解 Triton。
> 本文件仅在 W3–W4 有余力时作为可选动作；过程中自然发现的上游问题，记进
> `docs/notes/upstream-vs-mine.md` 即可，**不投入专门时间、不设 Gate、不算交付物**。

# 开源贡献大纲：给 Triton 上游

> 定位：**minitriton 的副产品，不是第二条主线。** 不设 Gate、不设 KPI，做出来是加分，做不出来不影响主线交付。

---

## 一、目的（三条，按重要性）

1. **可验证的"我真读过编译器代码"**：面试里说"我理解 Triton 的 tiling/layout"很容易，说"这是我提交到上游的 PR"无法造假。
2. **把自研编译器当差分测试器**：两个独立实现（官方 vs minitriton）结果不一致的地方，大概率就是 bug。这是找真问题最高效的方式，比读代码瞎猜强十倍。
3. **逼自己读上游实现**：为了给一个 pass 提 PR，你必须读懂它——收获最大的是自己。

## 二、手段（四层阶梯，按"成本 / 成功率"排）

| 层 | 做什么 | 成本 | 被接受概率 | 建议 |
|---|---|---|---|---|
| L0 | 最小复现 + issue（bug report） | 低 | 高 | **主力**。好 issue 有真实价值，且不依赖 review 排期 |
| L1 | 文档/注释的事实错误修正 | 低 | 中 | 顺手做，别当成果 |
| L2 | 回归测试（给已修 bug 补 test） | 低中 | 中高 | 跟着 L0 走：报了 bug 就顺手提 test |
| L3 | 代码修复 / 小特性（后端边界、错误信息、类型推断） | 高 | 中低 | 只在 L0 被 maintainer 确认后再做 |
| L4 | 改一个 pass 的优化逻辑 | 很高 | 低 | 4 周内不碰 |

**关键判断：先 issue，后 PR。** 上游明确"所有 maintainer 保留不保证及时 review 的权利"（CONTRIBUTING.md 原话），所以不要把计划压在"PR 被合入"上。

## 三、上游现状（我实测的三条事实，别再重复踩）

| 事实 | 影响 |
|---|---|
| `good first issue` 标签**没人用**（GitHub 搜索返回 total_count = 0） | 别去找"新手任务"列表，不存在 |
| **Interpreter 是少数有明确 owner 的 in-tree module** | 但它不是软柿子，见下条 |
| Interpreter 覆盖**比预期好**：我实测 `tl.gather / sort / histogram / cumsum / argmax / clamp / atomic_max / flip / erf / make_tensor_descriptor` 在 `TRITON_INTERPRET=1` 下**都能跑** | "补 interpreter 缺口"这条低垂果实基本不存在，别把它当前期目标 |

> 这三条合起来意味着：**容易的活已经被干完了**，剩下的价值集中在"真实后端行为"上——而那需要 H100。

## 四、我最推荐的路径：差分测试找 bug

```
同一 kernel × 同一 shapes
   ├─ 官方 Triton 编译执行  ─┐
   └─ minitriton 生成并执行 ─┴→ 都对齐 torch reference → 谁偏离谁可疑
                                 ↓
                        最小复现 → issue（附 shape、dtype、版本、期望/实际）
```

**为什么这条路对**：
- 你的比较框架（`bench/compare_official.py`）本来就要写，顺手就是 bug 探测器；
- 边界条件（非 2 幂 shape、奇数维度、`num_stages` 边界、大 block、空 seq）最容易触发上游问题，而这些正是 minitriton 早期会覆盖不到、官方实现长期没人测的组合；
- 产出（issue + 最小复现）本来就是社区最缺的东西，**不依赖 maintainer 排期**。

**加分做法**：issue 里直接给"我的两个实现分别算出 X 和 Y，torch 是 Z"，比"我觉得这里有问题"有力得多。

## 五、选题库（按建议顺序）

| # | 选题 | 难度 | 说明 |
|---|---|---|---|
| S1 | **Hopper sm_90 上的正确性差异**（TMA/descriptor、`num_stages` 边界、大 block 的 swizzle） | 中 | 你有 H100 · 这正是 W3 深挖 TMA 时会撞到的东西 |
| S2 | **错误信息可读性**（不支持的写法给出"为什么不支持 + 怎么改"） | 低 | 小、安全、几乎不会被拒，但要挑真正含糊的地方 |
| S3 | **文档中的事实错误**（pass 行为、参数语义与代码不符） | 低 | 顺手；能证明你逐行读过 |
| S4 | **CI/测试覆盖**（给无人测的 shape/dtype 组合补 case） | 中 | 价值稳定，但需要 GPU 跑 CI |
| S5 | ~~补 interpreter 缺口~~ | — | **已排除**：实测覆盖良好 |

## 六、时间线（与 minitriton 并行，不占额外工作日）

| 阶段 | 动作 | 产出 |
|---|---|---|
| W0–W1 | 只**读**：把 `tritongpu-*` 几条关键 pass 与 layout 编码读懂，记录疑问（不提交任何东西） | `docs/notes/upstream-reading.md` |
| W2 | 差分测试跑起来 → 收集 3~5 个"可疑点" | 候选 bug 列表 |
| W3 | 挑 1~2 个做最小复现 → 提 issue | issue 链接（无论是否被回） |
| W4 | 若 maintainer 确认 → 提对应 test / 小修 PR；否则停在 issue | 1 个 issue +（可选）1 个 PR |

## 七、风险与止损

| 风险 | 对策 |
|---|---|
| PR 长期无人 review，挫败感强 | 目标定为"**issue + 最小复现**"，PR 是 bonus；上游自己都说 review 不保证及时 |
| 撞上 main 分支大改，复现失效 | 复现里写清 commit hash；用容器镜像固定环境 |
| 本地无法从源码构建 Triton（需要 LLVM + CUDA） | **不要在 Mac 上构建**：需要构建的场景一律在 H100 机器或容器里做；纯 Python 层（`python/triton/language/`、interpreter、测试）可在容器里验证 |
| 花太多时间读代码 | 用"差分测试"驱动：先看到不一致，再定向读相关代码 |

## 八、面试怎么讲（两句话）

> "我给上游提过一个 Hopper 上的正确性问题，最小复现是 8 行 Triton，附带我自己的两个实现分别算出什么——这是我在写自己的编译器时发现的，因为两个独立实现打架的地方大概率有 bug。"
>
> "我没有以'合入 PR'为目标，因为上游明确不保证 review 排期；把问题定位清楚本身就有价值。"

（如果把"我在做一个 mini Triton 编译器，并用它给上游做差分测试"连起来讲，这两件事就变成了一件事。）
