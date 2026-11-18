"""Tile IR 核心数据结构。

设计取舍（为什么不用 MLIR）：
  - 4 周内学 MLIR 只会让人写不出东西；这里借它的**概念**（SSA、类型、layout 属性、pass pipeline），
    不借它的实现。IR 用 dataclass 表示，天然好做 golden 快照测试。

IR 的粒度是 **tile 级**（一个 Value 代表一整块张量，而不是一个线程的元素）。
线程映射发生在 codegen 阶段（P4），不在这个 IR 里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "Ty",
    "Param",
    "Value",
    "Op",
    "Function",
    "Module",
    "IRBuilder",
    "DTYPES",
]

DTYPES = {"f16": 2, "bf16": 2, "f32": 4, "f64": 8, "i8": 1, "i16": 2, "i32": 4, "i64": 8}


@dataclass(frozen=True)
class Ty:
    """IR 类型。

    kind: ptr | tensor | scalar | void
    """

    kind: str
    dtype: str = "f32"
    shape: tuple[int, ...] = ()

    def __str__(self) -> str:
        if self.kind == "void":
            return "void"
        if self.kind == "tensor":
            dims = "x".join(str(d) for d in self.shape)
            return f"tensor<{dims}:{self.dtype}>"
        return f"{self.kind}<{self.dtype}>"

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def with_shape(self, shape: Iterable[int]) -> "Ty":
        return Ty(self.kind, self.dtype, tuple(shape))


T_PTR_F32 = Ty("ptr", "f32")
T_VOID = Ty("void")


@dataclass
class Value:
    """SSA 值(定义了结果的 op)。"""

    id: int
    op: str
    ty: Ty
    operands: tuple[int, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Op:
    """没有结果的语句（目前只有 store）。"""

    op: str
    operands: tuple[int, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Param:
    """运行时参数（指针或标量）。constexpr 参数不进 params，而是进 Function.meta。"""

    id: int
    name: str
    ty: Ty


@dataclass
class Function:
    name: str
    params: list[Param] = field(default_factory=list)
    meta: dict[str, int] = field(default_factory=dict)
    body: list[Value | Op] = field(default_factory=list)

    def values(self) -> list[Value]:
        return [x for x in self.body if isinstance(x, Value)]

    def ops(self) -> list[Op]:
        return [x for x in self.body if isinstance(x, Op)]

    def op_sequence(self) -> list[str]:
        return [x.op for x in self.body]


@dataclass
class Module:
    functions: list[Function] = field(default_factory=list)


class IRBuilder:
    """唯一的 ID 分配者。所有 pass 都必须经由它产出新值，便于保证 SSA 合法。"""

    def __init__(self, func: Function):
        self.func = func
        # SSA 编号从运行时参数之后开始，保证 %id 全局唯一可读
        self._next = (max((p.id for p in func.params), default=-1) + 1)

    def emit(
        self,
        op: str,
        ty: Ty,
        operands: Iterable[int] = (),
        **attrs: Any,
    ) -> Value:
        v = Value(self._next, op, ty, tuple(operands), dict(attrs))
        self._next += 1
        self.func.body.append(v)
        return v

    def stmt(self, op: str, operands: Iterable[int] = (), **attrs: Any) -> Op:
        o = Op(op, tuple(operands), dict(attrs))
        self.func.body.append(o)
        return o

    def consume(self, v: Value) -> int:
        """把 Value 变成操作数 id。"""
        return v.id
