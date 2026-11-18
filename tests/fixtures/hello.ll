; S1 hello-cubin 夹具：一个最小的、能独立验证工具链的 kernel。
; 语义：out[tid] = (float)tid，仅在 tid < n 时写。
; 注意 `ptx_kernel` 调用约定（CC 71）：少了它就只能生成 `.func`，无法被 launch。
; 故意用 opaque pointer 语法（LLVM 15+ 的唯一形式），确保不会被版本演进淘汰。
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define ptx_kernel void @hello(ptr addrspace(1) %out, i32 %n) {
entry:
  %tid = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %ok = icmp ult i32 %tid, %n
  br i1 %ok, label %body, label %exit
body:
  %idx = sext i32 %tid to i64
  %p = getelementptr float, ptr addrspace(1) %out, i64 %idx
  %f = sitofp i32 %tid to float
  store float %f, ptr addrspace(1) %p
  br label %exit
exit:
  ret void
}
