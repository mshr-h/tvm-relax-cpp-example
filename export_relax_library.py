import numpy as np
import tvm
from tvm import relax

from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T

@I.ir_module
class TVMScriptModule:
    @T.prim_func
    def addone(A_handle: T.handle, B_handle: T.handle) -> None:
        m = T.int64()
        n = T.int64()
        A = T.match_buffer(A_handle, (m, n), "int32")
        B = T.match_buffer(B_handle, (m, n), "int32")
        T.func_attr(({"global_symbol": "addone"}))
        for i, j in T.grid(m, n):
            with T.block("addone"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + T.int32(1)

    @R.function
    def main(x: R.Tensor(("m", "n"), "int32")):
        m, n = T.int64(), T.int64()
        gv0 = R.call_tir(TVMScriptModule.addone, (x,), R.Tensor((m, n), dtype="int32"))
        return gv0

mod = TVMScriptModule
mod.show()

mod: tvm.IRModule = relax.transform.LegalizeOps()(mod)
mod.show()

mod: tvm.IRModule = relax.get_pipeline("zero")(mod)
mod.show()

target = tvm.target.Target("llvm")
executable = relax.build(mod, target, exec_mode="compiled")
executable.export_library("compiled_artifact.so")

dev = tvm.cpu()
vm = relax.VirtualMachine(executable, dev)
data: tvm.runtime.NDArray = tvm.nd.array(np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int32),
                                         device=dev)
cpu_out = vm["main"](data).numpy()
print(cpu_out)

loaded_mod: tvm.runtime.Module = tvm.runtime.load_module("compiled_artifact.so")
vm1 = relax.VirtualMachine(loaded_mod, dev)
cpu_out1 = vm1["main"](data).numpy()
print(cpu_out1)