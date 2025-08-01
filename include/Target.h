#pragma once
#include <string>
#include <vector>

#include "IR/Module.h"
#include "Instructions/Module.h"

namespace riscv64 {

class RISCV64Target {
   public:
    RISCV64Target() = default;
    ~RISCV64Target() = default;

    std::string compileToAssembly(const midend::Module& module);

    // 三阶段编译流程
    Module instructionSelectionPass(
        const midend::Module& module);  // 阶段1：指令选择
    Module& initialFrameIndexPass(
        riscv64::Module& module);  // 阶段1.5：初始Frame Index
    Module& registerAllocationPass(
        riscv64::Module& module);  // 阶段2：寄存器分配
    Module& frameIndexEliminationPass(
        riscv64::Module& module);  // 阶段3：Frame Index消除

    // 保留原有方法以兼容
    Module& reorderInstructionsPass(riscv64::Module& module);
    Module& basicBlockSchedulingPass(riscv64::Module& module);

    // 废弃的方法（使用frameIndexEliminationPass代替）
    [[deprecated("Use frameIndexEliminationPass instead")]] Module&
    frameIndexPass(riscv64::Module& module);
};

}  // namespace riscv64