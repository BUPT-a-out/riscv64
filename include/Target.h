#pragma once
#include <string>
#include <vector>

#include "IR/Module.h"
#include "Instructions/Module.h"
#include "Pass/Pass.h"

namespace riscv64 {

class RISCV64Target {
   public:
    RISCV64Target() = default;
    ~RISCV64Target() = default;

    std::string compileToAssembly(
        const midend::Module& module,
        const midend::AnalysisManager* analysisManager = nullptr);

    // 三阶段编译流程
    Module instructionSelectionPass(
        const midend::Module& module);  // 阶段1：指令选择
    Module& valueReusePass(riscv64::Module& riscv_module,
                           const midend::Module& midend_module,
                           const midend::AnalysisManager* analysisManager =
                               nullptr);  // 阶段0.5：值重用优化
    Module& constantFoldingPass(
        riscv64::Module& module);  // 阶段1.6：常量折叠优化
    Module& initialFrameIndexPass(
        riscv64::Module& module);  // 阶段1.5：初始Frame Index
    Module& basicBlockReorderingPass(
        riscv64::Module& module);  // 阶段1.7：基本块重排优化

    Module& RAGreedyPass(riscv64::Module& module);

    Module& registerAllocationPass(
        riscv64::Module& module,
        const midend::AnalysisManager* analysisManager =
            nullptr);  // 阶段2：寄存器分配
    Module& deadCodeEliminationPass(
        riscv64::Module& module);  // 寄存器分配后：死代码删除
    Module& frameIndexEliminationPass(
        riscv64::Module& module);  // 阶段3：Frame Index消除

    // 保留原有方法以兼容
    Module& reorderInstructionsPass(riscv64::Module& module);
    Module& basicBlockSchedulingPass(riscv64::Module& module);
};

}  // namespace riscv64