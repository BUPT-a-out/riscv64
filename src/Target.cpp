#include "Target.h"

#include "BasicBlockReordering.h"
#include "CodeGen.h"
#include "FrameIndexElimination.h"
#include "FrameIndexPass.h"
#include "IR/Function.h"
#include "RegAllocChaitin.h"
#include "Visit.h"

namespace riscv64 {

std::string RISCV64Target::compileToAssembly(const midend::Module& module) {
    std::vector<std::string> assembly;

    // 添加汇编头部
    assembly.emplace_back(".text");
    assembly.emplace_back(".global _start");

    // 执行完整的三阶段编译流程
    auto riscv_module = instructionSelectionPass(module);
    initialFrameIndexPass(riscv_module);  // 第一阶段
    basicBlockReorderingPass(riscv_module);  // 第1.7阶段：基本块重排优化
    registerAllocationPass(riscv_module);     // 第二阶段
    frameIndexEliminationPass(riscv_module);  // 第三阶段

    return riscv_module.toString();
}

Module RISCV64Target::instructionSelectionPass(const midend::Module& module) {
    std::cout << "\n=== Phase 1: Instruction Selection ===" << std::endl;
    CodeGenerator codegen;
    auto riscv_module = codegen.visitor_->visit(&module);
    std::cout << module.toString() << std::endl;
    return riscv_module;
}

Module& RISCV64Target::initialFrameIndexPass(riscv64::Module& module) {
    std::cout << "\n=== Phase 1.5: Initial Frame Index Creation ==="
              << std::endl;

    // 第一阶段已经在指令选择中完成了alloca的Frame Index创建
    // 这里只需要确保所有alloca都有对应的抽象Frame Index
    for (auto& function : module) {
        if (function->empty()) continue;

        std::cout << "Verifying abstract Frame Indices for function: "
                  << function->getName() << std::endl;

        // 验证所有frameaddr指令都有有效的Frame Index
        for (auto& bb : *function) {
            for (auto& inst : *bb) {
                if (inst->getOpcode() == Opcode::FRAMEADDR) {
                    const auto& operands = inst->getOperands();
                    if (operands.size() >= 2) {
                        if (auto* fi = dynamic_cast<FrameIndexOperand*>(
                                operands[1].get())) {
                            std::cout << "  Found abstract FI("
                                      << fi->getIndex() << ")" << std::endl;
                        }
                    }
                }
            }
        }
    }

    std::cout << module.toString() << std::endl;

    return module;
}

Module& RISCV64Target::basicBlockReorderingPass(riscv64::Module& module) {
    std::cout << "\n=== Phase 1.7: Basic Block Reordering ===" << std::endl;

    for (auto& function : module) {
        if (function->empty()) {
            std::cout << "Skipping empty function: " << function->getName()
                      << std::endl;
            continue;
        }

        std::cout << "Processing function: " << function->getName()
                  << std::endl;

        BasicBlockReordering reordering(function.get());
        reordering.run();
    }

    std::cout << "=== Basic Block Reordering Completed ===" << std::endl;
    std::cout << module.toString() << std::endl;

    return module;
}

Module& RISCV64Target::registerAllocationPass(riscv64::Module& module) {
    std::cout << "\n=== Phase 2: Register Allocation ===" << std::endl;

    for (auto& function : module) {
        std::cout << "RegAlloc for float" << std::endl;
        RegAllocChaitin allocatorFloat(function.get(), true);
        allocatorFloat.run();

        std::cout << function->toString() << std::endl;

        std::cout << "RegAlloc for int" << std::endl;
        RegAllocChaitin allocatorInt(function.get(), false);
        allocatorInt.run();

        std::cout << function->toString() << std::endl;
    }

    std::cout << module.toString() << std::endl;

    return module;
}

Module& RISCV64Target::frameIndexEliminationPass(riscv64::Module& module) {
    std::cout << "\n=== Phase 3: Frame Index Elimination ===" << std::endl;

    for (auto& function : module) {
        if (function->empty()) {
            std::cout << "Skipping empty function: " << function->getName()
                      << std::endl;
            continue;
        }

        std::cout << "Processing function: " << function->getName()
                  << std::endl;

        FrameIndexElimination elimination(function.get());
        elimination.run();
    }

    std::cout << "=== Frame Index Elimination Completed ===" << std::endl;
    return module;
}

}  // namespace riscv64