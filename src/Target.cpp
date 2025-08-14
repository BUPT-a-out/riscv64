#include "Target.h"

#include "BasicBlockReordering.h"
#include "CodeGen.h"
#include "ConstantFoldingPass.h"
#include "FrameIndexElimination.h"
#include "IR/Function.h"
#include "RAGreedy/LiveIntervals.h"
#include "RAGreedy/RegAllocGreedy.h"
#include "RAGreedy/RegisterRewriter.h"
#include "RAGreedy/SlotIndexes.h"
#include "RegAllocChaitin.h"
#include "ValueReusePass.h"
#include "Visit.h"

namespace riscv64 {

std::string RISCV64Target::compileToAssembly(
    const midend::Module& module,
    const midend::AnalysisManager* analysisManager) {
    std::vector<std::string> assembly;

    // 添加汇编头部
    assembly.emplace_back(".text");
    assembly.emplace_back(".global _start");

    // 执行完整的三阶段编译流程
    auto riscv_module = instructionSelectionPass(module);

    // 在寄存器分配之前运行Value Reuse优化，现在传递AnalysisManager
    if (analysisManager != nullptr) {
        valueReusePass(riscv_module, module, analysisManager);
    } else {
        std::cout << "No AnalysisManager provided for ValueReusePass, skipped. "
                     "Pass `-O1` param to enable."
                  << std::endl;
    }

    initialFrameIndexPass(riscv_module);     // 第一阶段
    constantFoldingPass(riscv_module);       // 第1.6阶段：常量折叠优化
    basicBlockReorderingPass(riscv_module);  // 第1.7阶段：基本块重排优化

    // slotIndexWrapperPass(riscv_module);

    registerAllocationPass(riscv_module);     // 第二阶段
    frameIndexEliminationPass(riscv_module);  // 第三阶段

    return riscv_module.toString();
}

Module& RISCV64Target::valueReusePass(
    riscv64::Module& riscv_module, const midend::Module& midend_module,
    const midend::AnalysisManager* analysisManager) {
    std::cout << "\n=== Phase 0.5: Value Reuse Optimization (Dominator Tree "
                 "Based) ==="
              << std::endl;

    ValueReusePass pass;

    // Process each RISCV64 function with its corresponding midend function
    for (auto& riscv_function : riscv_module) {
        if (!riscv_function->empty()) {
            // Find corresponding midend function by name
            const midend::Function* midend_function = nullptr;
            for (const auto& midend_func : midend_module) {
                if (midend_func->getName() == riscv_function->getName()) {
                    midend_function = midend_func;
                    break;
                }
            }

            if (midend_function != nullptr) {
                std::cout << "Running ValueReusePass on function: "
                          << riscv_function->getName()
                          << " (midend: " << midend_function->getName() << ")"
                          << std::endl;

                bool optimized = pass.runOnFunction(
                    riscv_function.get(), midend_function, analysisManager);
                if (optimized) {
                    const auto& stats = pass.getStatistics();
                    std::cout
                        << "  Optimization results: " << stats.loadsEliminated
                        << " loads eliminated, " << stats.virtualRegsReused
                        << " registers reused" << std::endl;
                }
            } else {
                std::cout << "No corresponding midend function found for: "
                          << riscv_function->getName() << std::endl;
            }
        }
    }

    std::cout << "=== Value Reuse Optimization Completed ===" << std::endl;
    return riscv_module;
}

Module RISCV64Target::instructionSelectionPass(const midend::Module& module) {
    std::cout << "\n=== Phase 1: Instruction Selection ===" << std::endl;
    CodeGenerator codegen;
    auto riscv_module = codegen.visitor_->visit(&module);
    std::cout << module.toString() << std::endl;
    return riscv_module;
}

Module& RISCV64Target::constantFoldingPass(riscv64::Module& module) {
    std::cout << "\n=== Phase 1.6: Constant Folding ===" << std::endl;

    for (auto& function : module) {
        if (function->empty()) continue;

        std::cout << "Processing function: " << function->getName()
                  << std::endl;

        ConstantFolding pass;
        pass.runOnFunction(function.get());
    }

    std::cout << "=== Constant Folding Completed ===" << std::endl;
    std::cout << module.toString() << std::endl;

    return module;
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

// TODO: rename fn
Module& RISCV64Target::slotIndexWrapperPass(riscv64::Module& module) {
    std::cout << "\n=== Phase 2.0: SlotIndexGeneration ===" << std::endl;

    SlotIndexesWrapperPass wrapper0;
    for (auto& function : module) {
        wrapper0.runOnFunction(function.get());
        auto& SI = wrapper0.getSI();
        SI.print(std::cout);

        auto LISFloat =
            std::make_unique<LiveIntervals>(function.get(), &SI, true);
        LISFloat->analyze(*function);
        LISFloat->print(std::cout);

        auto RAGreedyFloat =
            RegAllocGreedy(function.get(), LISFloat.get(), true);
        RAGreedyFloat.run();
        RAGreedyFloat.print(std::cout);

        auto rewriterFloat =
            RegisterRewriter(function.get(), RAGreedyFloat.getVRM());
        rewriterFloat.rewrite();
    }

    std::cout << "Alloc for float" << std::endl;
    std::cout << module.toString() << std::endl;

    SlotIndexesWrapperPass wrapper1;
    for (auto& function : module) {
        wrapper1.runOnFunction(function.get());
        auto& SI = wrapper1.getSI();
        SI.print(std::cout);

        auto LISInt = std::make_unique<LiveIntervals>(function.get(), &SI);
        LISInt->analyze(*function);
        LISInt->print(std::cout);

        auto RAGreedyInt = RegAllocGreedy(function.get(), LISInt.get());
        RAGreedyInt.run();
        RAGreedyInt.print(std::cout);

        auto rewriterInt =
            RegisterRewriter(function.get(), RAGreedyInt.getVRM());
        rewriterInt.rewrite();
    }

    std::cout << "Alloc for integer" << std::endl;
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