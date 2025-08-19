#include "Target.h"

#include "BasicBlockReordering.h"
#include "CodeGen.h"
#include "ConstantFoldingPass.h"
#include "DeadCodeElimination.h"
#include "FrameIndexElimination.h"
#include "IR/Function.h"
#include "Pass/Analysis/LoopInfo.h"
#include "RAGreedy/LiveIntervals.h"
#include "RAGreedy/RegAllocGreedy.h"
#include "RAGreedy/RegisterRewriter.h"
#include "RAGreedy/SlotIndexes.h"
#include "RegAllocChaitin.h"
#include "ValueReusePass.h"
#include "Visit.h"

// Debug output macro - only outputs when A_OUT_DEBUG is defined
#ifdef A_OUT_DEBUG
#define DEBUG_OUT() std::cout
#else
#define DEBUG_OUT() \
    if constexpr (false) std::cout
#endif

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
        DEBUG_OUT()
            << "No AnalysisManager provided for ValueReusePass, skipped. "
               "Pass `-O1` param to enable."
            << std::endl;
    }

    initialFrameIndexPass(riscv_module);  // 第一阶段
    constantFoldingPass(riscv_module);    // 第1.6阶段：常量折叠优化
    basicBlockReorderingPass(riscv_module);  // 第1.7阶段：基本块重排优化

    deadCodeEliminationPass(riscv_module, false);  // 第一阶段附加：DCE

    // RAGreedyPass(riscv_module);

    registerAllocationPass(riscv_module, analysisManager);  // 第二阶段
    frameIndexEliminationPass(riscv_module);                // 第三阶段

    foldMemoryAccessPass(riscv_module);

    deadCodeEliminationPass(riscv_module, true);  // 第一阶段附加：DCE


    return riscv_module.toString();
}

Module& RISCV64Target::valueReusePass(
    riscv64::Module& riscv_module, const midend::Module& midend_module,
    const midend::AnalysisManager* analysisManager) {
    DEBUG_OUT() << "\n=== Phase 0.5: Value Reuse Optimization (Dominator Tree "
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
                DEBUG_OUT() << "Running ValueReusePass on function: "
                            << riscv_function->getName()
                            << " (midend: " << midend_function->getName() << ")"
                            << std::endl;

                bool optimized = pass.runOnFunction(
                    riscv_function.get(), midend_function, analysisManager);
                if (optimized) {
                    const auto& stats = pass.getStatistics();
                    DEBUG_OUT()
                        << "  Optimization results: " << stats.loadsEliminated
                        << " loads eliminated, " << stats.virtualRegsReused
                        << " registers reused" << std::endl;
                }
            } else {
                DEBUG_OUT() << "No corresponding midend function found for: "
                            << riscv_function->getName() << std::endl;
            }
        }
    }

    DEBUG_OUT() << "=== Value Reuse Optimization Completed ===" << std::endl;
    return riscv_module;
}

Module RISCV64Target::instructionSelectionPass(const midend::Module& module) {
    DEBUG_OUT() << "\n=== Phase 1: Instruction Selection ===" << std::endl;
    CodeGenerator codegen;
    auto riscv_module = codegen.visitor_->visit(&module);
    DEBUG_OUT() << module.toString() << std::endl;
    return riscv_module;
}

Module& RISCV64Target::constantFoldingPass(riscv64::Module& module) {
    DEBUG_OUT() << "\n=== Phase 1.6: Constant Folding ===" << std::endl;

    for (auto& function : module) {
        if (function->empty()) continue;

        DEBUG_OUT() << "Processing function: " << function->getName()
                    << std::endl;

        ConstantFolding pass;
        pass.runOnFunction(function.get());
    }

    DEBUG_OUT() << "=== Constant Folding Completed ===" << std::endl;
    DEBUG_OUT() << module.toString() << std::endl;

    return module;
}

Module& RISCV64Target::initialFrameIndexPass(riscv64::Module& module) {
    DEBUG_OUT() << "\n=== Phase 1.5: Initial Frame Index Creation ==="
                << std::endl;

    // 第一阶段已经在指令选择中完成了alloca的Frame Index创建
    // 这里只需要确保所有alloca都有对应的抽象Frame Index
    for (auto& function : module) {
        if (function->empty()) continue;

        DEBUG_OUT() << "Verifying abstract Frame Indices for function: "
                    << function->getName() << std::endl;

        // 验证所有frameaddr指令都有有效的Frame Index
        for (auto& bb : *function) {
            for (auto& inst : *bb) {
                if (inst->getOpcode() == Opcode::FRAMEADDR) {
                    const auto& operands = inst->getOperands();
                    if (operands.size() >= 2) {
                        if (auto* fi = dynamic_cast<FrameIndexOperand*>(
                                operands[1].get())) {
                            DEBUG_OUT() << "  Found abstract FI("
                                        << fi->getIndex() << ")" << std::endl;
                        }
                    }
                }
            }
        }
    }

    DEBUG_OUT() << module.toString() << std::endl;

    return module;
}

Module& RISCV64Target::basicBlockReorderingPass(riscv64::Module& module) {
    DEBUG_OUT() << "\n=== Phase 1.7: Basic Block Reordering ===" << std::endl;

    for (auto& function : module) {
        if (function->empty()) {
            DEBUG_OUT() << "Skipping empty function: " << function->getName()
                        << std::endl;
            continue;
        }

        DEBUG_OUT() << "Processing function: " << function->getName()
                    << std::endl;

        BasicBlockReordering reordering(function.get());
        reordering.run();
    }

    DEBUG_OUT() << "=== Basic Block Reordering Completed ===" << std::endl;
    DEBUG_OUT() << module.toString() << std::endl;

    return module;
}

Module& RISCV64Target::RAGreedyPass(riscv64::Module& module) {
    DEBUG_OUT() << "\n=== Phase 2.0: RA Greedy ===" << std::endl;

    DEBUG_OUT() << "=== Float Register Allocation Phase ===\n";
    SlotIndexesWrapperPass wrapper0;
    for (auto& function : module) {
        DEBUG_OUT() << "=== SlotIndexGeneration ===\n";
        wrapper0.runOnFunction(function.get());
        auto& SI = wrapper0.getSI();
        SI.print(std::cout);

        DEBUG_OUT() << "=== LiveIntervalGeneration ===\n";
        auto LISFloat =
            std::make_unique<LiveIntervals>(function.get(), &SI, true);
        LISFloat->analyze(*function);
        LISFloat->print(std::cout);

        DEBUG_OUT() << "=== RAGreedy ===\n";
        auto RAGreedyFloat =
            RegAllocGreedy(function.get(), LISFloat.get(), true);
        RAGreedyFloat.run();
        RAGreedyFloat.print(std::cout);

        DEBUG_OUT() << "=== RegisterRewrite ===\n";
        auto rewriterFloat =
            RegisterRewriter(function.get(), RAGreedyFloat.getVRM(), true);
        rewriterFloat.rewrite();

        DEBUG_OUT() << function->toString() << std::endl;
    }

    DEBUG_OUT() << "=== Integer Register Allocation Phase ===\n";
    SlotIndexesWrapperPass wrapper1;
    for (auto& function : module) {
        DEBUG_OUT() << "=== SlotIndexGeneration ===\n";

        wrapper1.runOnFunction(function.get());
        auto& SI = wrapper1.getSI();
        SI.print(std::cout);

        DEBUG_OUT() << "=== LiveIntervalGeneration ===\n";
        auto LISInt = std::make_unique<LiveIntervals>(function.get(), &SI);
        LISInt->analyze(*function);
        LISInt->print(std::cout);

        DEBUG_OUT() << "=== RAGreedy ===\n";
        auto RAGreedyInt = RegAllocGreedy(function.get(), LISInt.get());
        RAGreedyInt.run();
        RAGreedyInt.print(std::cout);

        DEBUG_OUT() << "=== RegisterRewrite ===\n";
        auto rewriterInt =
            RegisterRewriter(function.get(), RAGreedyInt.getVRM());
        rewriterInt.rewrite();

        DEBUG_OUT() << function->toString() << std::endl;
    }

    return module;
}

Module& RISCV64Target::registerAllocationPass(
    riscv64::Module& module, const midend::AnalysisManager* analysisManager) {
    DEBUG_OUT() << "\n=== Phase 2: Register Allocation ===" << std::endl;

    for (auto& function : module) {
        midend::Function* midend_function =
            module.getMidendFunction(function->getName());

        midend::LoopInfo* loopAnal = nullptr;
        if (midend_function && analysisManager) {
            auto name = midend::LoopAnalysis::getName();

            loopAnal =
                const_cast<midend::AnalysisManager*>(analysisManager)
                    ->getAnalysis<midend::LoopInfo>(name, *midend_function);
        }

        DEBUG_OUT() << "RegAlloc for float" << std::endl;
        RegAllocChaitin allocatorFloat(function.get(), true, loopAnal);
        allocatorFloat.run();

        DEBUG_OUT() << function->toString() << std::endl;

        DEBUG_OUT() << "RegAlloc for int" << std::endl;
        RegAllocChaitin allocatorInt(function.get(), false, loopAnal);
        allocatorInt.run();

        DEBUG_OUT() << function->toString() << std::endl;
    }

    DEBUG_OUT() << module.toString() << std::endl;

    return module;
}

Module& RISCV64Target::deadCodeEliminationPass(riscv64::Module& module, bool forPhys) {
    DEBUG_OUT() << "\n=== Post-RA Dead Code Elimination ===" << std::endl;
    DeadCodeElimination dce;
    for (auto& function : module) {
        if (function->empty()) {
            continue;
        }
        bool changed = dce.runOnFunction(function.get(), forPhys);
        if (changed) {
            DEBUG_OUT() << "[DCE] Function optimized: " << function->getName()
                        << std::endl;
        }
    }
    DEBUG_OUT() << "=== DCE Completed ===" << std::endl;
    return module;
}

Module& RISCV64Target::frameIndexEliminationPass(riscv64::Module& module) {
    DEBUG_OUT() << "\n=== Phase 3: Frame Index Elimination ===" << std::endl;

    for (auto& function : module) {
        if (function->empty()) {
            DEBUG_OUT() << "Skipping empty function: " << function->getName()
                        << std::endl;
            continue;
        }

        DEBUG_OUT() << "Processing function: " << function->getName()
                    << std::endl;

        FrameIndexElimination elimination(function.get());
        elimination.run();
    }

    DEBUG_OUT() << "=== Frame Index Elimination Completed ===" << std::endl;
    return module;
}

}  // namespace riscv64