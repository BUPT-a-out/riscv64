#include "CopyPropagationPass.h"

#include <algorithm>
#include <queue>
#include <variant>

#include "Instructions/All.h"
#include "Visit.h"

#ifdef A_OUT_DEBUG
#define CP_DEBUG_OUT() std::cout
#else
#define CP_DEBUG_OUT() \
    if constexpr (false) std::cout
#endif

namespace riscv64 {

void CopyPropagationPass::runOnFunction(Function* function) {
    if (function->empty()) {
        return;
    }

    CP_DEBUG_OUT() << "Running Copy Propagation on function: "
                   << function->getName() << std::endl;

    clear();

    // 阶段1：计算gen和kill集合
    computeGenKill(function);

    // 阶段2：迭代数据流分析
    iterativeDataFlow(function);

    // 阶段3：代码转换
    transformCode(function);

    // 清理标记的指令
    removeMarkedInstructions(function);

    CP_DEBUG_OUT() << "Copy Propagation completed for function: "
                   << function->getName() << std::endl;
}

void CopyPropagationPass::clear() {
    genKillSeq.clear();
    copyIn.clear();
    copyOut.clear();
    instructionsToRemove.clear();
}

// 阶段1：计算gen和kill集合
void CopyPropagationPass::computeGenKill(Function* function) {
    CP_DEBUG_OUT() << "Phase 1: Computing gen and kill sets" << std::endl;

    for (auto& basicBlock : *function) {
        processBasicBlock(basicBlock.get());
    }
}

void CopyPropagationPass::processBasicBlock(BasicBlock* basicBlock) {
    CopyInfo currentGen;
    std::vector<GenOrKill> newSeq;

    for (auto& inst : *basicBlock) {
        // 获取指令定义和使用的寄存器
        auto defs = inst->getDefinedIntegerRegs();
        auto uses = inst->getUsedIntegerRegs();

        // 如果是复写指令，添加到gen集合
        if (inst->isCopyInstr()) {
            auto dest = inst->getOperand(0)->getRegNum();
            auto src = inst->getOperand(1)->getRegNum();

            if (isVirtualRegister(dest) && isVirtualRegister(src)) {
                newSeq.push_back(CopyPair(dest, src));
                CP_DEBUG_OUT() << "  Generated copy: " << dest << " <- " << src
                               << std::endl;
            }
        } else {
            // 处理定义的寄存器（添加到kill集合）
            for (unsigned def : defs) {
                if (isVirtualRegister(def)) {
                    newSeq.push_back(def);
                }
            }
        }
    }

    genKillSeq[basicBlock] = newSeq;

    CP_DEBUG_OUT() << "BasicBlock " << basicBlock->getLabel()
                   << ": gen/kill size=" << genKillSeq[basicBlock].size()
                   << std::endl;
}

// 阶段2：迭代数据流分析
void CopyPropagationPass::iterativeDataFlow(Function* function) {
    CP_DEBUG_OUT() << "Phase 2: Iterative dataflow analysis" << std::endl;

    // 初始化copyIn和copyOut
    for (auto& basicBlock : *function) {
        copyIn[basicBlock.get()] = CopyInfo{};
        copyOut[basicBlock.get()] = CopyInfo{};
    }

    bool changed = true;
    int iteration = 0;

    while (changed) {
        changed = false;
        iteration++;
        CP_DEBUG_OUT() << "  Iteration " << iteration << std::endl;

        for (auto& basicBlock : *function) {
            BasicBlock* currentBasicBlock = basicBlock.get();

            // 计算CopyIn[bb] = 交集(CopyOut[pred] for pred in predecessors)
            CopyInfo newCopyIn;
            auto predecessors = currentBasicBlock->getPredecessors();

            if (!predecessors.empty()) {
                std::vector<CopyInfo> predCopyOuts;
                predCopyOuts.reserve(predecessors.size());
                for (BasicBlock* pred : predecessors) {
                    predCopyOuts.push_back(copyOut[pred]);
                }
                newCopyIn = intersectCopyInfo(predCopyOuts);
            }

            if (newCopyIn != copyIn[currentBasicBlock]) {
                copyIn[currentBasicBlock] = newCopyIn;
                changed = true;
            }

            auto seq = genKillSeq[basicBlock.get()];
            CopyInfo newCopyOut = newCopyIn;

            // 合并存活的复写和新生成的复写
            // TODO: use a map.
            for (auto& op : seq) {
                if (auto copy = std::get_if<CopyPair>(&op)) {
                    auto dst = copy->dest();
                    auto src = copy->src();
                    auto s = findUltimateSource(newCopyOut, src);

                    auto it = newCopyOut.begin();
                    while (it != newCopyOut.end()) {
                        if (it->involves(dst)) {
                            it = newCopyOut.erase(it);
                        } else {
                            ++it;
                        }
                    }

                    // 加入新的 CopyPair{dst, s}
                    newCopyOut.insert(CopyPair{dst, s});
                } else if (auto kill = std::get_if<unsigned>(&op)) {
                    auto it = newCopyOut.begin();
                    while (it != newCopyOut.end()) {
                        if (it->involves(*kill)) {
                            it = newCopyOut.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }

            if (newCopyOut != copyOut[currentBasicBlock]) {
                copyOut[currentBasicBlock] = newCopyOut;
                changed = true;
            }
        }
    }

    CP_DEBUG_OUT() << "  Dataflow analysis converged after " << iteration
                   << " iterations" << std::endl;
}

auto CopyPropagationPass::intersectCopyInfo(const std::vector<CopyInfo>& sets)
    -> CopyInfo {
    if (sets.empty()) {
        return CopyInfo{};
    }

    CopyInfo result = sets[0];
    for (size_t i = 1; i < sets.size(); i++) {
        CopyInfo intersection;
        std::set_intersection(
            result.begin(), result.end(), sets[i].begin(), sets[i].end(),
            std::inserter(intersection, intersection.begin()));
        result = std::move(intersection);
    }

    return result;
}

// auto CopyPropagationPass::applyCopyKill(
//     const CopyInfo& copySet, const std::unordered_set<unsigned>& killSet)
//     -> CopyInfo {
//     CopyInfo result;

//     for (const auto& copy : copySet) {
//         bool killed = false;
//         for (unsigned killedReg : killSet) {
//             if (copy.involves(killedReg)) {
//                 killed = true;
//                 break;
//             }
//         }

//         if (!killed) {
//             result.insert(copy);
//         }
//     }

//     return result;
// }

// 阶段3：代码转换
void CopyPropagationPass::transformCode(Function* function) {
    CP_DEBUG_OUT() << "Phase 3: Code transformation" << std::endl;

    for (auto& basicBlock : *function) {
        transformBasicBlock(basicBlock.get());
    }
}

void CopyPropagationPass::transformBasicBlock(BasicBlock* basicBlock) {
    CopyInfo currentCopies = copyIn[basicBlock];

    for (auto& inst : *basicBlock) {
        // 如果是复写指令，添加新的复写关系并标记删除
        if (inst->isCopyInstr()) {
            auto dest = inst->getOperand(0)->getRegNum();
            auto src = inst->getOperand(1)->getRegNum();

            if (isVirtualRegister(dest) && isVirtualRegister(src)) {
                auto ss = findUltimateSource(currentCopies, src);
                auto ds = findUltimateSource(currentCopies, dest);

                if (ss == ds) {
                    markInstructionForRemoval(inst.get());
                } else {
                    auto iter = currentCopies.begin();
                    while (iter != currentCopies.end()) {
                        if (iter->involves(dest)) {
                            iter = currentCopies.erase(iter);
                        } else {
                            ++iter;
                        }
                    }
                    currentCopies.insert(CopyPair(dest, ss));
                }
                CP_DEBUG_OUT() << "  Marked copy instruction for removal: "
                               << inst->toString() << std::endl;
            }
        } else {
            // 替换使用的寄存器
            auto uses = inst->getUsedIntegerRegs();
            for (size_t i = 0; i < inst->getOprandCount(); i++) {
                auto* operand = inst->getOperand(i);
                if (operand->isReg() &&
                    isVirtualRegister(operand->getRegNum())) {
                    unsigned reg = operand->getRegNum();

                    // 检查是否是使用操作数（而非定义操作数）
                    bool isUse =
                        std::find(uses.begin(), uses.end(), reg) != uses.end();
                    if (isUse) {
                        // 查找复写关系
                        for (const auto& copy : currentCopies) {
                            if (copy.dest() == reg) {
                                // 替换寄存器
                                operand->setRegNum(copy.src());
                                CP_DEBUG_OUT()
                                    << "  Replaced " << reg << " with "
                                    << copy.src()
                                    << " in instruction: " << inst->toString()
                                    << std::endl;
                                break;
                            }
                        }
                    }
                }
            }
            // 更新当前复写集合
            auto defs = inst->getDefinedIntegerRegs();
            for (unsigned def : defs) {
                if (isVirtualRegister(def)) {
                    // 移除涉及该寄存器的复写关系
                    auto iter = currentCopies.begin();
                    while (iter != currentCopies.end()) {
                        if (iter->involves(def)) {
                            iter = currentCopies.erase(iter);
                        } else {
                            ++iter;
                        }
                    }
                }
            }
        }
    }
}

unsigned CopyPropagationPass::findUltimateSource(CopyInfo& copyInfo,
                                                 unsigned s) {
    bool found;
    do {
        found = false;
        for (const auto& existingCopy : copyInfo) {
            if (existingCopy.dest() == s) {
                s = existingCopy.src();  // 继续追溯
                found = true;
                break;
            }
        }
    } while (found);  // 直到找不到更早的源头
    return s;
}

// 辅助方法实现
// auto CopyPropagationPass::isCopyInstruction(Instruction* inst) -> bool {
//     return inst->getOpcode() == Opcode::MV && inst->getOprandCount() == 2 &&
//            inst->getOperand(0)->isReg() && inst->getOperand(1)->isReg();
// }

auto CopyPropagationPass::isVirtualRegister(unsigned reg) -> bool {
    return reg >= VIRTUAL_REG_START;
}

void CopyPropagationPass::markInstructionForRemoval(Instruction* inst) {
    instructionsToRemove.push_back(inst);
}

void CopyPropagationPass::removeMarkedInstructions(Function* function) {
    std::unordered_set<Instruction*> toRemove(instructionsToRemove.begin(),
                                              instructionsToRemove.end());

    for (auto& basicBlock : *function) {
        auto iter = basicBlock->begin();
        while (iter != basicBlock->end()) {
            if (toRemove.count(iter->get()) > 0) {
                CP_DEBUG_OUT() << "Removed instruction: " << (*iter)->toString()
                               << std::endl;
                iter = basicBlock->erase(iter);
            } else {
                ++iter;
            }
        }
    }
}

}  // namespace riscv64