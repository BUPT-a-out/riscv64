#include "SpillCodeOptimizer.h"
namespace riscv64 {

// It only plays with integer regs.
void SpillCodeOptimizer::optimizeSpillCode(Function* function) {
    std::cout << "Starting spill code optimization..." << std::endl;
    removeRedundantFrameAddr(function);
    std::cout << "Spill code optimization completed." << std::endl;
}

void SpillCodeOptimizer::removeRedundantFrameAddr(Function* function) {
    for (auto& bb : *function) {
        std::unordered_map<int, unsigned>
            frameToRegMap;  // frameIndex -> regNum
        std::unordered_map<unsigned, int>
            regToFrameMap;  // regNum -> frameIndex
        std::vector<BasicBlock::iterator> toErase;

        std::cout << "Processing basic block " << bb->getLabel() << " with "
                  << bb->size() << " instructions" << std::endl;

        for (auto it = bb->begin(); it != bb->end(); ++it) {
            Instruction* inst = it->get();
            int frameIndex = -1;
            unsigned dstReg = 0;

            // 检查是否是frameaddr指令
            if (isFrameAddrInstruction(inst, frameIndex, dstReg)) {
                std::cout << "Found frameaddr: reg=" << dstReg
                          << ", FI=" << frameIndex << std::endl;

                // 使用frameToRegMap快速查找是否已经有寄存器保存了相同frameIndex的地址
                auto existingIt = frameToRegMap.find(frameIndex);
                if (existingIt != frameToRegMap.end()) {
                    unsigned existingReg = existingIt->second;
                    std::cout << "Found redundant frameaddr! Replacing reg "
                              << dstReg << " with existing reg " << existingReg
                              << std::endl;

                    // 用现有寄存器替换当前寄存器的所有后续使用
                    replaceIntegerRegisterInBasicBlock(bb.get(), dstReg,
                                                       existingReg);

                    // 标记删除冗余指令
                    toErase.push_back(it);
                    continue;
                } else {
                    // 如果目标寄存器之前保存了其他frameIndex，先清除旧映射
                    auto oldFrameIt = regToFrameMap.find(dstReg);
                    if (oldFrameIt != regToFrameMap.end()) {
                        int oldFrameIndex = oldFrameIt->second;
                        frameToRegMap.erase(oldFrameIndex);
                        std::cout << "Register " << dstReg << " was holding FI("
                                  << oldFrameIndex << "), clearing old mapping"
                                  << std::endl;
                    }

                    // 建立新的双向映射
                    frameToRegMap[frameIndex] = dstReg;
                    regToFrameMap[dstReg] = frameIndex;
                    std::cout << "Cached frameaddr: FI(" << frameIndex
                              << ") <-> reg" << dstReg << std::endl;
                }
            } else {
                // 检查指令是否重新定义了缓存中的寄存器，如果重新定义了就清除缓存
                auto definedRegs = inst->getDefinedIntegerRegs();

                for (unsigned reg : definedRegs) {
                    auto regFrameIt = regToFrameMap.find(reg);
                    if (regFrameIt != regToFrameMap.end()) {
                        int frameIndex = regFrameIt->second;
                        std::cout << "Register " << reg
                                  << " redefined, invalidating FI("
                                  << frameIndex << ")" << std::endl;

                        // 同时从两个映射中删除
                        regToFrameMap.erase(regFrameIt);
                        frameToRegMap.erase(frameIndex);
                    }
                }
            }
        }

        // 删除标记的冗余指令
        std::cout << "Erasing " << toErase.size()
                  << " redundant frameaddr instructions" << std::endl;
        for (auto it : toErase) {
            bb->erase(it);
        }
    }
}

bool SpillCodeOptimizer::isFrameAddrInstruction(Instruction* inst,
                                                int& frameIndex,
                                                unsigned& dstReg) {
    if (inst->getOpcode() != Opcode::FRAMEADDR) {
        return false;
    }

    const auto& operands = inst->getOperands();
    if (operands.size() < 2) {
        return false;
    }

    // 第一个操作数应该是目标寄存器
    if (!operands[0]->isReg()) {
        return false;
    }

    // 第二个操作数应该是FrameIndex
    if (!operands[1]->isFrameIndex()) {
        return false;
    }

    dstReg = static_cast<RegisterOperand*>(operands[0].get())->getRegNum();
    frameIndex = static_cast<FrameIndexOperand*>(operands[1].get())->getIndex();
    return true;
}

void SpillCodeOptimizer::replaceIntegerRegisterInBasicBlock(BasicBlock* bb,
                                                            unsigned oldReg,
                                                            unsigned newReg) {
    if (oldReg == newReg) {
        return;
    }
    std::cout << "Replacing register " << oldReg << " with " << newReg
              << " in basic block" << std::endl;

    for (auto& inst : *bb) {
        const auto& operands = inst->getOperands();
        for (const auto& operand : operands) {
            if (operand->isReg()) {
                RegisterOperand* regOp =
                    static_cast<RegisterOperand*>(operand.get());
                if (regOp->getRegNum() == oldReg &&
                    regOp->isIntegerRegister()) {
                    std::cout << "  Replacing register operand " << oldReg
                              << " -> " << newReg << std::endl;
                    regOp->setRegNum(newReg);
                }
            } else if (operand->isMem()) {
                MemoryOperand* memOp =
                    static_cast<MemoryOperand*>(operand.get());
                if (memOp->getBaseReg() &&
                    memOp->getBaseReg()->getRegNum() == oldReg &&
                    memOp->getBaseReg()->isIntegerRegister()) {
                    std::cout << "  Replacing memory base register " << oldReg
                              << " -> " << newReg << std::endl;
                    memOp->getBaseReg()->setRegNum(newReg);
                }
            }
        }
    }
}

}  // namespace riscv64
