#include "FrameIndexElimination.h"

#include <algorithm>
#include <iostream>
#include <set>

#include "Instructions/All.h"

namespace riscv64 {

void FrameIndexElimination::run() {
    std::cout << "\n=== Running Frame Index Elimination (Phase 3) ==="
              << std::endl;

    // 第三阶段：计算最终布局并消除Frame Index
    computeFinalFrameLayout();
    generateFinalPrologueEpilogue();
    eliminateFrameIndices();

    printFinalLayout();
}

void FrameIndexElimination::computeFinalFrameLayout() {
    std::cout << "Computing final frame layout for function: "
              << function->getName() << std::endl;

    // 收集所有栈对象（alloca + spill slots）
    std::vector<StackObject*> allStackObjects;
    std::set<int> allFrameIndices;

    // 从指令中收集所有使用的Frame Index
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            if (inst->getOpcode() == Opcode::FRAMEADDR) {
                const auto& operands = inst->getOperands();
                if (operands.size() >= 2) {
                    if (auto* fi = dynamic_cast<FrameIndexOperand*>(
                            operands[1].get())) {
                        allFrameIndices.insert(fi->getIndex());
                    }
                }
            }
        }
    }

    // 获取对应的栈对象
    int totalLocalVarSize = 0;
    int totalSpillSize = 0;

    for (int fi : allFrameIndices) {
        auto* obj = stackManager->getStackObjectByIdentifier(fi);
        if (obj) {
            allStackObjects.push_back(obj);
            if (obj->type == StackObjectType::AllocatedStackSlot) {
                totalLocalVarSize += alignTo(obj->size, obj->alignment);
            } else if (obj->type == StackObjectType::SpilledRegister) {
                totalSpillSize += alignTo(obj->size, obj->alignment);
            }
            std::cout << "Found FI(" << fi << ") type=" << (int)obj->type
                      << ", size=" << obj->size << std::endl;
        }
    }

    assignFinalOffsets();

    std::cout << "Final layout - Local vars: " << totalLocalVarSize
              << " bytes, Spills: " << totalSpillSize << " bytes" << std::endl;
}

void FrameIndexElimination::assignFinalOffsets() {
    // 计算保存寄存器需要的空间
    int savedRegSize = calculateSavedRegisterSize();

    // 计算局部变量区大小
    int localVarSize = 0;
    int spillSize = 0;

    for (const auto& obj : stackManager->getAllStackObjects()) {
        if (obj->type == StackObjectType::AllocatedStackSlot) {
            localVarSize += alignTo(obj->size, obj->alignment);
        } else if (obj->type == StackObjectType::SpilledRegister) {
            spillSize += alignTo(obj->size, obj->alignment);
        }
    }

    // 计算调用参数所需的栈空间
    int callArgSize = calculateMaxCallArgSize();

    // 计算总栈帧大小：基础保存寄存器 + 局部变量 + 溢出寄存器 + 调用参数
    int totalSize = savedRegSize + localVarSize + spillSize + callArgSize;
    layout.totalFrameSize = alignTo(totalSize, 16);  // 16字节对齐

    // 计算各区域偏移
    layout.returnAddressOffset = layout.totalFrameSize - 8;  // ra
    layout.framePointerOffset = layout.totalFrameSize - 16;  // s0
    layout.localVariableAreaOffset = 0;                      // 从s0开始向下
    layout.spillAreaOffset = -localVarSize;                  // 在局部变量区下方

    // 为每个Frame Index分配具体偏移 (相对于s0)
    int currentLocalOffset = 0;
    int currentSpillOffset = layout.spillAreaOffset;

    for (const auto& obj : stackManager->getAllStackObjects()) {
        if (obj->type == StackObjectType::AllocatedStackSlot) {
            currentLocalOffset -= alignTo(obj->size, obj->alignment);
            layout.frameIndexToOffset[obj->identifier] = currentLocalOffset;
            std::cout << "FI(" << obj->identifier << ") [alloca] -> s0"
                      << currentLocalOffset << std::endl;
        } else if (obj->type == StackObjectType::SpilledRegister) {
            currentSpillOffset -= alignTo(obj->size, obj->alignment);
            layout.frameIndexToOffset[obj->identifier] = currentSpillOffset;
            std::cout << "FI(" << obj->identifier << ") [spill reg "
                      << obj->regNum << "] -> s0" << currentSpillOffset
                      << std::endl;
        }
    }
}

int FrameIndexElimination::calculateSavedRegisterSize() {
    // 分析函数中使用的callee-saved寄存器
    auto usedSavedRegs = collectSavedRegisters();

    return usedSavedRegs.size() * 8;  // 每个寄存器8字节
}

int FrameIndexElimination::calculateMaxCallArgSize() {
    int maxArgSize = 0;

    // 扫描所有call指令，找出最大的栈参数需求
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            if (inst->getOpcode() == Opcode::CALL) {
                // 查找在这个call之前的参数准备指令
                int argSize = 0;
                auto it = std::find_if(
                    bb->begin(), bb->end(),
                    [&inst](const std::unique_ptr<Instruction>& i) {
                        return i.get() == inst.get();
                    });

                // 向前扫描，查找sw指令到栈顶的最大偏移
                auto rev_it = std::make_reverse_iterator(it);
                for (auto rit = rev_it; rit != bb->rend(); ++rit) {
                    if ((*rit)->getOpcode() == Opcode::SW) {
                        const auto& operands = (*rit)->getOperands();
                        if (operands.size() >= 2) {
                            if (auto* memOp = dynamic_cast<MemoryOperand*>(
                                    operands[1].get())) {
                                if (auto* baseReg =
                                        dynamic_cast<RegisterOperand*>(
                                            memOp->getBaseReg())) {
                                    if (baseReg->getRegNum() == 2) {  // sp
                                        if (auto* offset =
                                                dynamic_cast<ImmediateOperand*>(
                                                    memOp->getOffset())) {
                                            argSize = std::max(
                                                argSize,
                                                static_cast<int>(
                                                    offset->getValue()) +
                                                    4);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                maxArgSize = std::max(maxArgSize, argSize);
            }
        }
    }

    return alignTo(maxArgSize, 16);  // 16字节对齐
}

void FrameIndexElimination::generateFinalPrologueEpilogue() {
    if (layout.totalFrameSize == 0) {
        return;
    }

    std::cout << "Generating final prologue/epilogue for frame size: "
              << layout.totalFrameSize << std::endl;

    // 收集需要保存的寄存器
    std::vector<int> savedRegs = collectSavedRegisters();

    // 删除现有的栈管理指令
    removeExistingPrologueEpilogue();

    // 生成序言 (插入到函数开头)
    BasicBlock* entryBlock = function->getBasicBlock(0);
    if (entryBlock) {
        std::vector<std::unique_ptr<Instruction>> prologueInsts;

        // 1. 调整栈指针: addi sp, sp, -frameSize
        auto adjustSp = std::make_unique<Instruction>(Opcode::ADDI);
        adjustSp->addOperand(
            std::make_unique<RegisterOperand>(2, false));  // sp
        adjustSp->addOperand(
            std::make_unique<RegisterOperand>(2, false));  // sp
        adjustSp->addOperand(
            std::make_unique<ImmediateOperand>(-layout.totalFrameSize));
        prologueInsts.push_back(std::move(adjustSp));

        // 2. 保存所有需要保存的寄存器
        int offset = layout.totalFrameSize - 8;  // 从栈顶开始
        for (int regNum : savedRegs) {
            auto saveReg = std::make_unique<Instruction>(Opcode::SD);
            saveReg->addOperand(
                std::make_unique<RegisterOperand>(regNum, false));
            saveReg->addOperand(std::make_unique<MemoryOperand>(
                std::make_unique<RegisterOperand>(2, false),  // sp
                std::make_unique<ImmediateOperand>(offset)));
            prologueInsts.push_back(std::move(saveReg));
            offset -= 8;
        }

        // 3. 设置帧指针: addi s0, sp, frameSize
        auto setFp = std::make_unique<Instruction>(Opcode::ADDI);
        setFp->addOperand(std::make_unique<RegisterOperand>(8, false));  // s0
        setFp->addOperand(std::make_unique<RegisterOperand>(2, false));  // sp
        setFp->addOperand(
            std::make_unique<ImmediateOperand>(layout.totalFrameSize));
        prologueInsts.push_back(std::move(setFp));

        // 逆序插入以保持正确顺序
        for (auto it = prologueInsts.rbegin(); it != prologueInsts.rend();
             ++it) {
            entryBlock->insert(entryBlock->begin(), std::move(*it));
        }
    }

    // 生成尾声 (插入到所有ret指令前)
    for (auto& bb : *function) {
        // TODO: 找基本块最后一条更有效率
        for (auto it = bb->begin(); it != bb->end(); ++it) {
            if ((*it)->getOpcode() == Opcode::RET) {
                // 恢复所有保存的寄存器
                int offset = layout.totalFrameSize - 8;
                for (int regNum : savedRegs) {
                    auto restoreReg = std::make_unique<Instruction>(Opcode::LD);
                    restoreReg->addOperand(
                        std::make_unique<RegisterOperand>(regNum, false));
                    restoreReg->addOperand(std::make_unique<MemoryOperand>(
                        std::make_unique<RegisterOperand>(2, false),
                        std::make_unique<ImmediateOperand>(offset)));
                    it = bb->insert(it, std::move(restoreReg));
                    ++it;
                    offset -= 8;
                }

                // 恢复栈指针: addi sp, sp, frameSize
                auto restoreSp = std::make_unique<Instruction>(Opcode::ADDI);
                restoreSp->addOperand(
                    std::make_unique<RegisterOperand>(2, false));
                restoreSp->addOperand(
                    std::make_unique<RegisterOperand>(2, false));
                restoreSp->addOperand(
                    std::make_unique<ImmediateOperand>(layout.totalFrameSize));
                it = bb->insert(it, std::move(restoreSp));
                ++it;

                break;  // 每个基本块最多一个ret
            }
        }
    }
}

// savedreg 只有整数寄存器.
std::vector<int> FrameIndexElimination::collectSavedRegisters() {
    std::set<int> usedSavedRegs;
    usedSavedRegs.insert(1);  // ra
    usedSavedRegs.insert(8);  // s0/fp

    // 扫描所有指令，查找使用的s寄存器
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            for (const auto& operand : inst->getOperands()) {
                if (auto* regOp =
                        dynamic_cast<RegisterOperand*>(operand.get())) {
                    int regNum = regOp->getRegNum();
                    // s1-s11 对应寄存器号 9, 18-27
                    if (regOp->isIntegerRegister()) {
                        if (regNum == 9 || (regNum >= 18 && regNum <= 27)) {
                            usedSavedRegs.insert(regNum);
                        }
                    }
                }
            }
        }
    }

    return std::vector<int>(usedSavedRegs.begin(), usedSavedRegs.end());
}

// TODO: no this fucking shit
void FrameIndexElimination::removeExistingPrologueEpilogue() {
    // 删除现有的栈管理指令，避免重复
    for (auto& bb : *function) {
        for (auto it = bb->begin(); it != bb->end();) {
            bool shouldRemove = false;

            // 删除旧的栈调整指令
            if ((*it)->getOpcode() == Opcode::ADDI) {
                const auto& operands = (*it)->getOperands();
                if (operands.size() >= 3) {
                    auto* dest =
                        dynamic_cast<RegisterOperand*>(operands[0].get());
                    auto* src =
                        dynamic_cast<RegisterOperand*>(operands[1].get());
                    if (dest && src && dest->getRegNum() == 2 &&
                        src->getRegNum() == 2) {
                        // 这是栈指针调整指令
                        shouldRemove = true;
                    }
                }
            }

            // 删除旧的寄存器保存/恢复指令
            if ((*it)->getOpcode() == Opcode::SD ||
                (*it)->getOpcode() == Opcode::LD) {
                const auto& operands = (*it)->getOperands();
                if (operands.size() >= 2) {
                    if (auto* memOp =
                            dynamic_cast<MemoryOperand*>(operands[1].get())) {
                        if (auto* baseReg = dynamic_cast<RegisterOperand*>(
                                memOp->getBaseReg())) {
                            if (baseReg->getRegNum() ==
                                2) {  // 基于sp的内存访问
                                auto* regOp = dynamic_cast<RegisterOperand*>(
                                    operands[0].get());
                                if (regOp && (regOp->getRegNum() == 1 ||
                                              regOp->getRegNum() == 8 ||
                                              (regOp->getRegNum() >= 9 &&
                                               regOp->getRegNum() <= 19))) {
                                    // 这是保存/恢复寄存器的指令
                                    shouldRemove = true;
                                }
                            }
                        }
                    }
                }
            }

            if (shouldRemove) {
                it = bb->erase(it);
            } else {
                ++it;
            }
        }
    }
}

void FrameIndexElimination::eliminateFrameIndices() {
    std::cout << "Eliminating frameaddr instructions..." << std::endl;

    for (auto& bb : *function) {
        for (auto it = bb->begin(); it != bb->end();) {
            if ((*it)->getOpcode() == Opcode::FRAMEADDR) {
                eliminateFrameIndexInstruction(bb.get(), it);
                // it已经被更新，不需要++
            } else {
                ++it;
            }
        }
    }
}

void FrameIndexElimination::eliminateFrameIndexInstruction(
    BasicBlock* bb, std::list<std::unique_ptr<Instruction>>::iterator& it) {
    auto& inst = *it;
    const auto& operands = inst->getOperands();

    if (operands.size() < 2) {
        throw std::runtime_error("frameaddr instruction must have 2 operands");
    }

    auto* destReg = dynamic_cast<RegisterOperand*>(operands[0].get());
    auto* frameIndex = dynamic_cast<FrameIndexOperand*>(operands[1].get());

    if (!destReg || !frameIndex) {
        throw std::runtime_error("Invalid operands for frameaddr instruction");
    }

    int fiIndex = frameIndex->getIndex();
    auto offsetIt = layout.frameIndexToOffset.find(fiIndex);
    if (offsetIt == layout.frameIndexToOffset.end()) {
        throw std::runtime_error("Frame index " + std::to_string(fiIndex) +
                                 " not found in final layout");
    }

    int offset = offsetIt->second;

    std::cout << "Eliminating frameaddr " << destReg->toString() << ", FI("
              << fiIndex << ") -> addi " << destReg->toString() << ", s0, "
              << offset << std::endl;

    // 创建最终的addi指令: addi destReg, s0, offset
    auto newInst = std::make_unique<Instruction>(Opcode::ADDI);
    newInst->addOperand(std::make_unique<RegisterOperand>(
        destReg->getRegNum(), destReg->isVirtual()));
    newInst->addOperand(std::make_unique<RegisterOperand>(8, false));  // s0
    newInst->addOperand(std::make_unique<ImmediateOperand>(offset));

    // 替换指令
    it = bb->insert(it, std::move(newInst));
    ++it;
    it = bb->erase(it);  // 删除原frameaddr指令
}

int FrameIndexElimination::alignTo(int value, int alignment) const {
    return (value + alignment - 1) & ~(alignment - 1);
}

void FrameIndexElimination::printFinalLayout() const {
    std::cout << "\n=== Final Frame Layout ===" << std::endl;
    std::cout << "Function: " << function->getName() << std::endl;
    std::cout << "Total frame size: " << layout.totalFrameSize << " bytes"
              << std::endl;
    std::cout << "Return address at: sp+" << layout.returnAddressOffset
              << std::endl;
    std::cout << "Frame pointer at: sp+" << layout.framePointerOffset
              << std::endl;
    std::cout << "Final Frame Index mappings:" << std::endl;

    for (const auto& [fi, offset] : layout.frameIndexToOffset) {
        std::cout << "  FI(" << fi << ") -> s0" << offset << std::endl;
    }
}

}  // namespace riscv64