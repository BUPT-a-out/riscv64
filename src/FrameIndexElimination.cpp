#include "FrameIndexElimination.h"

#include <algorithm>
#include <iostream>
#include <set>

// TODO: preserve emergency  stack slot
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

    assignFinalOffsets();
}

// 目前的布局:
// 高地址端

// 保留寄存器 ra s0 (s1-s11)
// alloca对象
// spill寄存器

// 4 * 8 字节应急区
// 低地址端
// TODO： 应急也许放在顶上，考虑和栈上参数的冲突。
// TODO： 也许用不到应急，

void FrameIndexElimination::assignFinalOffsets() {
    // 计算保存寄存器需要的空间
    int savedRegSize = calculateSavedRegisterSize();

    // 计算局部变量区大小
    int localVarSize = 0;
    int spillSize = 0;

    int spillCount = 0;

    for (const auto& obj : stackManager->getAllStackObjects()) {
        if (obj->type == StackObjectType::AllocatedStackSlot) {
            localVarSize += alignTo(obj->size, obj->alignment);
        } else if (obj->type == StackObjectType::SpilledRegister) {
            spillSize += alignTo(obj->size, obj->alignment);
            spillCount ++;
        }
    }

    // 应对溢出时寄存器高压情况
    int emergencySpace = 4 * 8; 

    // 计算调用参数在s0正偏移, 不计算在内

    // 计算总栈帧大小：基础保存寄存器 + 局部变量 + 溢出寄存器 +
    int safetySpace = 16;  // 额外的安全空间，确保所有栈对象都在栈帧范围内
    int totalSize = savedRegSize + localVarSize + spillSize + safetySpace + emergencySpace;
    layout.totalFrameSize = alignTo(totalSize, 16);  // 16字节对齐

    // 计算各区域偏移

    // 我们会把fp即s0指向栈帧的高地址端
    // 我们最后将fp作为基址

    // 这两个offset是相对sp的, 但是实际上在代码里完全没用到，仅仅用来打印
    layout.returnAddressOffset = layout.totalFrameSize - 8;  // ra
    layout.framePointerOffset = layout.totalFrameSize - 16;  // s0

    // 其实这两个变量也完全没用到

    // 为每个Frame Index分配具体偏移 (相对于s0)
    // s0 指向栈帧顶部，所有栈对象都应该在 s0 以下的位置

    // 保存寄存器区域在栈帧顶部附近，预留空间给保存的寄存器
    int currentLocalOffset = -savedRegSize - 8;  // 额外预留8字节安全空间
    int currentSpillOffset =
        currentLocalOffset - localVarSize - 8;  // 额外预留8字节安全空间

    for (const auto& obj : stackManager->getAllStackObjects()) {
        // 按这种初始化, 先做减法.
        if (obj->type == StackObjectType::AllocatedStackSlot) {
            currentLocalOffset -= alignTo(obj->size, obj->alignment);
            // 确保偏移量是8的倍数（RISCV64要求）- 向下对齐负偏移量
            currentLocalOffset = (currentLocalOffset / 8) * 8;
            layout.frameIndexToOffset[obj->identifier] = currentLocalOffset;

            std::cout << "FI(" << obj->identifier << ") [alloca] -> s0"
                      << currentLocalOffset << " (size: " << obj->size
                      << ", alignment: " << obj->alignment << ")" << std::endl;
        } else if (obj->type == StackObjectType::SpilledRegister) {
            currentSpillOffset -= alignTo(obj->size, obj->alignment);
            // 确保偏移量是8的倍数（RISCV64要求）- 向下对齐负偏移量
            currentSpillOffset = (currentSpillOffset / 8) * 8;
            layout.frameIndexToOffset[obj->identifier] = currentSpillOffset;

            std::cout << "FI(" << obj->identifier << ") [spill reg "
                      << obj->regNum << "] -> s0" << currentSpillOffset
                      << " (size: " << obj->size
                      << ", alignment: " << obj->alignment << ")" << std::endl;
        }
    }

    std::cout << "Total " << spillCount << " registers are spilled" << std::endl;
}

int FrameIndexElimination::calculateSavedRegisterSize() {
    // 分析函数中使用的callee-saved寄存器
    auto usedSavedIntegerRegs = collectSavedIntegerRegisters();
    auto usedSavedFloatRegs = collectSavedFloatRegisters();

    return usedSavedIntegerRegs.size() * 8 +
           usedSavedFloatRegs.size() * 4;  // 每个整数寄存器8字节，单精浮点4字节
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
    std::cout << "Generating final prologue/epilogue for frame size: "
              << layout.totalFrameSize << std::endl;

    // 收集需要保存的寄存器
    std::vector<int> savedIntRegs = collectSavedIntegerRegisters();
    auto savedFloatRegs = collectSavedFloatRegisters();

    // 生成序言 (插入到函数开头)
    BasicBlock* entryBlock = function->getEntryBlock();
    if (entryBlock) {
        std::vector<std::unique_ptr<Instruction>> prologueInsts;
        auto appendToProlog =
            [&prologueInsts](
                std::vector<std::unique_ptr<Instruction>>&& insts) {
                std::move(insts.begin(), insts.end(),
                          std::back_inserter(prologueInsts));
            };

        int offset = layout.totalFrameSize;

        appendToProlog(generateStackAdjustment(offset));

        // 保存整数寄存器
        for (int regNum : savedIntRegs) {
            offset -= 8;
            appendToProlog(generateRegisterSave(regNum, offset, Opcode::SD));
        }

        // 保存浮点寄存器
        for (int regNum : savedFloatRegs) {
            offset -= 4;
            appendToProlog(generateRegisterSave(regNum, offset, Opcode::FSW));
        }

        appendToProlog(generateFramePointerSetup(layout.totalFrameSize));

        // 逆序插入以保持正确顺序
        for (auto it = prologueInsts.rbegin(); it != prologueInsts.rend();
             ++it) {
            entryBlock->insert(entryBlock->begin(), std::move(*it));
        }
    }

    // 生成尾声 (插入到所有ret指令前)
    for (auto& bb : *function) {
        std::vector<std::unique_ptr<Instruction>> epilogueInsts;
        auto appendToEpilog =
            [&epilogueInsts](
                std::vector<std::unique_ptr<Instruction>>&& insts) {
                std::move(insts.begin(), insts.end(),
                          std::back_inserter(epilogueInsts));
            };
        // 从后往前遍历，更高效地找到RET指令
        for (auto rit = bb->rbegin(); rit != bb->rend(); ++rit) {
            if ((*rit)->getOpcode() == Opcode::RET) {
                // 转换为正向迭代器进行插入操作
                auto it = std::prev(rit.base());

                // 恢复所有保存的寄存器
                int offset = layout.totalFrameSize;
                for (int regNum : savedIntRegs) {
                    offset -= 8;
                    appendToEpilog(restoreRegister(regNum, offset, Opcode::LD));
                }

                for (int regNum : savedFloatRegs) {
                    offset -= 4;
                    appendToEpilog(
                        restoreRegister(regNum, offset, Opcode::FLW));
                }

                appendToEpilog(generateAddImm(2, 2, layout.totalFrameSize));

                for (auto& inst : epilogueInsts) {
                    it = bb->insert(it, std::move(inst));
                    ++it;
                }

                break;  // 找到第一个（从后往前看是最后一个）RET就退出
            }
        }
    }
}

// 应该保存的整数寄存器.
std::vector<int> FrameIndexElimination::collectSavedIntegerRegisters() {
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
                        if (ABI::isCalleeSaved(regNum, false)) {
                            usedSavedRegs.insert(regNum);
                        }
                    }
                }
            }
        }
    }

    return std::vector<int>(usedSavedRegs.begin(), usedSavedRegs.end());
}

// 应该保存的浮点寄存器
std::vector<int> FrameIndexElimination::collectSavedFloatRegisters() {
    std::set<int> usedSavedRegs;

    // 扫描所有指令，查找使用的fs寄存器
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            for (const auto& operand : inst->getOperands()) {
                if (auto* regOp =
                        dynamic_cast<RegisterOperand*>(operand.get())) {
                    int regNum = regOp->getRegNum();
                    // fs0-fs11
                    if (regOp->isFloatRegister()) {
                        if (ABI::isCalleeSaved(regNum, true)) {
                            usedSavedRegs.insert(regNum);
                        }
                    }
                }
            }
        }
    }

    return std::vector<int>(usedSavedRegs.begin(), usedSavedRegs.end());
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

// 检查偏移量是否在有效的立即数范围内（-2048 到 +2047）
bool FrameIndexElimination::isValidImmediateOffset(int64_t offset) const {
    return offset >= -2048 && offset <= 2047;
}

// 当偏移量超出范围时，生成地址计算指令
std::vector<std::unique_ptr<Instruction>>
FrameIndexElimination::generateAddressComputation(int offset) {
    std::vector<std::unique_ptr<Instruction>> insts;

    auto liOffsetInst = std::make_unique<Instruction>(Opcode::LI);
    liOffsetInst->addOperand(
        std::make_unique<RegisterOperand>(5, false));  // t0
    liOffsetInst->addOperand(std::make_unique<ImmediateOperand>(offset));
    insts.push_back(std::move(liOffsetInst));

    auto addAddrInst = std::make_unique<Instruction>(Opcode::ADD);
    addAddrInst->addOperand(std::make_unique<RegisterOperand>(5, false));  // t0
    addAddrInst->addOperand(std::make_unique<RegisterOperand>(2, false));  // sp
    addAddrInst->addOperand(std::make_unique<RegisterOperand>(5, false));  // t0
    insts.push_back(std::move(addAddrInst));

    return insts;
}

// 生成内存操作数
std::unique_ptr<MemoryOperand> FrameIndexElimination::generateMemoryOperand(
    int offset, bool useDirectOffset) {
    if (useDirectOffset) {
        return std::make_unique<MemoryOperand>(
            std::make_unique<RegisterOperand>(2, false),  // sp
            std::make_unique<ImmediateOperand>(offset));
    } else {
        return std::make_unique<MemoryOperand>(
            std::make_unique<RegisterOperand>(5, false),  // t0
            std::make_unique<ImmediateOperand>(0));
    }
}

// 生成单个寄存器保存指令
std::vector<std::unique_ptr<Instruction>>
FrameIndexElimination::generateRegisterSave(int regNum, int offset,
                                            Opcode opcode) {
    std::vector<std::unique_ptr<Instruction>> insts;

    auto saveReg = std::make_unique<Instruction>(opcode);
    saveReg->addOperand(std::make_unique<RegisterOperand>(regNum, false));

    if (isValidImmediateOffset(offset)) {
        // 直接使用偏移量
        saveReg->addOperand(generateMemoryOperand(offset, true));
        insts.push_back(std::move(saveReg));
    } else {
        // 使用地址计算
        auto addrInsts = generateAddressComputation(offset);
        // 先添加地址计算指令
        std::move(addrInsts.begin(), addrInsts.end(),
                  std::back_inserter(insts));

        // 然后添加保存指令，使用t0寄存器，偏移为0
        saveReg->addOperand(
            generateMemoryOperand(0, false));  // 使用t0寄存器，偏移为0
        insts.push_back(std::move(saveReg));
    }

    return insts;
}

// 恢复寄存器
std::vector<std::unique_ptr<Instruction>>
FrameIndexElimination::restoreRegister(int regNum, int offset, Opcode opcode) {
    std::vector<std::unique_ptr<Instruction>> insts;

    auto restoreReg = std::make_unique<Instruction>(opcode);
    restoreReg->addOperand(std::make_unique<RegisterOperand>(regNum, false));

    bool useDirectOffset = isValidImmediateOffset(offset);
    if (!useDirectOffset) {
        // 先生成地址计算指令
        auto addrInsts = generateAddressComputation(offset);
        std::move(addrInsts.begin(), addrInsts.end(),
                  std::back_inserter(insts));
    }

    restoreReg->addOperand(generateMemoryOperand(offset, useDirectOffset));
    insts.push_back(std::move(restoreReg));

    return insts;
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
              << fiIndex << ") -> ";

    if (isValidImmediateOffset(offset)) {
        std::cout << "addi " << destReg->toString() << ", s0, " << offset
                  << std::endl;
    } else {
        std::cout << "li t0, " << offset << "; add " << destReg->toString()
                  << ", s0, t0" << std::endl;
    }

    auto destRegNum = destReg->getRegNum();
    auto addrInsts = generateAddImm(destRegNum, 8, offset, false);

    for (auto& addrinst : addrInsts) {
        it = bb->insert(it, std::move(addrinst));
        ++it;
    }

    // 删除原frameaddr指令
    it = bb->erase(it);
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