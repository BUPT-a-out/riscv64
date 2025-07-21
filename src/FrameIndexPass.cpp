#include "FrameIndexPass.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <set>

#include "ABI.h"
#include "Instructions/All.h"

namespace riscv64 {

void FrameIndexPass::run() {
    std::cout << "\n=== Running Frame Index Pass ===" << std::endl;

    // 第1步：计算栈帧布局
    computeFrameLayout();

    // 第2步：生成序言代码
    generatePrologue();

    // 第3步：展开frameaddr伪指令
    expandFrameAddressInstructions();

    // 第4步：生成尾声代码
    generateEpilogue();

    // 调试输出
    printFrameLayout();
}

void FrameIndexPass::computeFrameLayout() {
    std::cout << "Computing frame layout for function: " << function->getName()
              << std::endl;

    // 移除未使用的frameInfo变量引用

    // 收集所有唯一的frame index
    std::set<int> uniqueFrameIndices;
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            if (inst->getOpcode() == Opcode::FRAMEADDR) {
                const auto& operands = inst->getOperands();
                if (operands.size() >= 2) {
                    if (auto* fi = dynamic_cast<FrameIndexOperand*>(
                            operands[1].get())) {
                        uniqueFrameIndices.insert(fi->getIndex());
                    }
                }
            }
        }
    }

    // 为每个唯一的FI获取其对应的栈对象信息
    std::vector<std::pair<int, StackObject*>> uniqueStackObjects;
    int totalLocalVarSize = 0;

    for (int fi : uniqueFrameIndices) {
        auto* obj = stackManager->getStackObjectByIdentifier(fi);
        if (obj) {
            uniqueStackObjects.push_back({fi, obj});
            totalLocalVarSize += alignTo(obj->size, obj->alignment);
            std::cout << "Object FI(" << fi << ") size: " << obj->size
                      << " bytes" << std::endl;
        }
    }

    std::cout << "Found " << uniqueStackObjects.size()
              << " unique stack objects" << std::endl;
    std::cout << "Total local variable size: " << totalLocalVarSize << " bytes"
              << std::endl;

    calculateFrameOffsets();
}

void FrameIndexPass::calculateFrameOffsets() {
    // 重新收集唯一的frame indices及其大小信息
    std::map<int, int> fiToSize;  // {fi_index, size}
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            if (inst->getOpcode() == Opcode::FRAMEADDR) {
                const auto& operands = inst->getOperands();
                if (operands.size() >= 2) {
                    if (auto* fi = dynamic_cast<FrameIndexOperand*>(
                            operands[1].get())) {
                        int fiIndex = fi->getIndex();
                        // 只记录每个FI一次
                        if (fiToSize.find(fiIndex) == fiToSize.end()) {
                            auto* obj =
                                stackManager->getStackObjectByIdentifier(
                                    fiIndex);
                            if (obj) {
                                fiToSize[fiIndex] = obj->size;
                            }
                        }
                    }
                }
            }
        }
    }

    // 计算局部变量总大小
    int localVarSize = 0;
    for (const auto& [fi, size] : fiToSize) {
        localVarSize += alignTo(size, 4);  // 4字节对齐
    }

    // 计算总栈帧大小
    int savedRegSize = 16;  // ra(8) + fp(8)
    int totalSize = savedRegSize + localVarSize;

    // 对齐到16字节边界
    layout.totalFrameSize = alignTo(totalSize, 16);

    // 计算各区域偏移量
    layout.returnAddressOffset = layout.totalFrameSize - 8;  // ra在最顶部
    layout.framePointerOffset = layout.totalFrameSize - 16;  // fp在ra下面

    // 正确的偏移量分配算法：从fp开始向下累积分配
    int currentOffset = 0;  // 从fp-0开始

    // 按FI索引排序，确保一致的布局
    for (const auto& [fiIndex, size] : fiToSize) {
        // 为当前对象分配空间，先增加偏移量
        currentOffset += alignTo(size, 4);
        // 当前对象的偏移量是负的累积偏移量
        layout.frameIndexToOffset[fiIndex] = -currentOffset;

        std::cout << "FI(" << fiIndex << ") -> offset "
                  << layout.frameIndexToOffset[fiIndex] << " (size: " << size
                  << ")" << std::endl;
    }

    std::cout << "Total frame size: " << layout.totalFrameSize << " bytes"
              << std::endl;
    std::cout << "RA offset: " << layout.returnAddressOffset << "(sp)"
              << std::endl;
    std::cout << "FP offset: " << layout.framePointerOffset << "(sp)"
              << std::endl;
}

void FrameIndexPass::generatePrologue() {
    if (layout.totalFrameSize == 0) {
        return;  // 不需要栈帧
    }

    BasicBlock* entryBlock = function->getBasicBlock(0);
    if (!entryBlock) {
        throw std::runtime_error("Function has no entry block");
    }

    std::cout << "Generating prologue for frame size: " << layout.totalFrameSize
              << std::endl;

    // 创建所有指令，然后按正确顺序插入
    std::vector<std::unique_ptr<Instruction>> prologueInsts;

    // 1. 设置新的帧指针: addi s0, sp, frameSize
    auto setFp = std::make_unique<Instruction>(Opcode::ADDI);
    setFp->addOperand(std::make_unique<RegisterOperand>(8, false));  // s0/fp
    setFp->addOperand(std::make_unique<RegisterOperand>(2, false));  // sp
    setFp->addOperand(
        std::make_unique<ImmediateOperand>(layout.totalFrameSize));
    prologueInsts.push_back(std::move(setFp));

    // 2. 保存帧指针: sd s0, offset(sp)
    auto saveFp = std::make_unique<Instruction>(Opcode::SD);
    saveFp->addOperand(std::make_unique<RegisterOperand>(8, false));  // s0/fp
    saveFp->addOperand(std::make_unique<MemoryOperand>(
        std::make_unique<RegisterOperand>(2, false),  // sp
        std::make_unique<ImmediateOperand>(layout.framePointerOffset)));
    prologueInsts.push_back(std::move(saveFp));

    // 3. 保存返回地址: sd ra, offset(sp)
    auto saveRa = std::make_unique<Instruction>(Opcode::SD);
    saveRa->addOperand(std::make_unique<RegisterOperand>(1, false));  // ra
    saveRa->addOperand(std::make_unique<MemoryOperand>(
        std::make_unique<RegisterOperand>(2, false),  // sp
        std::make_unique<ImmediateOperand>(layout.returnAddressOffset)));
    prologueInsts.push_back(std::move(saveRa));

    // 4. 调整栈指针: addi sp, sp, -frameSize （这个必须最先执行）
    auto stackAdjust = std::make_unique<Instruction>(Opcode::ADDI);
    stackAdjust->addOperand(std::make_unique<RegisterOperand>(2, false));  // sp
    stackAdjust->addOperand(std::make_unique<RegisterOperand>(2, false));  // sp
    stackAdjust->addOperand(
        std::make_unique<ImmediateOperand>(-layout.totalFrameSize));
    prologueInsts.push_back(std::move(stackAdjust));

    // 按正确顺序插入指令（从后往前插入，所以最后插入的是第一个执行的）
    for (auto& inst : prologueInsts) {
        insertInstructionAtBeginning(entryBlock, std::move(inst));
    }
}

void FrameIndexPass::generateEpilogue() {
    if (layout.totalFrameSize == 0) {
        return;  // 不需要栈帧清理
    }

    // 为所有包含ret指令的基本块生成尾声
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            if (inst->getOpcode() == Opcode::RET) {
                std::cout << "Generating epilogue for basic block: "
                          << bb->getLabel() << std::endl;

                // 1. 恢复帧指针: ld s0, offset(sp)
                auto restoreFp = std::make_unique<Instruction>(Opcode::LD);
                restoreFp->addOperand(
                    std::make_unique<RegisterOperand>(8, false));  // s0/fp
                restoreFp->addOperand(std::make_unique<MemoryOperand>(
                    std::make_unique<RegisterOperand>(2, false),  // sp
                    std::make_unique<ImmediateOperand>(
                        layout.framePointerOffset)));
                insertInstructionBeforeRet(bb.get(), std::move(restoreFp));

                // 2. 恢复返回地址: ld ra, offset(sp)
                auto restoreRa = std::make_unique<Instruction>(Opcode::LD);
                restoreRa->addOperand(
                    std::make_unique<RegisterOperand>(1, false));  // ra
                restoreRa->addOperand(std::make_unique<MemoryOperand>(
                    std::make_unique<RegisterOperand>(2, false),  // sp
                    std::make_unique<ImmediateOperand>(
                        layout.returnAddressOffset)));
                insertInstructionBeforeRet(bb.get(), std::move(restoreRa));

                // 3. 恢复栈指针: addi sp, sp, frameSize
                auto restoreSp = std::make_unique<Instruction>(Opcode::ADDI);
                restoreSp->addOperand(
                    std::make_unique<RegisterOperand>(2, false));  // sp
                restoreSp->addOperand(
                    std::make_unique<RegisterOperand>(2, false));  // sp
                restoreSp->addOperand(
                    std::make_unique<ImmediateOperand>(layout.totalFrameSize));
                insertInstructionBeforeRet(bb.get(), std::move(restoreSp));

                break;  // 每个基本块最多一个ret指令
            }
        }
    }
}

void FrameIndexPass::expandFrameAddressInstructions() {
    std::cout << "Expanding frameaddr instructions..." << std::endl;

    for (auto& bb : *function) {
        for (auto it = bb->begin(); it != bb->end();) {
            if ((*it)->getOpcode() == Opcode::FRAMEADDR) {
                expandFrameAddressInstruction(bb.get(), it);
                // it已经被更新，不需要++
            } else {
                ++it;
            }
        }
    }
}

void FrameIndexPass::expandFrameAddressInstruction(
    BasicBlock* bb, std::list<std::unique_ptr<Instruction>>::iterator& it) {
    auto& inst = *it;
    const auto& operands = inst->getOperands();

    if (operands.size() < 2) {
        throw std::runtime_error("frameaddr instruction must have 2 operands");
    }

    // 获取目标寄存器和FI
    auto* destReg = dynamic_cast<RegisterOperand*>(operands[0].get());
    auto* frameIndex = dynamic_cast<FrameIndexOperand*>(operands[1].get());

    if (!destReg || !frameIndex) {
        throw std::runtime_error("Invalid operands for frameaddr instruction");
    }

    int fiIndex = frameIndex->getIndex();
    auto offsetIt = layout.frameIndexToOffset.find(fiIndex);
    if (offsetIt == layout.frameIndexToOffset.end()) {
        throw std::runtime_error("Frame index " + std::to_string(fiIndex) +
                                 " not found in layout");
    }

    int offset = offsetIt->second;

    std::cout << "Expanding frameaddr " << destReg->toString() << ", FI("
              << fiIndex << ") -> addi " << destReg->toString() << ", s0, "
              << offset << std::endl;

    // 创建替换指令: addi destReg, s0, offset
    auto newInst = std::make_unique<Instruction>(Opcode::ADDI);
    newInst->addOperand(std::make_unique<RegisterOperand>(
        destReg->getRegNum(), destReg->isVirtual()));
    newInst->addOperand(std::make_unique<RegisterOperand>(8, false));  // s0/fp
    newInst->addOperand(std::make_unique<ImmediateOperand>(offset));

    // 替换指令
    it = bb->insert(it, std::move(newInst));
    ++it;                // 移动到新插入的指令之后
    it = bb->erase(it);  // 删除原来的frameaddr指令，it自动移动到下一个指令
}

void FrameIndexPass::insertInstructionAtBeginning(
    BasicBlock* bb, std::unique_ptr<Instruction> inst) {
    bb->insert(bb->begin(), std::move(inst));
}

void FrameIndexPass::insertInstructionBeforeRet(
    BasicBlock* bb, std::unique_ptr<Instruction> inst) {
    // 找到ret指令的位置
    for (auto it = bb->begin(); it != bb->end(); ++it) {
        if ((*it)->getOpcode() == Opcode::RET) {
            bb->insert(it, std::move(inst));
            return;
        }
    }

    // 如果没有找到ret指令，插入到末尾
    bb->insert(bb->end(), std::move(inst));
}

int FrameIndexPass::alignTo(int value, int alignment) const {
    return (value + alignment - 1) & ~(alignment - 1);
}

void FrameIndexPass::printFrameLayout() const {
    std::cout << "\n=== Frame Layout Summary ===" << std::endl;
    std::cout << "Function: " << function->getName() << std::endl;
    std::cout << "Total frame size: " << layout.totalFrameSize << " bytes"
              << std::endl;
    std::cout << "Return address at: sp+" << layout.returnAddressOffset
              << std::endl;
    std::cout << "Frame pointer at: sp+" << layout.framePointerOffset
              << std::endl;
    std::cout << "Frame index mappings:" << std::endl;

    if (layout.frameIndexToOffset.empty()) {
        std::cout << "  (no frame indices)" << std::endl;
    } else {
        std::cout << "=========================" << std::endl;
    }
}

}  // namespace riscv64
