#include "FrameIndexElimination.h"

#include <iostream>
#include <algorithm>
#include <set>

#include "Instructions/All.h"

namespace riscv64 {

void FrameIndexElimination::run() {
    std::cout << "\n=== Running Frame Index Elimination (Phase 3) ===" << std::endl;
    
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
            std::cout << "Found FI(" << fi << ") type=" 
                      << (int)obj->type << ", size=" << obj->size << std::endl;
        }
    }

    assignFinalOffsets();
    
    std::cout << "Final layout - Local vars: " << totalLocalVarSize 
              << " bytes, Spills: " << totalSpillSize << " bytes" << std::endl;
}

void FrameIndexElimination::assignFinalOffsets() {
    // RISC-V栈帧最终布局 (从高地址到低地址):
    // sp+frameSize:  调用者栈帧
    // sp+X:          ra (return address)  
    // sp+Y:          s0 (frame pointer)
    // sp+0 (s0-Z):   局部变量区
    // s0-A:          溢出寄存器区
    // sp:            当前栈指针位置

    // 计算保存寄存器需要的空间
    int savedRegSize = 16;  // ra(8) + s0(8)
    
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
    
    // 计算总栈帧大小
    int totalSize = savedRegSize + localVarSize + spillSize;
    layout.totalFrameSize = alignTo(totalSize, 16);  // 16字节对齐
    
    // 计算各区域偏移
    layout.returnAddressOffset = layout.totalFrameSize - 8;    // ra
    layout.framePointerOffset = layout.totalFrameSize - 16;   // s0
    layout.localVariableAreaOffset = 0;   // 从s0开始向下
    layout.spillAreaOffset = -localVarSize; // 在局部变量区下方
    
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
                      << obj->regNum << "] -> s0" << currentSpillOffset << std::endl;
        }
    }
}

void FrameIndexElimination::generateFinalPrologueEpilogue() {
    if (layout.totalFrameSize == 0) {
        return;
    }

    std::cout << "Generating final prologue/epilogue for frame size: " 
              << layout.totalFrameSize << std::endl;

    // 生成序言 (插入到函数开头)
    BasicBlock* entryBlock = function->getBasicBlock(0);
    if (entryBlock) {
        std::vector<std::unique_ptr<Instruction>> prologueInsts;
        
        // 1. 调整栈指针: addi sp, sp, -frameSize
        auto adjustSp = std::make_unique<Instruction>(Opcode::ADDI);
        adjustSp->addOperand(std::make_unique<RegisterOperand>(2, false));  // sp
        adjustSp->addOperand(std::make_unique<RegisterOperand>(2, false));  // sp
        adjustSp->addOperand(std::make_unique<ImmediateOperand>(-layout.totalFrameSize));
        prologueInsts.push_back(std::move(adjustSp));
        
        // 2. 保存ra: sd ra, offset(sp)
        auto saveRa = std::make_unique<Instruction>(Opcode::SD);
        saveRa->addOperand(std::make_unique<RegisterOperand>(1, false));   // ra
        saveRa->addOperand(std::make_unique<MemoryOperand>(
            std::make_unique<RegisterOperand>(2, false),  // sp
            std::make_unique<ImmediateOperand>(layout.returnAddressOffset)));
        prologueInsts.push_back(std::move(saveRa));
        
        // 3. 保存并设置帧指针: sd s0, offset(sp); addi s0, sp, frameSize
        auto saveFp = std::make_unique<Instruction>(Opcode::SD);
        saveFp->addOperand(std::make_unique<RegisterOperand>(8, false));   // s0
        saveFp->addOperand(std::make_unique<MemoryOperand>(
            std::make_unique<RegisterOperand>(2, false),  // sp
            std::make_unique<ImmediateOperand>(layout.framePointerOffset)));
        prologueInsts.push_back(std::move(saveFp));
        
        auto setFp = std::make_unique<Instruction>(Opcode::ADDI);
        setFp->addOperand(std::make_unique<RegisterOperand>(8, false));    // s0
        setFp->addOperand(std::make_unique<RegisterOperand>(2, false));    // sp
        setFp->addOperand(std::make_unique<ImmediateOperand>(layout.totalFrameSize));
        prologueInsts.push_back(std::move(setFp));
        
        // 逆序插入以保持正确顺序
        for (auto it = prologueInsts.rbegin(); it != prologueInsts.rend(); ++it) {
            entryBlock->insert(entryBlock->begin(), std::move(*it));
        }
    }

    // 生成尾声 (插入到所有ret指令前)
    for (auto& bb : *function) {
        for (auto it = bb->begin(); it != bb->end(); ++it) {
            if ((*it)->getOpcode() == Opcode::RET) {
                // 1. 恢复s0: ld s0, offset(sp)
                auto restoreFp = std::make_unique<Instruction>(Opcode::LD);
                restoreFp->addOperand(std::make_unique<RegisterOperand>(8, false));
                restoreFp->addOperand(std::make_unique<MemoryOperand>(
                    std::make_unique<RegisterOperand>(2, false),
                    std::make_unique<ImmediateOperand>(layout.framePointerOffset)));
                it = bb->insert(it, std::move(restoreFp));
                ++it;
                
                // 2. 恢复ra: ld ra, offset(sp)
                auto restoreRa = std::make_unique<Instruction>(Opcode::LD);
                restoreRa->addOperand(std::make_unique<RegisterOperand>(1, false));
                restoreRa->addOperand(std::make_unique<MemoryOperand>(
                    std::make_unique<RegisterOperand>(2, false),
                    std::make_unique<ImmediateOperand>(layout.returnAddressOffset)));
                it = bb->insert(it, std::move(restoreRa));
                ++it;
                
                // 3. 恢复栈指针: addi sp, sp, frameSize
                auto restoreSp = std::make_unique<Instruction>(Opcode::ADDI);
                restoreSp->addOperand(std::make_unique<RegisterOperand>(2, false));
                restoreSp->addOperand(std::make_unique<RegisterOperand>(2, false));
                restoreSp->addOperand(std::make_unique<ImmediateOperand>(layout.totalFrameSize));
                it = bb->insert(it, std::move(restoreSp));
                ++it;
                
                break;  // 每个基本块最多一个ret
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
    std::cout << "Total frame size: " << layout.totalFrameSize << " bytes" << std::endl;
    std::cout << "Return address at: sp+" << layout.returnAddressOffset << std::endl;
    std::cout << "Frame pointer at: sp+" << layout.framePointerOffset << std::endl;
    std::cout << "Final Frame Index mappings:" << std::endl;
    
    for (const auto& [fi, offset] : layout.frameIndexToOffset) {
        std::cout << "  FI(" << fi << ") -> s0" << offset << std::endl;
    }
}

}