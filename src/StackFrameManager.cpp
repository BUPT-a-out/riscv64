#include "StackFrameManager.h"

#include "Instructions/MachineOperand.h"

namespace riscv64 {
// 静态成员定义
const std::vector<unsigned> StackFrameManager::callerSavedRegs = {
    1,                              // ra (return address)
    5,  6,  7,  28, 29, 30, 31,     // t0-t2, t3-t6 (temporaries)
    10, 11, 12, 13, 14, 15, 16, 17  // a0-a7 (arguments/return values)
};

const std::vector<unsigned> StackFrameManager::calleeSavedRegs_static = {
    8,  9,                                  // s0-s1 (saved registers)
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27  // s2-s11 (saved registers)
};

const std::vector<unsigned> StackFrameManager::argumentRegs = {
    10, 11, 12, 13, 14, 15, 16, 17  // a0-a7
};

const std::vector<unsigned> StackFrameManager::returnValueRegs = {
    10, 11  // a0, a1
};

// 实现文件 StackFrameManager.cpp
void StackFrameManager::computeStackFrame() {
    // 1. 分析寄存器使用情况
    analyzeRegisterUsage();

    // 2. 计算栈布局
    calculateStackLayout();

    // 3. 分配栈偏移量
    assignStackOffsets();

    // 4. 插入序言和尾声
    insertPrologueEpilogue();
}

void StackFrameManager::analyzeRegisterUsage() {
    std::unordered_set<unsigned> usedRegs;
    std::unordered_set<unsigned> definedRegs;

    // 遍历所有指令，收集使用的寄存器
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            // 分析指令使用的寄存器
            const auto& operands = inst->getOperands();
            for (const auto& operand : operands) {
                if (operand->isReg()) {
                    unsigned regNum = operand->getRegNum();
                    usedRegs.insert(regNum);

                    // 如果是定义操作数（通常是第一个）
                    if (&operand == &operands[0]) {
                        definedRegs.insert(regNum);
                    }
                }
            }

            // 特殊处理函数调用
            if (inst->isCallInstr()) {
                // 函数调用会破坏所有caller-saved寄存器
                for (unsigned reg : callerSavedRegs) {
                    if (usedRegs.count(reg)) {
                        // 如果使用了caller-saved寄存器，可能需要保存
                    }
                }
            }
        }
    }

    // 确定需要保存的callee-saved寄存器
    for (unsigned reg : calleeSavedRegs_static) {
        if (definedRegs.count(reg)) {
            markRegisterForSaving(reg);
        }
    }

    // 总是保存返回地址寄存器（如果有函数调用）
    bool hasCall = false;
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            if (inst->isCallInstr()) {
                hasCall = true;
                break;
            }
        }
        if (hasCall) break;
    }

    if (hasCall) {
        markRegisterForSaving(1);  // ra
    }
}

void StackFrameManager::calculateStackLayout() {
    // RISC-V栈布局（从高地址到低地址）：
    // +-------------------+  <- 调用者的栈帧
    // | 参数区 (>8个参数)   |
    // +-------------------+  <- sp (函数入口)
    // | 返回地址 (ra)       |
    // +-------------------+
    // | 保存的寄存器区       |
    // +-------------------+
    // | 局部变量区          |
    // +-------------------+
    // | 溢出寄存器区        |
    // +-------------------+
    // | 被调用函数参数区     |
    // +-------------------+  <- sp (函数内部)

    int currentOffset = 0;

    // 1. 被调用函数参数区（如果有函数调用）
    if (frameInfo.maxCalleeArgs > 8) {
        frameInfo.parameterSize = (frameInfo.maxCalleeArgs - 8) * 8;
        frameInfo.parameterOffset = currentOffset;
        currentOffset += frameInfo.parameterSize;
    }

    // 2. 溢出寄存器区
    frameInfo.spilledRegisterOffset = currentOffset;
    currentOffset += frameInfo.spilledRegisterSize;

    // 3. 局部变量区
    frameInfo.localVariableOffset = currentOffset;
    currentOffset += frameInfo.localVariableSize;

    // 4. 保存的寄存器区
    frameInfo.savedRegisterSize = calleeSavedRegs.size() * 8;
    frameInfo.savedRegisterOffset = currentOffset;
    currentOffset += frameInfo.savedRegisterSize;

    // 5. 返回地址
    if (calleeSavedRegs.count(1)) {  // ra
        frameInfo.returnAddressOffset = currentOffset;
        currentOffset += 8;
    }

    // 对齐到16字节边界（RISC-V ABI要求）
    frameInfo.totalSize = alignTo(currentOffset, 16);
    frameInfo.hasStackPointerAdjustment = frameInfo.totalSize > 0;
}

void StackFrameManager::assignStackOffsets() {
    // 为每个栈对象分配具体的偏移量
    for (auto& obj : stackObjects) {
        switch (obj->type) {
            case StackObjectType::SpilledRegister:
                obj->offset = frameInfo.spilledRegisterOffset +
                              spilledRegToStackSlot[obj->regNum] * 8;
                break;
            case StackObjectType::LocalVariable:
                obj->offset = frameInfo.localVariableOffset +
                              localVarToStackSlot[obj->regNum] * obj->size;
                break;
            case StackObjectType::SavedRegister:
                // 保存寄存器的偏移量由寄存器编号决定
                obj->offset = frameInfo.savedRegisterOffset + obj->regNum * 8;
                break;
            case StackObjectType::ReturnAddress:
                obj->offset = frameInfo.returnAddressOffset;
                break;
            case StackObjectType::AllocatedStackSlot:
                // 为 alloca 产生的栈对象分配偏移量
                obj->offset =
                    frameInfo.localVariableOffset + obj->identifier * obj->size;
                break;
            default:
                break;
        }
    }
}

void StackFrameManager::insertPrologueEpilogue() {
    if (!hasStackFrame()) {
        return;
    }

    // 为函数的第一个基本块插入序言
    BasicBlock* entryBlock = function->getBasicBlock(0);
    generatePrologue();

    // 为所有返回基本块插入尾声
    for (auto& bb : *function) {
        // 检查基本块是否以返回指令结束
        if (!bb->getInstructionCount()) continue;

        auto lastInst = bb->getInstruction(bb->getInstructionCount() - 1);
        if (lastInst->getOpcode() == RET || lastInst->getOpcode() == JR) {
            generateEpilogue();
        }
    }
}

void StackFrameManager::generatePrologue() {
    BasicBlock* entryBlock = function->getBasicBlock(0);

    // 1. 调整栈指针
    if (frameInfo.hasStackPointerAdjustment) {
        auto stackAdjust = createStackAdjustInstruction(-frameInfo.totalSize);
        insertInstructionAtBeginning(entryBlock, std::move(stackAdjust));
    }

    // 2. 保存返回地址
    if (calleeSavedRegs.count(1)) {
        auto saveRa = createSaveInstruction(1, frameInfo.returnAddressOffset);
        insertInstructionAtBeginning(entryBlock, std::move(saveRa));
    }

    // 3. 保存callee-saved寄存器
    for (unsigned reg : calleeSavedRegs) {
        if (reg != 1) {  // ra已经处理过了
            auto saveInst = createSaveInstruction(
                reg, frameInfo.savedRegisterOffset + reg * 8);
            insertInstructionAtBeginning(entryBlock, std::move(saveInst));
        }
    }

    // 4. 设置帧指针（如果使用）
    if (calleeSavedRegs.count(8)) {  // s0/fp
        // mv s0, sp
        auto setFp = std::make_unique<Instruction>(MV);
        setFp->addOperand(std::make_unique<RegisterOperand>(8, false));  // s0
        setFp->addOperand(std::make_unique<RegisterOperand>(2, false));  // sp
        insertInstructionAtBeginning(entryBlock, std::move(setFp));
    }
}

void StackFrameManager::generateEpilogue() {
    // 这里需要在所有返回指令前插入尾声代码
    // 为简化，假设只有一个返回点

    // 1. 恢复callee-saved寄存器
    for (unsigned reg : calleeSavedRegs) {
        if (reg != 1) {
            auto restoreInst = createRestoreInstruction(
                reg, frameInfo.savedRegisterOffset + reg * 8);
            // 需要在返回指令前插入
        }
    }

    // 2. 恢复返回地址
    if (calleeSavedRegs.count(1)) {
        auto restoreRa =
            createRestoreInstruction(1, frameInfo.returnAddressOffset);
        // 需要在返回指令前插入
    }

    // 3. 恢复栈指针
    if (frameInfo.hasStackPointerAdjustment) {
        auto stackRestore = createStackAdjustInstruction(frameInfo.totalSize);
        // 需要在返回指令前插入
    }
}

// 栈对象管理函数
int StackFrameManager::allocateStackSlot(StackObjectType type, int size,
                                         int alignment, unsigned regNum) {
    auto obj = std::make_unique<StackObject>(type, size, alignment, regNum);
    int index = stackObjects.size();
    stackObjects.push_back(std::move(obj));

    // 更新相应的大小信息
    switch (type) {
        case StackObjectType::SpilledRegister:
            spilledRegToStackSlot[regNum] = frameInfo.spilledRegisterSize / 8;
            frameInfo.spilledRegisterSize += alignTo(size, alignment);
            break;
        case StackObjectType::LocalVariable:
            localVarToStackSlot[regNum] = frameInfo.localVariableSize / size;
            frameInfo.localVariableSize += alignTo(size, alignment);
            break;
        case StackObjectType::SavedRegister:
            frameInfo.savedRegisterSize += alignTo(size, alignment);
            break;
        case StackObjectType::AllocatedStackSlot:
            // 对于 alloca 产生的栈对象，不需要特殊的映射处理
            frameInfo.localVariableSize += alignTo(size, alignment);
            break;
        default:
            break;
    }

    return index;
}

int StackFrameManager::allocateSpillSlot(unsigned regNum) {
    if (spilledRegToStackSlot.find(regNum) != spilledRegToStackSlot.end()) {
        return spilledRegToStackSlot[regNum];
    }

    return allocateStackSlot(StackObjectType::SpilledRegister, 8, 8, regNum);
}

int StackFrameManager::getSpillSlotOffset(unsigned regNum) const {
    auto it = spilledRegToStackSlot.find(regNum);
    if (it != spilledRegToStackSlot.end()) {
        return frameInfo.spilledRegisterOffset + it->second * 8;
    }
    return -1;
}

int StackFrameManager::allocateLocalVariable(int size, int alignment) {
    static unsigned localVarCounter = 1000;
    return allocateStackSlot(StackObjectType::LocalVariable, size, alignment,
                             localVarCounter++);
}

void StackFrameManager::markRegisterForSaving(unsigned regNum) {
    if (isCalleeSaved(regNum)) {
        calleeSavedRegs.insert(regNum);
    }
}

// 辅助函数实现
bool StackFrameManager::isCallerSaved(unsigned regNum) const {
    return std::find(callerSavedRegs.begin(), callerSavedRegs.end(), regNum) !=
           callerSavedRegs.end();
}

bool StackFrameManager::isCalleeSaved(unsigned regNum) const {
    return std::find(calleeSavedRegs_static.begin(),
                     calleeSavedRegs_static.end(),
                     regNum) != calleeSavedRegs_static.end();
}

bool StackFrameManager::isArgumentRegister(unsigned regNum) const {
    return std::find(argumentRegs.begin(), argumentRegs.end(), regNum) !=
           argumentRegs.end();
}

bool StackFrameManager::isReturnRegister(unsigned regNum) const {
    return std::find(returnValueRegs.begin(), returnValueRegs.end(), regNum) !=
           returnValueRegs.end();
}

int StackFrameManager::alignTo(int value, int alignment) const {
    return (value + alignment - 1) & ~(alignment - 1);
}

// 指令生成辅助函数
std::unique_ptr<Instruction> StackFrameManager::createStackAdjustInstruction(
    int offset) {
    auto inst = std::make_unique<Instruction>(ADDI);
    inst->addOperand(std::make_unique<RegisterOperand>(2, false));  // sp
    inst->addOperand(std::make_unique<RegisterOperand>(2, false));  // sp
    inst->addOperand(std::make_unique<ImmediateOperand>(offset));
    return inst;
}

std::unique_ptr<Instruction> StackFrameManager::createSaveInstruction(
    unsigned regNum, int offset) {
    auto inst = std::make_unique<Instruction>(SD);
    inst->addOperand(std::make_unique<RegisterOperand>(regNum, false));
    inst->addOperand(std::make_unique<MemoryOperand>(
        std::make_unique<RegisterOperand>(2, false),  // sp
        std::make_unique<ImmediateOperand>(offset)));
    return inst;
}

std::unique_ptr<Instruction> StackFrameManager::createRestoreInstruction(
    unsigned regNum, int offset) {
    auto inst = std::make_unique<Instruction>(LD);
    inst->addOperand(std::make_unique<RegisterOperand>(regNum, false));
    inst->addOperand(std::make_unique<MemoryOperand>(
        std::make_unique<RegisterOperand>(2, false),  // sp
        std::make_unique<ImmediateOperand>(offset)));
    return inst;
}

void StackFrameManager::insertInstructionAtBeginning(
    BasicBlock* bb, std::unique_ptr<Instruction> inst) {
    bb->insert(bb->begin(), std::move(inst));
}

void StackFrameManager::insertInstructionAtEnd(
    BasicBlock* bb, std::unique_ptr<Instruction> inst) {
    bb->insert(bb->end(), std::move(inst));
}

// 调试输出
void StackFrameManager::printStackLayout() const {
    std::cout << "Stack Frame Layout for function " << function->getName()
              << ":\n";
    std::cout << "Total size: " << frameInfo.totalSize << " bytes\n";
    std::cout << "Layout (from high to low address):\n";

    if (frameInfo.parameterSize > 0) {
        std::cout << "  Parameters: " << frameInfo.parameterSize
                  << " bytes at offset " << frameInfo.parameterOffset << "\n";
    }

    if (frameInfo.spilledRegisterSize > 0) {
        std::cout << "  Spilled registers: " << frameInfo.spilledRegisterSize
                  << " bytes at offset " << frameInfo.spilledRegisterOffset
                  << "\n";
    }

    std::cout << "Spilled registers: " << std::endl;
    for (auto reg : spilledRegToStackSlot) {
        std::cout << "%vreg_" << reg.first << " -> " << reg.second << std::endl;
    }

    if (frameInfo.localVariableSize > 0) {
        std::cout << "  Local variables: " << frameInfo.localVariableSize
                  << " bytes at offset " << frameInfo.localVariableOffset
                  << "\n";
    }

    if (frameInfo.savedRegisterSize > 0) {
        std::cout << "  Saved registers: " << frameInfo.savedRegisterSize
                  << " bytes at offset " << frameInfo.savedRegisterOffset
                  << "\n";
    }

    std::cout << "Saved registers: ";
    for (unsigned reg : calleeSavedRegs) {
        std::cout << ABI::getABINameFromRegNum(reg) << " ";
    }
    std::cout << "\n";
}

}  // namespace riscv64