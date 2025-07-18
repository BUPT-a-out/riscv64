#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"
#include "Instructions/BasicBlock.h"

namespace riscv64 {


bool Instruction::isCopyInstr() const {
    switch (opcode) {
        case MV:
            // MV rd, rs - 将 rs 的值复制到 rd
            return operands.size() == 2;
            
        case COPY:
            // COPY rd, rs - 寄存器分配约束指令
            return operands.size() == 2;
            
        case FMOV_S:
            // FMOV.S frd, frs - 单精度浮点数移动
            return operands.size() == 2;
            
        case FMOV_D:
            // FMOV.D frd, frs - 双精度浮点数移动
            return operands.size() == 2;
            
        case ADDI:
            // ADDI rd, rs, 0 等价于 MV rd, rs
            if (operands.size() == 3) {
                // 检查第三个操作数是否为立即数 0
                return operands[2]->isImm() && operands[2]->getValue() == 0;
            }
            return false;
            
        case OR:
            // OR rd, rs, x0 等价于 MV rd, rs (假设 x0 是零寄存器)
            if (operands.size() == 3) {
                // 检查第三个操作数是否为零寄存器
                return operands[2]->isReg() && operands[2]->getRegNum() == 0;
            }
            return false;
            
        case ORI:
            // ORI rd, rs, 0 等价于 MV rd, rs
            if (operands.size() == 3) {
                // 检查第三个操作数是否为立即数 0
                return operands[2]->isImm() && operands[2]->getValue() == 0;
            }
            return false;
            
        default:
            return false;
    }
}

bool Instruction::isJumpInstr() const {
    return opcode == JAL || opcode == JALR || opcode == J || opcode == JR || opcode == RET;
}

bool Instruction::isCallInstr() const {
    // 直接的函数调用指令
    if (opcode == CALL) {
        return true;
    }
    
    // JAL指令：如果目标寄存器是ra(x1)，则是函数调用
    if (opcode == JAL) {
        if (!operands.empty()) {
            auto* dest_operand = operands[0].get();
            if (dest_operand->getType() == OperandType::Register) {
                RegisterOperand* reg_op = static_cast<RegisterOperand*>(dest_operand);
                // 检查是否是ra寄存器(x1)
                return reg_op->getRegNum() == 1;  // ra寄存器编号为1
            }
        }
        return true;  // 如果无法确定，保守地认为是调用
    }
    
    // JALR指令：如果目标寄存器是ra(x1)，则是函数调用
    if (opcode == JALR) {
        if (!operands.empty()) {
            auto* dest_operand = operands[0].get();
            if (dest_operand->getType() == OperandType::Register) {
                RegisterOperand* reg_op = static_cast<RegisterOperand*>(dest_operand);
                // 检查是否是ra寄存器(x1)
                return reg_op->getRegNum() == 1;  // ra寄存器编号为1
            }
        }
        return false;  // JALR如果不是写入ra，通常是间接跳转而非调用
    }
    
    // TAIL伪指令也是函数调用的一种形式（尾调用）
    if (opcode == TAIL) {
        return true;
    }
    
    return false;
}

bool Instruction::isBranch() const {
    // 条件分支指令
    if (opcode == BEQ || opcode == BNE || opcode == BLT || opcode == BGE || 
        opcode == BLTU || opcode == BGEU || opcode == BEQZ || opcode == BNEZ || 
        opcode == BLEZ || opcode == BGEZ || opcode == BLTZ || opcode == BGTZ || 
        opcode == BGT || opcode == BLE || opcode == BGTU || opcode == BLEU) {
        return true;
    }
    
    // 无条件跳转指令
    if (isJumpInstr()) {
        return true;
    }
    
    // 函数调用指令
    if (isCallInstr()) {
        return true;
    }
    
    return false;
}

bool Instruction::isBackEdge() const {
    if (!isBranch()) {
        return false;
    }
    
    BasicBlock* target_bb = nullptr;
    
    // 获取目标基本块
    if (opcode == BEQ || opcode == BNE || opcode == BLT || opcode == BGE || 
        opcode == BLTU || opcode == BGEU || opcode == BEQZ || opcode == BNEZ || 
        opcode == BLEZ || opcode == BGEZ || opcode == BLTZ || opcode == BGTZ || 
        opcode == BGT || opcode == BLE || opcode == BGTU || opcode == BLEU) {
        if (operands.size() >= 3) {
            auto* target_operand = operands.back().get();
            if (target_operand->getType() == OperandType::Label) {
                LabelOperand* label_op = static_cast<LabelOperand*>(target_operand);
                // 需要根据LabelOperand的实际设计调整这里的类型转换
                target_bb = reinterpret_cast<BasicBlock*>(label_op->getBlock());
            }
        }
    } else if (opcode == JAL || opcode == J) {
        size_t target_idx = (opcode == JAL) ? 1 : 0;
        if (operands.size() > target_idx) {
            auto* target_operand = operands[target_idx].get();
            if (target_operand->getType() == OperandType::Label) {
                LabelOperand* label_op = static_cast<LabelOperand*>(target_operand);
                target_bb = reinterpret_cast<BasicBlock*>(label_op->getBlock());
            }
        }
    }
    
    // 简单的回边检测：检查目标是否在当前基本块的前驱中
    if (target_bb && parent) {
        // 这种方法假设如果跳转目标是当前块的前驱，则可能是回边
        // 更准确的判断需要完整的CFG分析
        return target_bb == parent;  // 自循环一定是回边
    }
    
    return false;
}


// 辅助函数：检查指令是否涉及栈指针
bool Instruction::involvesStackPointer() const {
    // 需要在Instruction类中实现
    const auto& operands = getOperands();
    for (const auto& operand : operands) {
        if (operand->isReg()) {
            RegisterOperand* regOp = static_cast<RegisterOperand*>(operand.get());
            if (regOp->getRegNum() == 2) { // sp = x2
                return true;
            }
        } else if (operand->isMem()) {
            MemoryOperand* memOp = static_cast<MemoryOperand*>(operand.get());
            if (memOp->getBaseReg() && memOp->getBaseReg()->isReg()) {
                RegisterOperand* baseReg = static_cast<RegisterOperand*>(memOp->getBaseReg());
                if (baseReg->getRegNum() == 2) { // 基于栈指针的内存访问
                    return true;
                }
            }
        }
    }
    return false;
}

// 辅助函数：检查指令是否是参数移动指令
bool Instruction::isParameterMove() const {
    // 检查是否是将参数寄存器值移动到虚拟寄存器的指令
    if (!isCopyInstr()) return false;
    
    const auto& operands = getOperands();
    if (operands.size() >= 2 && operands[0]->isReg() && operands[1]->isReg()) {
        unsigned srcReg = operands[1]->getRegNum();
        return srcReg >= 10 && srcReg <= 17; // a0-a7
    }
    return false;
}

// 辅助函数：检查指令是否是帧设置指令  
bool Instruction::isFrameSetup() const {
    // 检查是否是设置栈帧的指令，如 addi sp, sp, -framesize
    if (getOpcode() == ADDI) {
        const auto& operands = getOperands();
        if (operands.size() >= 3 && 
            operands[0]->isReg() && operands[1]->isReg() && operands[2]->isImm()) {
            unsigned dstReg = operands[0]->getRegNum();
            unsigned srcReg = operands[1]->getRegNum();
            if (dstReg == 2 && srcReg == 2) { // sp = sp + offset
                return true;
            }
        }
    }
    return false;
}

// 辅助函数：检查指令是否与帧指针相关
bool Instruction::isFramePointerRelated() const {
    const auto& operands = getOperands();
    for (const auto& operand : operands) {
        if (operand->isReg()) {
            RegisterOperand* regOp = static_cast<RegisterOperand*>(operand.get());
            if (regOp->getRegNum() == 8) { // s0/fp = x8
                return true;
            }
        }
    }
    return false;
}

bool Instruction::isReturnInstr() const {
    // 直接的RET指令
    if (opcode == RET) {
        return true;
    }
    
    // JALR指令：jalr x0, ra, 0 形式（这是RET的展开形式）
    if (opcode == JALR) {
        if (operands.size() >= 2) {
            auto* dest_operand = operands[0].get();
            auto* src_operand = operands[1].get();
            
            // 检查目标寄存器是否是x0，源寄存器是否是ra(x1)
            if (dest_operand->isReg() && src_operand->isReg()) {
                RegisterOperand* dest_reg = static_cast<RegisterOperand*>(dest_operand);
                RegisterOperand* src_reg = static_cast<RegisterOperand*>(src_operand);
                
                // jalr x0, ra, offset 形式（通常offset为0）
                if (dest_reg->getRegNum() == 0 && src_reg->getRegNum() == 1) {
                    return true;
                }
            }
        }
    }
    
    // JR指令：jr ra 形式
    if (opcode == JR) {
        if (!operands.empty()) {
            auto* operand = operands[0].get();
            if (operand->isReg()) {
                RegisterOperand* reg_op = static_cast<RegisterOperand*>(operand);
                // 检查是否是ra寄存器(x1)
                if (reg_op->getRegNum() == 1) {
                    return true;
                }
            }
        }
    }
    
    return false;
}


}  // namespace riscv64