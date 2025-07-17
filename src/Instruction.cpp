#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"
#include "Instructions/BasicBlock.h"

namespace riscv64 {
std::optional<DestSourcePair> Instruction::isCopyInstrImpl() const {
    // 首先检查是否是显式的寄存器移动指令
    if (opcode == Opcode::MV) {
        if (operands.size() >= 2 && operands[0]->isReg() &&
            operands[1]->isReg()) {
            return DestSourcePair{operands[0].get(), operands[1].get()};
        }
    }

    // 检查各种可能隐含复制操作的指令模式
    switch (opcode) {
        default:
            break;
        case Opcode::ADD:
        case Opcode::OR:
        case Opcode::XOR:
            // ADD/OR/XOR rd, rs, x0 或 ADD/OR/XOR rd, x0, rs 形式
            if (operands.size() >= 3) {
                if (operands[1]->isReg() &&
                    operands[1]->getRegNum() == 0 &&
                    operands[2]->isReg()) {
                    return DestSourcePair{operands[0].get(), operands[2].get()};
                }
                if (operands[2]->isReg() &&
                    operands[2]->getRegNum() == 0 &&
                    operands[1]->isReg()) {
                    return DestSourcePair{operands[0].get(), operands[1].get()};
                }
            }
            break;
        case Opcode::ADDI:
            // ADDI rd, rs, 0 形式
            if (operands.size() >= 3 && operands[1]->isReg() &&
                operands[2]->isImm() && operands[2]->getValue() == 0) {
                return DestSourcePair{operands[0].get(), operands[1].get()};
            }
            break;
        case Opcode::SUB:
            // SUB rd, rs, x0 形式
            if (operands.size() >= 3 && operands[2]->isReg() &&
                operands[2]->getRegNum() == 0 && operands[1]->isReg()) {
                return DestSourcePair{operands[0].get(), operands[1].get()};
            }
            break;
        case Opcode::FSGNJ_S:
        case Opcode::FSGNJ_D:
            // FSGNJ.[S/D] rd, rs, rs 形式 (浮点寄存器移动)
            if (operands.size() >= 3 && operands[1]->isReg() &&
                operands[2]->isReg() &&
                operands[1]->getRegNum() == operands[2]->getRegNum()) {
                return DestSourcePair{operands[0].get(), operands[1].get()};
            }
            break;
    }

    return std::nullopt;
}

std::optional<DestSourcePair> Instruction::isCopyInstr() const {
    // 首先检查显式的COPY指令
    if (opcode == Opcode::COPY) {
        if (operands.size() >= 2 && operands[0]->isReg() &&
            operands[1]->isReg()) {
            return DestSourcePair{operands[0].get(), operands[1].get()};
        }
        return std::nullopt;
    }

    // 检查其他可能的复制指令形式
    return isCopyInstrImpl();
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


}  // namespace riscv64