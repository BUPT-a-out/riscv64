#include "Instructions/Instruction.h"

#include <algorithm>

#include "Instructions/BasicBlock.h"
#include "Instructions/MachineOperand.h"

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
    return opcode == JAL || opcode == JALR || opcode == J || opcode == JR ||
           opcode == RET;
}

// note: only int32/64 store/load
bool Instruction::isMemoryInstr() const {
    return opcode == Opcode::SW || opcode == Opcode::LW || opcode == Opcode::SD || opcode == Opcode::LD;
}

bool Instruction::isTerminator() const {
    return opcode == Opcode::J || opcode == Opcode::JAL ||
           opcode == Opcode::JALR || opcode == Opcode::BNEZ ||
           opcode == Opcode::BEQZ || opcode == Opcode::BEQ ||
           opcode == Opcode::BNE || opcode == Opcode::BLT ||
           opcode == Opcode::BGE || opcode == Opcode::BLTU ||
           opcode == Opcode::BGEU || opcode == Opcode::RET || isJumpInstr();
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
                RegisterOperand* reg_op =
                    static_cast<RegisterOperand*>(dest_operand);
                // 检查是否是ra寄存器(x1)
                return reg_op->isIntegerRegister() &&
                       reg_op->getRegNum() == 1;  // ra寄存器编号为1
            }
        }
        return true;  // 如果无法确定，保守地认为是调用
    }

    // JALR指令：如果目标寄存器是ra(x1)，则是函数调用
    if (opcode == JALR) {
        if (!operands.empty()) {
            auto* dest_operand = operands[0].get();
            if (dest_operand->getType() == OperandType::Register) {
                RegisterOperand* reg_op =
                    static_cast<RegisterOperand*>(dest_operand);
                // 检查是否是ra寄存器(x1)
                return reg_op->isIntegerRegister() &&
                       reg_op->getRegNum() == 1;  // ra寄存器编号为1
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
                LabelOperand* label_op =
                    static_cast<LabelOperand*>(target_operand);
                target_bb = reinterpret_cast<BasicBlock*>(label_op->getBlock());
            }
        }
    } else if (opcode == JAL || opcode == J) {
        size_t target_idx = (opcode == JAL) ? 1 : 0;
        if (operands.size() > target_idx) {
            auto* target_operand = operands[target_idx].get();
            if (target_operand->getType() == OperandType::Label) {
                LabelOperand* label_op =
                    static_cast<LabelOperand*>(target_operand);
                target_bb = reinterpret_cast<BasicBlock*>(label_op->getBlock());
            }
        }
    }

    // 简单的回边检测：检查目标是否在当前基本块的前驱中
    if (target_bb && parent) {
        return target_bb == parent;  // 自循环一定是回边
    }

    return false;
}

bool Instruction::involvesStackPointer() const {
    const auto& operands = getOperands();
    for (const auto& operand : operands) {
        if (operand->isReg()) {
            RegisterOperand* regOp =
                static_cast<RegisterOperand*>(operand.get());
            if (regOp->isIntegerRegister() &&
                regOp->getRegNum() == 2) {  // sp = x2
                return true;
            }
        } else if (operand->isMem()) {
            MemoryOperand* memOp = static_cast<MemoryOperand*>(operand.get());
            if (memOp->getBaseReg() && memOp->getBaseReg()->isReg()) {
                RegisterOperand* baseReg =
                    static_cast<RegisterOperand*>(memOp->getBaseReg());
                if (baseReg->isIntegerRegister() &&
                    baseReg->getRegNum() == 2) {  // 基于栈指针的内存访问
                    return true;
                }
            }
        }
    }
    return false;
}

bool Instruction::isParameterMove() const {
    if (!isCopyInstr()) return false;

    const auto& operands = getOperands();
    if (operands.size() >= 2 && operands[0]->isReg() && operands[1]->isReg()) {
        unsigned srcReg = operands[1]->getRegNum();
        return srcReg >= 10 && srcReg <= 17;  // a0-a7 f0-f7
    }
    return false;
}

bool Instruction::isFrameSetup() const {
    if (getOpcode() == ADDI) {
        const auto& operands = getOperands();
        if (operands.size() >= 3 && operands[0]->isReg() &&
            operands[1]->isReg() && operands[0]->isIntegerRegister() &&
            operands[1]->isIntegerRegister() && operands[2]->isImm()) {
            unsigned dstReg = operands[0]->getRegNum();
            unsigned srcReg = operands[1]->getRegNum();
            if (dstReg == 2 && srcReg == 2) {  // sp = sp + offset
                return true;
            }
        }
    }
    return false;
}

bool Instruction::isFramePointerRelated() const {
    const auto& operands = getOperands();
    for (const auto& operand : operands) {
        if (operand->isReg()) {
            RegisterOperand* regOp =
                static_cast<RegisterOperand*>(operand.get());
            if (regOp->isIntegerRegister() &&
                regOp->getRegNum() == 8) {  // s0/fp = x8
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

            if (dest_operand->isReg() && src_operand->isReg()) {
                RegisterOperand* dest_reg =
                    static_cast<RegisterOperand*>(dest_operand);
                RegisterOperand* src_reg =
                    static_cast<RegisterOperand*>(src_operand);

                if (dest_reg->isIntegerRegister() &&
                    dest_reg->getRegNum() == 0 &&
                    src_reg->isIntegerRegister() && src_reg->getRegNum() == 1) {
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
                RegisterOperand* reg_op =
                    static_cast<RegisterOperand*>(operand);
                if (reg_op->isIntegerRegister() && reg_op->getRegNum() == 1) {
                    return true;
                }
            }
        }
    }

    return false;
}

// 获取使用的整数寄存器
std::vector<unsigned> Instruction::getUsedIntegerRegs() const {
    std::vector<unsigned> usedRegs;
    if (opcode == RET) {
        usedRegs.push_back(10);
    }
    if (operands.empty()) return usedRegs;

    switch (opcode) {
        // 算术指令：ADD rd, rs1, rs2 - 使用rs1和rs2
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case AND:
        case OR:
        case XOR:
        case SLL:
        case SRL:
        case SRA:
        case SLT:
        case SLTU:
        case REM:
        case REMU:
        case DIVW:
        case DIVUW:
        case REMW:
        case REMUW:
        case ADDW:
        case SUBW:
        case MULW:
        case SLLW:
        case SRLW:
        case SRAW: {
            if (operands.size() >= 3) {
                if (operands[1]->isReg() && operands[1]->isIntegerRegister()) {
                    usedRegs.push_back(operands[1]->getRegNum());
                }
                if (operands[2]->isReg() && operands[1]->isIntegerRegister()) {
                    usedRegs.push_back(operands[2]->getRegNum());
                }
            }
            break;
        }

        // 立即数算术指令：ADDI rd, rs1, imm - 使用rs1
        case ADDI:
        case SLTI:
        case SLTIU:
        case XORI:
        case ORI:
        case ANDI:
        case SLLI:
        case SRLI:
        case SRAI:
        case ADDIW:
        case SLLIW:
        case SRLIW:
        case SRAIW: {
            if (operands.size() >= 2 && operands[1]->isReg() &&
                operands[1]->isIntegerRegister()) {
                usedRegs.push_back(operands[1]->getRegNum());
            }
            break;
        }

        case LI:
            break;

        // 加载指令：LD rd, offset(rs1) - 使用rs1作为基址寄存器
        case LD:
        case LW:
        case LH:
        case LB:
        case LWU:
        case LHU:
        case LBU: {
            if (operands.size() >= 2 && operands[1]->isMem()) {
                MemoryOperand* memOp =
                    static_cast<MemoryOperand*>(operands[1].get());
                if (memOp->getBaseReg() && memOp->getBaseReg()->isReg() &&
                    memOp->getBaseReg()->isIntegerRegister()) {
                    usedRegs.push_back(memOp->getBaseReg()->getRegNum());
                }
                if (memOp->getOffset() && memOp->getOffset()->isReg() &&
                    memOp->getOffset()->isIntegerRegister()) {
                    usedRegs.push_back(memOp->getOffset()->getRegNum());
                }
            }
            break;
        }

        // 存储指令：SD rs2, offset(rs1) - 使用rs1作为基址寄存器，rs2作为数据源
        case SD:
        case SW:
        case SH:
        case SB: {
            if (operands.size() >= 2) {
                if (operands[0]->isReg() && operands[0]->isIntegerRegister()) {
                    usedRegs.push_back(operands[0]->getRegNum());
                }
                if (operands[1]->isMem()) {
                    MemoryOperand* memOp =
                        static_cast<MemoryOperand*>(operands[1].get());
                    if (memOp->getBaseReg() && memOp->getBaseReg()->isReg() &&
                        memOp->getBaseReg()->isIntegerRegister()) {
                        usedRegs.push_back(memOp->getBaseReg()->getRegNum());
                    }
                    if (memOp->getOffset() && memOp->getOffset()->isReg() &&
                        memOp->getOffset()->isIntegerRegister()) {
                        usedRegs.push_back(memOp->getOffset()->getRegNum());
                    }
                }
            }
            break;
        }

            // 浮点-整数移动指令
        case FMV_X_W:
        case FMV_X_D: {
            // 这些指令使用浮点寄存器作为源，在浮点寄存器方法中处理
            break;
        }

        case FMV_W_X:
        case FMV_D_X: {
            // 这些指令使用整数寄存器作为源
            if (operands.size() >= 2 && operands[1]->isReg() &&
                operands[1]->isIntegerRegister()) {
                usedRegs.push_back(operands[1]->getRegNum());
            }
            break;
        }

        // 分支指令：BEQ rs1, rs2, label - 使用rs1和rs2进行比较
        case BEQ:
        case BNE:
        case BLT:
        case BGE:
        case BLTU:
        case BGEU: {
            if (operands.size() >= 2) {
                if (operands[0]->isReg() && operands[0]->isIntegerRegister()) {
                    usedRegs.push_back(operands[0]->getRegNum());
                }
                if (operands[1]->isReg() && operands[1]->isIntegerRegister()) {
                    usedRegs.push_back(operands[1]->getRegNum());
                }
            }
            break;
        }

        case BNEZ: {
            if (operands.size() >= 2) {
                if (operands[0]->isReg() && operands[0]->isIntegerRegister()) {
                    usedRegs.push_back(operands[0]->getRegNum());
                }
            }
            break;
        }

        case JALR: {
            if (operands.size() >= 2 && operands[1]->isReg() &&
                operands[1]->isIntegerRegister()) {
                usedRegs.push_back(operands[1]->getRegNum());
            }
            break;
        }

        case MV: {
            if (operands.size() >= 2 && operands[1]->isReg() &&
                operands[1]->isIntegerRegister()) {
                usedRegs.push_back(operands[1]->getRegNum());
            }
            break;
        }

        case CALL: {
            for (int i = 10; i <= 17; i++) {
                usedRegs.push_back(i);
            }
            break;
        }

        default: {
            if (isCallInstr()) {
                // 函数调用的特殊处理, incorrect
                for (size_t i = 0; i < operands.size(); ++i) {
                    if (operands[i]->isReg() &&
                        operands[i]->isIntegerRegister()) {
                        unsigned regNum = operands[i]->getRegNum();
                        if (std::find(usedRegs.begin(), usedRegs.end(),
                                      regNum) == usedRegs.end()) {
                            usedRegs.push_back(regNum);
                        }
                    }
                }
            } else {
                // 通用策略：除第一个操作数外都是源操作数
                for (size_t i = 1; i < operands.size(); ++i) {
                    if (operands[i]->isReg() &&
                        operands[i]->isIntegerRegister()) {
                        usedRegs.push_back(operands[i]->getRegNum());
                    } else if (operands[i]->isMem()) {
                        MemoryOperand* memOp =
                            static_cast<MemoryOperand*>(operands[i].get());
                        if (memOp->getBaseReg() &&
                            memOp->getBaseReg()->isReg() &&
                            memOp->getBaseReg()->isIntegerRegister()) {
                            usedRegs.push_back(
                                memOp->getBaseReg()->getRegNum());
                        }
                        if (memOp->getOffset() && memOp->getOffset()->isReg() &&
                            memOp->getOffset()->isIntegerRegister()) {
                            usedRegs.push_back(memOp->getOffset()->getRegNum());
                        }
                    }
                }
            }
            break;
        }
    }

    // 去除重复的寄存器
    std::sort(usedRegs.begin(), usedRegs.end());
    usedRegs.erase(std::unique(usedRegs.begin(), usedRegs.end()),
                   usedRegs.end());

    return usedRegs;
}

// 获取使用的浮点寄存器
std::vector<unsigned> Instruction::getUsedFloatRegs() const {
    std::vector<unsigned> usedRegs;
    if (opcode == RET) {
        usedRegs.push_back(42);
    }
    if (operands.empty()) return usedRegs;

    switch (opcode) {
        // 浮点算术指令
        case FADD_S:
        case FSUB_S:
        case FMUL_S:
        case FDIV_S:
        case FADD_D:
        case FSUB_D:
        case FMUL_D:
        case FDIV_D:
        case FMIN_S:
        case FMIN_D:
        case FMAX_S:
        case FMAX_D:
        case FSGNJ_S:
        case FSGNJ_D:
        case FSGNJN_S:
        case FSGNJN_D:
        case FSGNJX_S:
        case FSGNJX_D: {
            if (operands.size() >= 3) {
                if (operands[1]->isReg() && operands[1]->isFloatRegister()) {
                    usedRegs.push_back(operands[1]->getRegNum());
                }
                if (operands[2]->isReg() && operands[2]->isFloatRegister()) {
                    usedRegs.push_back(operands[2]->getRegNum());
                }
            }
            break;
        }

            // 单操作数浮点指令
        case FSQRT_S:
        case FSQRT_D:
        case FCLASS_S:
        case FCLASS_D: {
            if (operands.size() >= 2 && operands[1]->isReg() &&
                operands[1]->isFloatRegister()) {
                usedRegs.push_back(operands[1]->getRegNum());
            }
            break;
        }

        // 浮点加载指令 - 不使用浮点寄存器作为源操作数
        case FLD:
        case FLW: {
            break;
        }

        // 浮点存储指令：FSD frs2, offset(rs1) - 使用frs2作为数据源
        case FSD:
        case FSW: {
            if (operands.size() >= 1 && operands[0]->isReg() &&
                operands[0]->isFloatRegister()) {
                usedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

        // 浮点移动指令
        case FMOV_S:
        case FMOV_D: {
            if (operands.size() >= 2 && operands[1]->isReg() &&
                operands[1]->isFloatRegister()) {
                usedRegs.push_back(operands[1]->getRegNum());
            }
            break;
        }

        // 浮点比较指令
        case FEQ_S:
        case FLT_S:
        case FLE_S:
        case FEQ_D:
        case FLT_D:
        case FLE_D: {
            if (operands.size() >= 3) {
                if (operands[1]->isReg() && operands[1]->isFloatRegister()) {
                    usedRegs.push_back(operands[1]->getRegNum());
                }
                if (operands[2]->isReg() && operands[2]->isFloatRegister()) {
                    usedRegs.push_back(operands[2]->getRegNum());
                }
            }
            break;
        }

        // 浮点转换指令 - 使用浮点寄存器作为源
        case FCVT_W_S:
        case FCVT_L_S:
        case FCVT_WU_S:
        case FCVT_LU_S:
        case FCVT_W_D:
        case FCVT_L_D:
        case FCVT_WU_D:
        case FCVT_LU_D:
        case FCVT_S_D:
        case FCVT_D_S: {
            if (operands.size() >= 2 && operands[1]->isReg() &&
                operands[1]->isFloatRegister()) {
                usedRegs.push_back(operands[1]->getRegNum());
            }
            break;
        }

            // 浮点-整数移动指令
        case FMV_X_W:
        case FMV_X_D: {
            if (operands.size() >= 2 && operands[1]->isReg() &&
                operands[1]->isFloatRegister()) {
                usedRegs.push_back(operands[1]->getRegNum());
            }
            break;
        }

        case FMV_W_X:
        case FMV_D_X: {
            // 这些指令使用整数寄存器作为源，不使用浮点寄存器
            break;
        }

        case CALL: {
            for (int i = 42; i <= 49; i++) {
                usedRegs.push_back(i);
            }
        }

        default: {
            // 通用策略：检查所有操作数中的浮点寄存器（排除第一个操作数，通常是目标）
            for (size_t i = 1; i < operands.size(); ++i) {
                if (operands[i]->isReg() && operands[i]->isFloatRegister()) {
                    usedRegs.push_back(operands[i]->getRegNum());
                }
            }
            break;
        }
    }

    // 去除重复的寄存器
    std::sort(usedRegs.begin(), usedRegs.end());
    usedRegs.erase(std::unique(usedRegs.begin(), usedRegs.end()),
                   usedRegs.end());

    return usedRegs;
}

// 获取定义的整数寄存器
std::vector<unsigned> Instruction::getDefinedIntegerRegs() const {
    std::vector<unsigned> definedRegs;
    if (operands.empty()) return definedRegs;

    switch (opcode) {
        // 大多数整数指令：第一个操作数是目标寄存器
        case ADD:
        case SUB:
        case MUL:
        case DIV:
        case AND:
        case OR:
        case XOR:
        case SLL:
        case SRL:
        case SRA:
        case SLT:
        case SLTU:
        case ADDI:
        case SLTI:
        case SLTIU:
        case XORI:
        case ORI:
        case ANDI:
        case SLLI:
        case SRLI:
        case SRAI:
        case LD:
        case LW:
        case LH:
        case LB:
        case LWU:
        case LHU:
        case LBU:
        case LUI:
        case AUIPC:
        case MV:
        case LI:
        case LA:
        case FRAMEADDR:
        case REM:
        case REMU:
        case DIVW:
        case DIVUW:
        case REMW:
        case REMUW:
        case ADDW:
        case SUBW:
        case MULW:
        case SLLW:
        case SRLW:
        case SRAW:
        case ADDIW:
        case SLLIW:
        case SRLIW:
        case SRAIW: {
            if (operands[0]->isReg() && operands[0]->isIntegerRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

            // 浮点转换到整数指令定义整数寄存器
        case FCVT_W_S:
        case FCVT_L_S:
        case FCVT_WU_S:
        case FCVT_LU_S:
        case FCVT_W_D:
        case FCVT_L_D:
        case FCVT_WU_D:
        case FCVT_LU_D: {
            if (operands[0]->isReg() && operands[0]->isIntegerRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

        // 浮点分类指令定义整数寄存器
        case FCLASS_S:
        case FCLASS_D: {
            if (operands[0]->isReg() && operands[0]->isIntegerRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

        // 浮点到整数移动指令定义整数寄存器
        case FMV_X_W:
        case FMV_X_D: {
            if (operands[0]->isReg() && operands[0]->isIntegerRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

        // CALL指令定义所有caller-saved整数寄存器
        case CALL: {
            // ra (x1)
            // definedRegs.push_back(1);

            // t0-t2 (x5-x7)
            for (unsigned i = 5; i <= 7; ++i) {
                definedRegs.push_back(i);
            }

            // a0-a7 (x10-x17)
            for (unsigned i = 10; i <= 17; ++i) {
                definedRegs.push_back(i);
            }

            // t3-t6 (x28-x31)
            for (unsigned i = 28; i <= 31; ++i) {
                definedRegs.push_back(i);
            }
            break;
        }

        // 存储指令不定义寄存器
        case SD:
        case SW:
        case SH:
        case SB:
        case FSD:
        case FSW: {
            break;
        }

        // 分支指令不定义寄存器
        case BEQ:
        case BNE:
        case BLT:
        case BGE:
        case BLTU:
        case BGEU:
        case BNEZ: {
            break;
        }

        // 跳转指令定义返回地址寄存器
        case JAL:
        case JALR: {
            if (operands[0]->isReg() && operands[0]->isIntegerRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

        // 浮点比较指令定义整数寄存器
        case FEQ_S:
        case FLT_S:
        case FLE_S:
        case FEQ_D:
        case FLT_D:
        case FLE_D: {
            if (operands[0]->isReg() && operands[0]->isIntegerRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

        default: {
            // 默认情况：如果第一个操作数是整数寄存器，则认为是目标寄存器
            if (!operands.empty() && operands[0]->isReg() &&
                operands[0]->isIntegerRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }
    }

    return definedRegs;
}

// 获取定义的浮点寄存器
std::vector<unsigned> Instruction::getDefinedFloatRegs() const {
    std::vector<unsigned> definedRegs;
    if (operands.empty()) return definedRegs;

    switch (opcode) {
        // 浮点算术指令：第一个操作数是目标寄存器
        case FADD_S:
        case FSUB_S:
        case FMUL_S:
        case FDIV_S:
        case FADD_D:
        case FSUB_D:
        case FMUL_D:
        case FDIV_D:
        case FMOV_S:
        case FMOV_D:
        case FSQRT_S:
        case FSQRT_D:
        case FMIN_S:
        case FMIN_D:
        case FMAX_S:
        case FMAX_D:
        case FSGNJ_S:
        case FSGNJ_D:
        case FSGNJN_S:
        case FSGNJN_D:
        case FSGNJX_S:
        case FSGNJX_D: {
            if (operands[0]->isReg() && operands[0]->isFloatRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

            // 整数到浮点转换指令 - 定义浮点寄存器
        case FCVT_S_W:
        case FCVT_S_L:
        case FCVT_S_WU:
        case FCVT_S_LU:
        case FCVT_D_W:
        case FCVT_D_L:
        case FCVT_D_WU:
        case FCVT_D_LU: {
            if (operands[0]->isReg() && operands[0]->isFloatRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

        // 浮点转换指令（浮点之间）
        case FCVT_S_D:
        case FCVT_D_S: {
            if (operands[0]->isReg() && operands[0]->isFloatRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

        // 整数到浮点移动指令定义浮点寄存器
        case FMV_W_X:
        case FMV_D_X: {
            if (operands[0]->isReg() && operands[0]->isFloatRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

        // 浮点加载指令
        case FLD:
        case FLW: {
            if (operands[0]->isReg() && operands[0]->isFloatRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }

        // 浮点存储指令不定义浮点寄存器
        case FSD:
        case FSW: {
            break;
        }

        // 浮点比较指令不定义浮点寄存器（定义整数寄存器）
        case FEQ_S:
        case FLT_S:
        case FLE_S:
        case FEQ_D:
        case FLT_D:
        case FLE_D: {
            break;
        }

            // 浮点分类指令不定义浮点寄存器（定义整数寄存器）
        case FCLASS_S:
        case FCLASS_D: {
            break;
        }

        // 浮点到整数转换指令不定义浮点寄存器
        case FCVT_W_S:
        case FCVT_L_S:
        case FCVT_WU_S:
        case FCVT_LU_S:
        case FCVT_W_D:
        case FCVT_L_D:
        case FCVT_WU_D:
        case FCVT_LU_D: {
            break;
        }

        // 浮点到整数移动指令不定义浮点寄存器
        case FMV_X_W:
        case FMV_X_D: {
            break;
        }

        case CALL: {
            // ft0-ft7 (f0-f7) -> 编号32-39
            for (unsigned i = 32; i <= 39; ++i) {
                definedRegs.push_back(i);
            }

            // fa0-fa7 (f10-f17) -> 编号42-49
            for (unsigned i = 42; i <= 49; ++i) {
                definedRegs.push_back(i);
            }

            // ft8-ft11 (f28-f31) -> 编号60-63
            for (unsigned i = 60; i <= 63; ++i) {
                definedRegs.push_back(i);
            }
            break;
        }

        default: {
            // 默认情况：如果第一个操作数是浮点寄存器，则认为是目标寄存器
            if (!operands.empty() && operands[0]->isReg() &&
                operands[0]->isFloatRegister()) {
                definedRegs.push_back(operands[0]->getRegNum());
            }
            break;
        }
    }

    return definedRegs;
}

}  // namespace riscv64
