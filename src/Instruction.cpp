#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"


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

}  // namespace riscv64