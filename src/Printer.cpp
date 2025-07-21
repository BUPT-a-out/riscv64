#include <stdexcept>
#include <string>
#include <unordered_map>

#include "Instructions/All.h"

namespace riscv64 {
// Machine Operand 相关的 toString 实现
std::string MachineOperand::toString() const {
    switch (getType()) {
        case OperandType::Register:
            return dynamic_cast<const RegisterOperand*>(this)->toString();
        case OperandType::Immediate:
            return dynamic_cast<const ImmediateOperand*>(this)->toString();
        case OperandType::Label:
            return dynamic_cast<const LabelOperand*>(this)->toString();
        case OperandType::Memory:
            return dynamic_cast<const MemoryOperand*>(this)->toString();
        case OperandType::FrameIndex:
            return dynamic_cast<const FrameIndexOperand*>(this)->toString();
        default:
            throw std::runtime_error("Unknown operand type");
    }
}

std::string RegisterOperand::toString(bool use_abi) const {
    if (isVirtual()) {
        return "%vreg_" + std::to_string(regNum);
    }
    if (use_abi) {
        return ABI::getABINameFromRegNum(regNum);
    }
    return "x" + std::to_string(regNum);
}

std::string ImmediateOperand::toString() const { return std::to_string(value); }

std::string LabelOperand::toString() const { return getLabelName(); }

std::string MemoryOperand::toString() const {
    return std::to_string(getOffset()->getValue()) + "(" +
           getBaseReg()->toString() + ")";
}

std::string FrameIndexOperand::toString() const {
    return "FI(" + std::to_string(getIndex()) + ")";
}

std::string getInstructionName(Opcode opcode) {
    static const std::unordered_map<Opcode, std::string> opcodeNames = {
        {Opcode::ADD, "add"},
        {Opcode::SUB, "sub"},
        {Opcode::MUL, "mul"},
        {Opcode::DIV, "div"},
        {Opcode::REM, "rem"},
        {Opcode::XOR, "xor"},
        {Opcode::AND, "and"},
        {Opcode::RET, "ret"},
        {Opcode::LI, "li"},
        {Opcode::MV, "mv"},
        {Opcode::ADDI, "addi"},
        {Opcode::BNEZ, "bnez"},
        {Opcode::SEQZ, "seqz"},
        {Opcode::SNEZ, "snez"},
        {Opcode::SLTZ, "sltz"},
        {Opcode::SGTZ, "sgtz"},
        {Opcode::BEQZ, "beqz"},
        {Opcode::BGTZ, "bgtz"},
        {Opcode::BGEZ, "bgez"},
        {Opcode::BLTZ, "bltz"},
        {Opcode::BGT, "bgt"},
        {Opcode::BLE, "ble"},
        {Opcode::BGTU, "bgtu"},
        {Opcode::J, "j"},
        {Opcode::SLT, "slt"},
        {Opcode::SGT, "sgt"},
        {Opcode::SLTI, "slti"},
        {Opcode::CALL, "call"},
        {Opcode::FRAMEADDR, "frameaddr"},
        {Opcode::SW, "sw"},
        {Opcode::LW, "lw"},
        {Opcode::BEQZ, "beqz"},
        {Opcode::SD, "sd"},
        {Opcode::LD, "ld"},
        {Opcode::SLL, "sll"},
        {Opcode::SRL, "srl"},
        {Opcode::SLLI, "slli"},
        {Opcode::SRLI, "srli"},
        {Opcode::SRA, "sra"},
        {Opcode::OR, "or"},
        {Opcode::MULH, "mulh"},
        {Opcode::MULHSU, "mulhsu"},
        {Opcode::MULHU, "mulhu"},
        {Opcode::DIVU, "divu"},
        {Opcode::REMU, "remu"},
        {Opcode::SLTU, "sltu"},
        {Opcode::BEQ, "beq"},
        {Opcode::BNE, "bne"},
        {Opcode::BLT, "blt"},
        {Opcode::BGE, "bge"},
        {Opcode::BLTU, "bltu"},
        {Opcode::BGEU, "bgeu"},
        {Opcode::ORI, "ori"},
        {Opcode::XORI, "xori"},
        {Opcode::ANDI, "andi"},
        {Opcode::LUI, "lui"},
        // TODO(rikka): 添加其他操作码的名称...
    };

    auto it = opcodeNames.find(opcode);
    if (it != opcodeNames.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown opcode: " +
                             std::to_string(static_cast<int>(opcode)));
}

std::string Instruction::toString() const {
    auto result = getInstructionName(opcode);
    auto operand_count = getOprandCount();
    if (operand_count == 0) {
        return result;
    }

    result += ' ';
    for (size_t index = 0; index < operand_count; ++index) {
        if (index > 0) {
            result += ", ";
        }
        result += getOperand(index)->toString();
    }

    return result;
}

std::string BasicBlock::toString() const {
    std::string result = label + ":\n";
    for (const auto& inst : instructions) {
        result += "  " + inst->toString() + "\n";
    }
    return result;
}

std::string Function::toString() const {
    std::string result = name + ":\n";
    for (const auto& bb : basic_blocks) {
        result += bb->toString();
    }
    return result;
}

std::string Module::toString() const {
    std::string result;
    result += "  .text\n";
    // for (const auto& global : global_vars) {
    //     result += "  " + global->toString() + "\n";
    // }
    // TODO(rikka): 处理全局变量的输出，修改数据结构
    for (const auto& func : functions) {
        result += "  .globl " + func->getName() + "\n";
        result += func->toString() + "\n";
    }
    return result;
}

}  // namespace riscv64