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

std::string LabelOperand::toString() const { return ":"; }

std::string MemoryOperand::toString() const {
    return std::to_string(getOffset()->getValue()) + "(" +
           getBaseReg()->toString() + ")";
}

std::string getInstructionName(Opcode opcode) {
    static const std::unordered_map<Opcode, std::string> opcodeNames = {
        {Opcode::ADD, "add"}, {Opcode::SUB, "sub"},   {Opcode::MUL, "mul"},
        {Opcode::DIV, "div"}, {Opcode::RET, "ret"},   {Opcode::LI, "li"},
        {Opcode::MV, "mv"},   {Opcode::ADDI, "addi"},
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
    for (const auto& func : functions) {
        result += func->toString() + "\n";
    }
    return result;
}

}  // namespace riscv64