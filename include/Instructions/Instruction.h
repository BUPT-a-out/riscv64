#pragma once

// #include "BasicBlock.h"
#include <list>
#include <memory>
#include <string>
#include <vector>

#include "MachineOperand.h"

namespace riscv64 {

class BasicBlock;  // 前向声明

// RISC-V 指令操作码枚举 (仅为示例)
enum Opcode {
    // R-Type
    ADD,
    SUB,
    SLT,
    // I-Type
    ADDI,
    SLTI,
    LW,
    // S-Type
    SW,
    // B-Type
    BEQ,
    BNE,
    // J-Type
    JAL,
    // ... 其他指令
    // 伪指令
    LI,
    MV,
    CALL
};

class Instruction {
   public:
    explicit Instruction(Opcode op) : opcode(op) {}
    explicit Instruction(Opcode op, BasicBlock* parent)
        : opcode(op), parent(parent) {}

    // 添加操作数
    void addOperand(std::unique_ptr<MachineOperand> operand) {
        operands.push_back(std::move(operand));
    }

    Opcode getOpcode() const { return opcode; }
    const std::vector<std::unique_ptr<MachineOperand>>& getOperands() const {
        return operands;
    }
    const auto getOprandCount() const { return operands.size(); }

    // 为了方便，可以提供一些辅助函数
    // 例如：获取第 n 个操作数
    MachineOperand* getOperand(std::size_t n) const {
        return operands[n].get();
    }

    BasicBlock* getParent() const { return parent; }
    void setParent(BasicBlock* bb) { parent = bb; }

    // 生成文本表示
    std::string toString() const;

   private:
    Opcode opcode;
    std::vector<std::unique_ptr<MachineOperand>> operands;

    // 指向其所在的基本块 (可选，但非常有用) - 弱引用
    BasicBlock* parent{};
};

}  // namespace riscv64