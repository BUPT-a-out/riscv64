#pragma once

#include <cstdint>
namespace riscv64 {

class BasicBlock;  // 前向声明

enum class OperandType {
    Register,   // 寄存器
    Immediate,  // 立即数
    Label       // 标签 (指向一个基本块)
    // ...
};

// 操作数基类
class MachineOperand {
   public:
    virtual ~MachineOperand() = default;
    OperandType getType() const { return type; }

   protected:
    explicit MachineOperand(OperandType t) : type(t) {}
    OperandType type;
};

// 派生类：寄存器操作数
class RegisterOperand : public MachineOperand {
   public:
    // regNum: 寄存器的编号 (例如 x10 就是 10)
    // isVirtual: 是否是虚拟寄存器
    RegisterOperand(unsigned regNum, bool is_virtual = false)
        : MachineOperand(OperandType::Register),
          regNum(regNum),
          is_virtual(is_virtual) {}

    unsigned getRegNum() const { return regNum; }
    bool isVirtual() const { return is_virtual; }

   private:
    unsigned regNum;
    bool is_virtual;
};

// 派生类：立即数操作数
class ImmediateOperand : public MachineOperand {
   public:
    explicit ImmediateOperand(std::int64_t value)
        : MachineOperand(OperandType::Immediate), value(value) {}

    std::int64_t getValue() const { return value; }

   private:
    std::int64_t value;
};

// 派生类：标签操作数 (用于跳转指令)
class LabelOperand : public MachineOperand {
   public:
    explicit LabelOperand(BasicBlock* block)  // 指向基本块的指针
        : MachineOperand(OperandType::Label), block(block) {}

    BasicBlock* getBlock() const { return block; }

   private:
    BasicBlock* block;
};

}  // namespace riscv64