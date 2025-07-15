#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "IR/BasicBlock.h"
namespace riscv64 {

class BasicBlock;  // 前向声明

enum class OperandType {
    Register,   // 寄存器
    Immediate,  // 立即数
    Label,      // 标签 (指向一个基本块)
    Memory      // 内存地址 [base + offset]
};

// 操作数基类
class MachineOperand {
   public:
    virtual ~MachineOperand() = default;
    OperandType getType() const { return type; }

    std::string toString() const;

   protected:
    explicit MachineOperand(OperandType t) : type(t) {}
    OperandType type;
};

// 派生类：寄存器操作数
class RegisterOperand : public MachineOperand {
   public:
    // regNum: 寄存器的编号 (例如 x10 就是 10)
    // isVirtual: 是否是虚拟寄存器
    explicit RegisterOperand(unsigned regNum, bool is_virtual = false)
        : MachineOperand(OperandType::Register),
          regNum(regNum),
          is_virtual(is_virtual) {}

    // 支持字符串构造函数
    // explicit RegisterOperand(const std::string& regName)
    //     : MachineOperand(OperandType::Register),
    //       regNum(0),  // 解析寄存器名称
    //       is_virtual(false) {
    //     // TODO(rikka): 解析寄存器名称到编号的逻辑
    // }

    unsigned getRegNum() const { return regNum; }
    bool isVirtual() const { return is_virtual; }

    unsigned abiToRegNum() const;

    std::string toString() const;

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

    std::string toString() const;

   private:
    std::int64_t value;
};

// 派生类：内存操作数
// 例如：lw x1, 0(x2) -> base = x2, offset = 0
// 这里的 base 是寄存器，offset 是立即数
class MemoryOperand : public MachineOperand {
   public:
    // 使用智能指针管理操作数
    MemoryOperand(std::unique_ptr<RegisterOperand> base,
                  std::unique_ptr<ImmediateOperand> offset)
        : MachineOperand(OperandType::Memory),
          baseReg(std::move(base)),
          offsetVal(std::move(offset)) {}

    RegisterOperand* getBaseReg() const { return baseReg.get(); }
    ImmediateOperand* getOffset() const { return offsetVal.get(); }

    std::string toString() const;

   private:
    std::unique_ptr<RegisterOperand> baseReg;
    std::unique_ptr<ImmediateOperand> offsetVal;
};

// 派生类：标签操作数 (用于跳转指令)
class LabelOperand : public MachineOperand {
   public:
    explicit LabelOperand(midend::BasicBlock* block)  // 指向基本块的指针
        : MachineOperand(OperandType::Label), block(block) {}

    // 支持字符串构造函数
    explicit LabelOperand(const std::string& labelName)
        : MachineOperand(OperandType::Label),
          block(nullptr),
          labelName(labelName) {}

    midend::BasicBlock* getBlock() const { return block; }
    const std::string& getLabelName() const { return labelName; }

    std::string toString() const;

   private:
    midend::BasicBlock* block;
    std::string labelName;
};

}  // namespace riscv64