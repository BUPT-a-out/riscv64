#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "ABI.h"
#include "IR/BasicBlock.h"
// #include "Function.h"

namespace riscv64 {

class BasicBlock;  // 前向声明
class Function;    // 前向声明

enum class OperandType {
    Register,    // 寄存器
    Immediate,   // 立即数
    Label,       // 标签 (指向一个基本块)
    Memory,      // 内存地址 [base + offset]
    FrameIndex,  // 栈帧索引 (用于访问栈上的变量)
};

// 操作数基类
class MachineOperand {
   public:
    virtual ~MachineOperand() = default;
    OperandType getType() const { return type; }

    std::string toString() const;

    virtual unsigned getRegNum() const {
        throw std::runtime_error("Not a register operand");
    }

    virtual std::int64_t getValue() const {
        throw std::runtime_error("Not a immediate operand");
    }

    bool isReg() const { return type == OperandType::Register; }

    bool isImm() const { return type == OperandType::Immediate; }

   protected:
    explicit MachineOperand(OperandType t) : type(t) {}
    OperandType type;
};

// 派生类：寄存器操作数
class RegisterOperand : public MachineOperand {
   public:
    // regNum: 寄存器的编号 (例如 x10 就是 10)
    // isVirtual: 是否是虚拟寄存器
    explicit RegisterOperand(unsigned reg_num, bool is_virtual = true)
        : MachineOperand(OperandType::Register),
          regNum(reg_num),
          is_virtual(is_virtual) {}

    // 支持字符串构造函数
    explicit RegisterOperand(const std::string& reg_name)
        : MachineOperand(OperandType::Register),
          regNum(ABI::getRegNumFromABIName(reg_name)),  // 解析寄存器名称
          is_virtual(false) {}

    unsigned getRegNum() const { return regNum; }
    bool isVirtual() const { return is_virtual; }

    // unsigned abiToRegNum() const;

    std::string toString(bool use_abi = true) const;

    void setPhysicalReg(unsigned new_reg_num) {
        assert(is_virtual &&
               "Cannot set physical register for non-virtual register");
        regNum = new_reg_num;
        is_virtual = false;
    }

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

class FrameIndexOperand : public MachineOperand {
   public:
    // 用于栈帧索引的操作数
    explicit FrameIndexOperand(int index)
        : MachineOperand(OperandType::FrameIndex), index(index) {}

    int getIndex() const { return index; }

    std::string toString() const;

   private:
    int index;
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
        : MachineOperand(OperandType::Label),
          block(block),
          labelName(block->getName()) {}

    // 支持字符串构造函数
    explicit LabelOperand(const std::string& labelName)
        : MachineOperand(OperandType::Label),
          block(nullptr),
          labelName(labelName) {}

    explicit LabelOperand(midend::BasicBlock* block, std::string labelName)
        : MachineOperand(OperandType::Label),
          block(block),
          labelName(std::move(labelName)) {}

    midend::BasicBlock* getBlock() const { return block; }
    const std::string& getLabelName() const { return labelName; }

    std::string toString() const;

   private:
    midend::BasicBlock* block;
    std::string labelName;
};

}  // namespace riscv64