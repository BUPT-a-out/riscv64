#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "../ABI.h"
#include "IR/BasicBlock.h"
#include "IR/Type.h"
// #include "Function.h"

inline auto getContext() {
    static const auto context = std::make_unique<midend::Context>();
    return context.get();
}

inline auto getVoidType() { return getContext()->getVoidType(); }

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

enum class RegisterType {
    Integer,  // 整数寄存器 x0-x31
    Float     // 浮点寄存器 f0-f31
};

// 操作数基类
class MachineOperand : midend::Value {
   public:
    virtual ~MachineOperand() = default;
    OperandType getType() const { return type; }

    std::string toString() const;

    virtual unsigned getRegNum() const {
        throw std::runtime_error("Not a register operand");
    }

    virtual bool isFloatRegister() const { return false; }

    virtual bool isIntegerRegister() const { return false; }

    virtual std::int64_t getValue() const {
        throw std::runtime_error("Not a immediate operand");
    }

    bool isReg() const { return type == OperandType::Register; }

    bool isImm() const { return type == OperandType::Immediate; }

    bool isMem() const { return type == OperandType::Memory; }

    bool isFrameIndex() const { return type == OperandType::FrameIndex; }

   protected:
    explicit MachineOperand(OperandType t)
        : Value(getVoidType(), midend::ValueKind::ReturnInst, ""), type(t) {}

   private:
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
          is_virtual(is_virtual),
          regType(is_virtual
                      ? RegisterType::Integer  // 虚拟寄存器默认为整数类型
                  : (reg_num >= 32 && reg_num <= 63) ? RegisterType::Float
                                                     : RegisterType::Integer) {}

    // 支持字符串构造函数
    explicit RegisterOperand(const std::string& reg_name)
        : MachineOperand(OperandType::Register),
          regNum(ABI::getRegNumFromABIName(reg_name)),  // 解析寄存器名称
          is_virtual(false),
          regType((ABI::getRegNumFromABIName(reg_name) >= 32 &&
                   ABI::getRegNumFromABIName(reg_name) <= 63)
                      ? RegisterType::Float
                      : RegisterType::Integer) {}

    explicit RegisterOperand(unsigned reg_num, bool is_virtual,
                             RegisterType type)
        : MachineOperand(OperandType::Register),
          regNum(reg_num),
          is_virtual(is_virtual),
          regType(type) {}

    unsigned getRegNum() const { return regNum; }
    void setRegNum(unsigned reg) { regNum = reg; }
    bool isVirtual() const { return is_virtual; }
    RegisterType getRegisterType() const { return regType; }

    bool isFloatRegister() const { return regType == RegisterType::Float; }
    bool isIntegerRegister() const { return regType == RegisterType::Integer; }

    std::string toString(bool use_abi = true) const;

    void setPhysicalReg(unsigned new_reg_num) {
        assert(is_virtual &&
               "Cannot set physical register for non-virtual register");
        regNum = new_reg_num;
        is_virtual = false;
        // 根据物理寄存器编号更新类型：32-63为浮点寄存器，0-31为整数寄存器
        regType = (new_reg_num >= 32 && new_reg_num <= 63)
                      ? RegisterType::Float
                      : RegisterType::Integer;
    }

    bool operator==(const RegisterOperand& other) const {
        return regNum == other.regNum && is_virtual == other.is_virtual &&
               regType == other.regType;
    }

    bool operator<(const RegisterOperand& other) const {
        // 首先按 regNum 排序
        if (regNum != other.regNum) {
            return regNum < other.regNum;
        }

        // regNum 相同时，按 is_virtual 排序 (false < true)
        if (is_virtual != other.is_virtual) {
            return is_virtual < other.is_virtual;
        }

        // regNum 和 is_virtual 都相同时，按 regType 排序
        return regType < other.regType;
    }

   private:
    unsigned regNum;
    bool is_virtual;
    RegisterType regType;
};

// 派生类：立即数操作数
class ImmediateOperand : public MachineOperand {
   public:
    RegisterType type;  // 用于标识立即数的类型，整数或浮点数

    explicit ImmediateOperand(std::int64_t value)
        : MachineOperand(OperandType::Immediate),
          type(RegisterType::Integer),
          value(value) {}

    explicit ImmediateOperand(std::int32_t int_value)
        : MachineOperand(OperandType::Immediate),
          type(RegisterType::Integer),  // 存储为 64 位
          value(static_cast<std::int64_t>(int_value)) {}

    explicit ImmediateOperand(size_t value)
        : MachineOperand(OperandType::Immediate),
          type(RegisterType::Integer),
          value(static_cast<std::int64_t>(value)) {}

    // 存储 float32
    explicit ImmediateOperand(float float_value)
        : MachineOperand(OperandType::Immediate), type(RegisterType::Float) {
        // 使用 memcpy 来安全地进行类型转换（type-punning）
        // 这是标准兼容的做法，可以避免未定义行为
        // 我们将 32 位的 float 的二进制表示复制到一个 32 位的整数中
        if (sizeof(float) != sizeof(std::int32_t)) {
            throw std::runtime_error("Size of float is not 4 bytes");
        }
        std::int32_t temp_int_bits = 0;
        std::memcpy(&temp_int_bits, &float_value, sizeof(float));

        // 然后将这个 32 位的整数存入 64 位的 value 成员
        this->value = static_cast<std::int64_t>(temp_int_bits);
    }

    RegisterType getValueType() const { return type; }
    bool isInt() const { return type == RegisterType::Integer; }
    bool isFloat() const { return type == RegisterType::Float; }

    std::int64_t getValue() const {
        return value;
    }  // 获取存储的 64 位整数值，适配旧的方法定义
    // 获取整数值 (如果不是整数则会触发断言)
    std::int32_t getIntValue() const {
        assert(isInt() &&
               "Attempted to get integer value from a float immediate operand");
        return static_cast<std::int32_t>(value);
    }

    // 获取浮点数值 (如果不是浮点数则会触发断言)
    float getFloatValue() const {
        assert(
            isFloat() &&
            "Attempted to get float value from an integer immediate operand");
        float temp_float_val;
        // 将存储的 32 位二进制表示转换回 float
        auto int_bits = static_cast<std::int32_t>(value);
        std::memcpy(&temp_float_val, &int_bits, sizeof(float));
        return temp_float_val;
    }

    std::string toString() const;

    bool operator==(const ImmediateOperand& other) const {
        return value == other.value && type == other.type;
    }

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

    bool operator==(const FrameIndexOperand& other) const {
        return index == other.index;
    }

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

    bool operator==(const MemoryOperand& other) const {
        return *baseReg == *other.baseReg && *offsetVal == *other.offsetVal;
    }

   private:
    std::unique_ptr<RegisterOperand> baseReg;
    std::unique_ptr<ImmediateOperand> offsetVal;
};

// 派生类：标签操作数 (用于跳转指令)
class LabelOperand : public MachineOperand {
   public:
    // 重定位类型：用于打印 %pcrel_hi / %pcrel_lo 形式
    enum class RelocKind { None, PCREL_HI, PCREL_LO };

    explicit LabelOperand(midend::BasicBlock* block)  // 指向基本块的指针
        : MachineOperand(OperandType::Label),
          block(block),
          labelName(block->getName()),
          relocKind(RelocKind::None),
          offset(0) {}

    // 支持字符串构造函数
    explicit LabelOperand(const std::string& labelName)
        : MachineOperand(OperandType::Label),
          block(nullptr),
          labelName(labelName),
          relocKind(RelocKind::None),
          offset(0) {}

    explicit LabelOperand(midend::BasicBlock* block, std::string labelName)
        : MachineOperand(OperandType::Label),
          block(block),
          labelName(std::move(labelName)),
          relocKind(RelocKind::None),
          offset(0) {}

    // 带偏移与重定位类型的构造函数（用于 auipc/addi PC 相对地址计算）
    LabelOperand(std::string labelName, std::int64_t offset, RelocKind kind)
        : MachineOperand(OperandType::Label),
          block(nullptr),
          labelName(std::move(labelName)),
          relocKind(kind),
          offset(offset) {}

    RelocKind getRelocKind() const { return relocKind; }
    std::int64_t getOffset() const { return offset; }

    midend::BasicBlock* getBlock() const { return block; }
    const std::string& getLabelName() const { return labelName; }

    std::string toString() const;

   private:
    midend::BasicBlock* block;
    std::string labelName;
    RelocKind relocKind;
    std::int64_t offset;  // 可以为 0
};

}  // namespace riscv64

// 为 RegisterOperand 定义哈希函数
namespace std {
template <>
struct hash<riscv64::RegisterOperand> {
    std::size_t operator()(const riscv64::RegisterOperand& reg) const {
        // 将三个字段组合成一个哈希值
        std::size_t h1 = std::hash<unsigned>{}(reg.getRegNum());
        std::size_t h2 = std::hash<bool>{}(reg.isVirtual());
        std::size_t h3 =
            std::hash<int>{}(static_cast<int>(reg.getRegisterType()));

        // 使用位移和异或来组合哈希值
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};
}  // namespace std
