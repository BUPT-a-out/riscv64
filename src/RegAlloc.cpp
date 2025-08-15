#include <memory>

#include "CodeGen.h"

namespace riscv64 {

std::unique_ptr<RegisterOperand> CodeGenerator::allocateIntReg() {
    return std::make_unique<RegisterOperand>(nextRegNum_++,
                                             true);  // 分配一个新的虚拟寄存器
}

RegisterOperand* CodeGenerator::getOrAllocateIntReg(const midend::Value* val) {
    auto it = valueToReg_.find(val);
    if (it != valueToReg_.end()) {
        return it->second.get();
    }

    // 检查是否是常量
    if (auto* constant = dynamic_cast<const midend::Constant*>(val)) {
        // 对于立即数，可能需要先加载到寄存器
        auto reg = allocateIntReg();
        auto* regPtr = reg.get();
        valueToReg_[val] = std::move(reg);
        return regPtr;
    }

    auto reg = allocateIntReg();
    auto* regPtr = reg.get();
    valueToReg_[val] = std::move(reg);
    return regPtr;
}

std::unique_ptr<RegisterOperand> CodeGenerator::allocateFloatReg() {
    return std::make_unique<RegisterOperand>(nextRegNum_++, true,
                                             RegisterType::Float);
}

RegisterOperand* CodeGenerator::getOrAllocateFloatReg(
    const midend::Value* val) {
    auto it = valueToReg_.find(val);
    if (it != valueToReg_.end()) {
        return it->second.get();
    }

    // 检查是否是浮点常量
    if (auto* constant = dynamic_cast<const midend::Constant*>(val)) {
        auto reg = allocateFloatReg();
        auto* regPtr = reg.get();
        valueToReg_[val] = std::move(reg);
        return regPtr;
    }

    auto reg = allocateFloatReg();
    auto* regPtr = reg.get();
    valueToReg_[val] = std::move(reg);
    return regPtr;
}

std::unique_ptr<RegisterOperand> CodeGenerator::allocateReg(bool is_float) {
    if (is_float) {
        return allocateFloatReg();
    }
    return allocateIntReg();
}

RegisterOperand* CodeGenerator::getOrAllocateReg(const midend::Value* val,
                                                 bool is_float) {
    if (is_float) {
        return getOrAllocateFloatReg(val);
    }
    return getOrAllocateIntReg(val);
}

}  // namespace riscv64