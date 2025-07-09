#include "CodeGen.h"

namespace riscv64 {

std::string CodeGenerator::allocateReg() {
    return "t" + std::to_string(nextRegNum_++);
}

std::string CodeGenerator::getOrAllocateReg(const midend::Value* val) {
    auto it = valueToReg_.find(val);
    if (it != valueToReg_.end()) {
        return it->second;
    }
    
    // 检查是否是常量
    if (auto* constant = dynamic_cast<const midend::Constant*>(val)) {
        // 对于立即数，可能需要先加载到寄存器
        std::string reg = allocateReg();
        valueToReg_[val] = reg;
        return reg;
    }
    
    std::string reg = allocateReg();
    valueToReg_[val] = reg;
    return reg;
}

}  // namespace riscv64