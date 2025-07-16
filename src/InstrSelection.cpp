#include <memory>
#include <stdexcept>

#include "CodeGen.h"

namespace riscv64 {

std::string CodeGenerator::generateInstruction(
    const midend::Instruction* inst) {
    if (inst->isUnaryOp()) {
        // TODO(rikka): ...
        return "";
    }

    return "# Not implemented instr: " +
           std::to_string(static_cast<int>(inst->getOpcode()));
}

RegisterOperand* CodeGenerator::mapValueToReg(const midend::Value* val,
                                              unsigned reg_num,
                                              bool is_virtual) {
    // 创建新的 RegisterOperand 并使用智能指针管理
    auto reg = std::make_unique<RegisterOperand>(reg_num, is_virtual);
    auto* regPtr = reg.get();
    valueToReg_[val] = std::move(reg);
    return regPtr;
}

LabelOperand* CodeGenerator::mapBBToLabel(const midend::BasicBlock* bb,
                                          const std::string& label) {
    // 创建新的 LabelOperand 并使用智能指针管理
    auto labelOp =
        std::make_unique<LabelOperand>(const_cast<midend::BasicBlock*>(bb));
    auto* labelPtr = labelOp.get();
    bbToLabel_[bb] = std::move(labelOp);
    return labelPtr;
}

RegisterOperand* CodeGenerator::getRegForValue(const midend::Value* val) const {
    auto it = valueToReg_.find(val);
    if (it != valueToReg_.end()) {
        return it->second.get();
    }
    throw std::runtime_error("No register allocated for value: " +
                             val->getName());
}

LabelOperand* CodeGenerator::getLabelForBB(const midend::BasicBlock* bb) const {
    auto it = bbToLabel_.find(bb);
    if (it != bbToLabel_.end()) {
        return it->second.get();
    }
    throw std::runtime_error("No label allocated for basic block: " +
                             bb->getName());
}

void CodeGenerator::reset() {
    valueToReg_.clear();  // 智能指针会自动释放内存
    bbToLabel_.clear();   // 智能指针会自动释放内存
    nextRegNum_ = 0;
    nextLabelNum_ = 0;
}

}  // namespace riscv64