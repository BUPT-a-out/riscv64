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

void CodeGenerator::mapValueToReg(const midend::Value* val,
                                  const std::string& reg) {
    valueToReg_[val] = reg;
}

void CodeGenerator::mapBBToLabel(const midend::BasicBlock* bb,
                                 const std::string& label) {
    bbToLabel_[bb] = label;
}

std::string CodeGenerator::getRegForValue(const midend::Value* val) const {
    auto it = valueToReg_.find(val);
    if (it != valueToReg_.end()) {
        return it->second;
    }
    throw std::runtime_error("No register allocated for value: " +
                             val->getName());
}

std::string CodeGenerator::getLabelForBB(const midend::BasicBlock* bb) const {
    auto it = bbToLabel_.find(bb);
    if (it != bbToLabel_.end()) {
        return it->second;
    }
    throw std::runtime_error("No label allocated for basic block: " +
                             bb->getName());
}

void CodeGenerator::reset() {
    valueToReg_.clear();
    bbToLabel_.clear();
    nextRegNum_ = 0;
    nextLabelNum_ = 0;
}

}  // namespace riscv64