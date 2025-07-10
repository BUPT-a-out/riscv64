#pragma once
#include "IR/Module.h"
#include <string>
#include <unordered_map>

namespace riscv64 {

class CodeGenerator {
private:
    std::unordered_map<const midend::Value*, std::string> valueToReg_;
    std::unordered_map<const midend::BasicBlock*, std::string> bbToLabel_;
    int nextRegNum_ = 0;
    int nextLabelNum_ = 0;
    
    std::string allocateReg();
    std::string getOrAllocateReg(const midend::Value* val);
    std::string getBBLabel(const midend::BasicBlock* bb);
    
public:
    std::vector<std::string> generateFunction(const midend::Function* func);
    std::vector<std::string> generateBasicBlock(const midend::BasicBlock* bb);
    std::string generateInstruction(const midend::Instruction* inst);

    // 维护映射关系
    void mapValueToReg(const midend::Value* val, const std::string& reg);
    void mapBBToLabel(const midend::BasicBlock* bb, const std::string& label);
    std::string getRegForValue(const midend::Value* val) const;
    std::string getLabelForBB(const midend::BasicBlock* bb) const;
    int getNextRegNum() { return nextRegNum_++; }
    int getNextLabelNum() { return nextLabelNum_++; }
    void reset();
};

}  // namespace riscv64