#pragma once
#include <memory>
#include <string>
#include <unordered_map>

#include "IR/Module.h"
#include "Instructions/MachineOperand.h"
#include "Visit.h"

namespace riscv64 {

class Visitor;  // 前向声明

class CodeGenerator {
   private:
    std::unordered_map<const midend::Value*, std::unique_ptr<RegisterOperand>>
        valueToReg_;
    std::unordered_map<const midend::BasicBlock*, std::unique_ptr<LabelOperand>>
        bbToLabel_;
    int nextRegNum_ = 100;
    int nextLabelNum_ = 0;

   public:
    CodeGenerator() { visitor_ = std::make_unique<Visitor>(this); };
    ~CodeGenerator() = default;
    // std::vector<std::string> generateFunction(const midend::Function* func);
    // std::vector<std::string> generateBasicBlock(const midend::BasicBlock*
    // bb);
    std::string generateInstruction(const midend::Instruction* inst);

    // 维护映射关系
    RegisterOperand* mapValueToReg(const midend::Value* val, unsigned reg_num,
                                   bool is_virtual = false);
    LabelOperand* mapBBToLabel(const midend::BasicBlock* bb,
                               const std::string& label);
    RegisterOperand* getRegForValue(const midend::Value* val) const;
    LabelOperand* getLabelForBB(const midend::BasicBlock* bb) const;
    int getNextRegNum() { return nextRegNum_++; }
    int getNextLabelNum() { return nextLabelNum_++; }
    void reset();

    std::unique_ptr<RegisterOperand> allocateReg();
    RegisterOperand* getOrAllocateReg(const midend::Value* val);
    LabelOperand* getBBLabel(const midend::BasicBlock* bb);

    // class Visitor;
    std::unique_ptr<Visitor> visitor_;
};

}  // namespace riscv64