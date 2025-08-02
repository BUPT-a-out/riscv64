#pragma once
#include <memory>
#include <string>
#include <unordered_map>

#include "FloatConstantPool.h"
#include "FrameIndexPass.h"
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
    int nextFloatRegNum_ = 1000;  // 使用不同的编号空间避免冲突
    int nextLabelNum_ = 0;
    std::unique_ptr<FloatConstantPool> floatConstantPool_;

   public:
    CodeGenerator() {
        visitor_ = std::make_unique<Visitor>(this);
        floatConstantPool_ = std::make_unique<FloatConstantPool>();
    };
    ~CodeGenerator() = default;

    std::string generateInstruction(const midend::Instruction* inst);

    // 维护映射关系
    RegisterOperand* mapValueToReg(const midend::Value* val, unsigned reg_num,
                                   bool is_virtual = false);
    RegisterOperand* mapValueToReg(const midend::Value* val, unsigned reg_num,
                                   bool is_virtual, RegisterType reg_type);
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

    std::unique_ptr<RegisterOperand> allocateFloatReg();
    RegisterOperand* getOrAllocateFloatReg(const midend::Value* val);

    // 浮点常量池相关方法
    FloatConstantPool* getFloatConstantPool() {
        return floatConstantPool_.get();
    }

    std::unique_ptr<Visitor> visitor_;

    // 运行栈帧布局Pass
    void runFrameIndexPass(Function* func) {
        FrameIndexPass framePass(func);
        framePass.run();
    }

    // 清理函数级别的映射（保留全局变量映射）
    void clearFunctionLevelMappings() { valueToReg_.clear(); }
};

}  // namespace riscv64