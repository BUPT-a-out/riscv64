#pragma once

#include <memory>
#include <optional>

#include "../../midend/include/IR/Module.h"
#include "Instructions/All.h"
#include "Segment.h"

namespace riscv64 {

class CodeGenerator;

class Visitor {
   public:
    explicit Visitor(CodeGenerator* code_gen);
    ~Visitor() = default;

    // 禁止复制，允许移动
    Visitor(const Visitor&) = delete;
    Visitor& operator=(const Visitor&) = delete;
    Visitor(Visitor&&) = default;
    Visitor& operator=(Visitor&&) = default;

    // 访问方法声明
    Module visit(const midend::Module* module);
    void visit(const midend::Function* func, Module* parent_module);
    BasicBlock* visit(const midend::BasicBlock* bb, Function* parent_func);
    std::unique_ptr<MachineOperand> visit(const midend::Instruction* inst,
                                          BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visit(const midend::Value* value,
                                          BasicBlock* parent_bb);
    void visitRetInstruction(const midend::Instruction* retInst,
                             BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitUnaryOp(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitBinaryOp(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitFloatBinaryOp(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitAllocaInst(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitLoadInst(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    void visitStoreInst(const midend::Instruction* inst, BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitPhiInst(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    void visitBranchInst(const midend::Instruction* inst,
                         BasicBlock* parent_bb);
    std::unique_ptr<RegisterOperand> immToReg(
        std::unique_ptr<MachineOperand> operand, BasicBlock* parent_bb);
    std::unique_ptr<RegisterOperand> ensureFloatReg(
        std::unique_ptr<MachineOperand> operand, BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitCallInst(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitGEPInst(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitCastInst(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    void storeOperandToReg(
        std::unique_ptr<MachineOperand> source_operand,
        std::unique_ptr<MachineOperand> reg_operand, BasicBlock* parent_bb,
        std::list<std::unique_ptr<Instruction>>::const_iterator insert_pos =
            {});
    std::unique_ptr<MachineOperand> funcArgToReg(
        const midend::Argument* argument, BasicBlock* parent_bb = nullptr);

    void assignVirtRegsToFuncArgs(midend::Function* func);

    void createCFG(Function* func);

    void visit(const midend::Constant* constant);
    void visit(const midend::GlobalVariable* var, Module* parent_module);

    ConstantInitializer convertLLVMInitializerToConstantInitializer(
        const midend::Value* init, const midend::Type* type);
    CompilerType convertLLVMTypeToCompilerType(const midend::Type* llvm_type);

   private:
    CodeGenerator* codeGen_;

    std::optional<RegisterOperand*> findRegForValue(const midend::Value* value);

    // 辅助函数：处理大偏移量的内存操作
    bool isValidImmediateOffset(int64_t offset);
    std::unique_ptr<RegisterOperand> handleLargeOffset(
        std::unique_ptr<RegisterOperand> base_reg, int64_t offset,
        BasicBlock* parent_bb);
    void generateMemoryInstruction(Opcode opcode,
                                   std::unique_ptr<RegisterOperand> target_reg,
                                   std::unique_ptr<RegisterOperand> base_reg,
                                   int64_t offset, BasicBlock* parent_bb);
};

}  // namespace riscv64