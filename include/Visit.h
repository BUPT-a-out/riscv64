#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "IR/Module.h"
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
    std::unique_ptr<MachineOperand> visitLogicalOp(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitAllocaInst(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitLoadInst(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    void visitStoreInst(const midend::Instruction* inst, BasicBlock* parent_bb);
    std::unique_ptr<MachineOperand> visitPhiInst(
        const midend::Instruction* inst, BasicBlock* parent_bb);
    void processDeferredPhiNode(const midend::Instruction* inst,
                                const midend::BasicBlock* bb_midend,
                                Function* parent_func);
    void processAllPhiNodes(const midend::Function* func,
                            Function* parent_func);
    void visitBranchInst(const midend::Instruction* inst,
                         BasicBlock* parent_bb);
    std::unique_ptr<RegisterOperand> immToReg(
        std::unique_ptr<MachineOperand> operand, BasicBlock* parent_bb);
    std::unique_ptr<RegisterOperand> floatImmToReg(
        std::unique_ptr<ImmediateOperand> operand, BasicBlock* parent_bb);
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

    template <typename T, typename ConstantType>
    std::vector<T> processTypedArray(
        const midend::ConstantArray* const_array, const midend::Type* type,
        T default_value, std::function<T(const ConstantType*)> extractor);

    template <typename T>
    std::vector<T> processMultiDimArray(
        const midend::ConstantArray* const_array, const midend::Type* type,
        const midend::Type* element_type, T default_value);

    // 返回一个新的寄存器操作数
    static std::unique_ptr<RegisterOperand> cloneRegister(RegisterOperand* reg_op);
    static std::unique_ptr<RegisterOperand> cloneRegister(RegisterOperand* reg_op,
                                                   bool is_float);

    static void createCFG(Function* func);

    void visit(const midend::Constant* constant);
    void visit(const midend::GlobalVariable* var, Module* parent_module);

    void preProcessFuncArgs(const midend::Function* midend_func,
                            midend::BasicBlock* midend_bb,
                            Function* riscv_func);

    ConstantInitializer convertLLVMInitializerToConstantInitializer(
        const midend::Value* init, const midend::Type* type);
    CompilerType convertLLVMTypeToCompilerType(const midend::Type* llvm_type);

    static bool isValidImmediateOffset(int64_t offset);

   private:
    CodeGenerator* codeGen_;

    // PHI节点并行拷贝相关的结构体和函数
    struct CopyOperation {
        RegisterOperand* dest_reg;                    // 目标寄存器
        std::unique_ptr<MachineOperand> src_operand;  // 源操作数
        bool is_constant;                             // 是否是常量拷贝
    };

    void generateParallelCopyForEdge(
        const std::vector<const midend::Instruction*>& phi_nodes,
        const midend::BasicBlock* pred_bb_midend, Function* parent_func);

    void scheduleParallelCopy(
        std::vector<CopyOperation>& copy_ops, BasicBlock* bb,
        std::list<std::unique_ptr<Instruction>>::const_iterator insert_pos);

    void scheduleRegisterCopies(
        std::vector<CopyOperation>& register_copies, BasicBlock* bb,
        std::list<std::unique_ptr<Instruction>>::const_iterator insert_pos);

    void generateCopyInstruction(
        RegisterOperand* dest_reg, std::unique_ptr<MachineOperand> src_operand,
        BasicBlock* bb,
        std::list<std::unique_ptr<Instruction>>::const_iterator insert_pos);

    std::optional<RegisterOperand*> findRegForValue(const midend::Value* value);

    // 辅助函数：处理大偏移量的内存操作
    std::unique_ptr<RegisterOperand> handleLargeOffset(
        std::unique_ptr<RegisterOperand> base_reg, int64_t offset,
        BasicBlock* parent_bb);
    void generateMemoryInstruction(Opcode opcode,
                                   std::unique_ptr<RegisterOperand> target_reg,
                                   std::unique_ptr<RegisterOperand> base_reg,
                                   int64_t offset, BasicBlock* parent_bb);
};

namespace CFG {
void print(Function* func);
} // namespace CFG

}  // namespace riscv64
