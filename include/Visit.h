#pragma once

#include <memory>
#include <optional>

#include "IR/Module.h"
#include "Instructions/MachineOperand.h"

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
    void visit(const midend::Module* module);
    void visit(const midend::Function* func);
    void visit(const midend::BasicBlock* bb);
    void visit(const midend::Instruction* inst);
    void visitRetInstruction(const midend::Instruction* retInst);

    std::unique_ptr<MachineOperand> visit(const midend::Value* value);
    void visit(const midend::Constant* constant);
    void visit(const midend::GlobalVariable* var);

   private:
    CodeGenerator* codeGen_;

    std::optional<std::string> findRegForValue(const midend::Value* value);
};

}  // namespace riscv64
