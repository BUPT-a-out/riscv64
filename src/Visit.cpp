#include "CodeGen.h"
#include "IR/Module.h"
#include "Instructions/MachineOperand.h"


class riscv64::CodeGenerator::Visitor {
   public:
    explicit Visitor(CodeGenerator* code_gen) : codeGen_(code_gen) {}
    ~Visitor() = default;
    // 其他构造
    Visitor(const Visitor&) = delete;
    Visitor& operator=(const Visitor&) = delete;
    Visitor(Visitor&&) = default;
    Visitor& operator=(Visitor&&) = default;

    // 访问 module
    void visit(const midend::Module* module) {
        for (const auto& global : module->globals()) {
            visit(global);
        }
        for (auto* const func : *module) {
            visit(func);
        }
    }

    // 访问函数
    void visit(const midend::Function* func) {
        // 其他操作...
        for (const auto& bb : *func) {
            visit(bb);
        }
    }

    // 访问基本块
    void visit(const midend::BasicBlock* bb) {
        // 其他操作...
        for (const auto& inst : *bb) {
            visit(inst);
        }
    }

    // 访问指令
    void visit(const midend::Instruction* inst) {
        switch (inst->getOpcode()) {
            case midend::Opcode::Add:
            case midend::Opcode::Sub:
            case midend::Opcode::Mul:
            case midend::Opcode::Div:
                // 处理算术指令
                break;
            case midend::Opcode::Load:
            case midend::Opcode::Store:
                // 处理内存操作指令
                break;
            case midend::Opcode::Br:
            case midend::Opcode::Ret:
                // 处理控制流指令
                break;
            default:
                // 其他指令类型
                break;
        }
    }

    // 处理 ret 指令
    void visitRetInstruction(const midend::Instruction* retInst) {
        // 处理返回指令
        if (retInst->getOpcode() == midend::Opcode::Ret) {
            // 检查是否有返回值
            if (retInst->getNumOperands() > 0) {
                // 处理返回值
                visit(retInst->getOperand(0));
            }
        }
    }

    MachineOperand visit(const midend::Value* value) {
        // 处理值的访问
        // 检查是否已经处理过该值
        const auto foundReg = codeGen_->getRegForValue(value);
    }

    // 访问常量
    void visit(const midend::Constant* constant) {}

    // 访问 global variable
    void visit(const midend::GlobalVariable* var) {
        // 其他操作...
    }

   private:
    CodeGenerator* codeGen_;
};

riscv64::CodeGenerator::CodeGenerator() : visitor_(std::make_unique<Visitor>(this)) {}

riscv64::CodeGenerator::~CodeGenerator() = default;
