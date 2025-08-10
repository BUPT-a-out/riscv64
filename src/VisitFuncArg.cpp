#include "CodeGen.h"
#include "IR/BasicBlock.h"
#include "IR/Function.h"
#include "Visit.h"

namespace riscv64 {

void riscv64::Visitor::preProcessFuncArgs(const midend::Function* midend_func,
                                          midend::BasicBlock* midend_bb,
                                          Function* riscv_func) {
    auto* first_riscv_bb = riscv_func->getBasicBlock(midend_bb);
    if (first_riscv_bb != nullptr) {
        // 1. 先处理前8个参数（寄存器传递）
        int arg_idx = 0;
        for (auto arg_it = midend_func->arg_begin();
             arg_it != midend_func->arg_end(); ++arg_it, ++arg_idx) {
            if (arg_idx >= 8) break;
            const auto* argument = arg_it->get();
            bool is_float_arg = argument->getType()->isFloatType();
            std::unique_ptr<RegisterOperand> new_reg =
                codeGen_->allocateReg(is_float_arg);
            codeGen_->mapValueToReg(argument, new_reg->getRegNum(),
                                    new_reg->isVirtual());
            auto source_reg = funcArgToReg(argument, first_riscv_bb);
            auto dest_reg = cloneRegister(new_reg.get(), is_float_arg);
            if (is_float_arg) {
                auto* source_reg_operand =
                    dynamic_cast<RegisterOperand*>(source_reg.get());
                if ((source_reg_operand != nullptr) &&
                    source_reg_operand->isFloatRegister()) {
                    auto fmov_inst = std::make_unique<Instruction>(
                        Opcode::FMOV_S, first_riscv_bb);
                    fmov_inst->addOperand(std::move(dest_reg));
                    fmov_inst->addOperand(std::move(source_reg));
                    first_riscv_bb->addInstruction(std::move(fmov_inst));
                } else {
                    storeOperandToReg(std::move(source_reg),
                                      std::move(dest_reg), first_riscv_bb);
                }
            } else {
                storeOperandToReg(std::move(source_reg), std::move(dest_reg),
                                  first_riscv_bb);
            }
        }
        // 2.
        // 栈参数（第9个及以后）延迟处理：不在此处发射加载指令，改为在首次使用时再加载
        //    这样可以保证在入口处先完成对寄存器参数的保存，再访问栈参数，避免顺序不一致。
    }
}

}  // namespace riscv64