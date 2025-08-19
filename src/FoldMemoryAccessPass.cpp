#include "Instructions/All.h"
#include "Target.h"
#include "Visit.h"

namespace riscv64 {

void handleInstruction(Instruction* inst, BasicBlock* parent_bb) {
    // if (inst->getOprandCount() == 0) {
    //     return;  // No operands to process
    // }

    // if (inst->getOprandCount() == 2 &&
    //     inst->getOperand(1)->getType() == OperandType::Memory) {
    //     auto* memory_op = dynamic_cast<MemoryOperand*>(inst->getOperand(1));
    //     if (memory_op == nullptr) {
    //         return;
    //     }
    //     auto* base_reg =
    //         dynamic_cast<RegisterOperand*>(memory_op->getBaseReg());
    //     auto* offset_imm =
    //         dynamic_cast<ImmediateOperand*>(memory_op->getOffset());
    //     if (base_reg == nullptr || offset_imm == nullptr) {
    //         return;
    //     }

    //     auto* base_reg_def = parent_bb->getIntVRegDef(base_reg->getRegNum());
    //     if (base_reg_def == nullptr) {
    //         return;
    //     }

    //     if (base_reg_def->getOpcode() != ADDI &&
    //         base_reg_def->getOpcode() != ADDIW) {
    //         return;
    //     }

    //     auto* base_reg_def_src_op = base_reg_def->getOperand(1);
    //     auto* base_reg_def_imm =
    //         dynamic_cast<ImmediateOperand*>(base_reg_def->getOperand(2));
    //     if (!base_reg_def_src_op->isReg() || base_reg_def_imm == nullptr) {
    //         return;
    //     }
    //     auto new_offset_val =
    //         offset_imm->getIntValue() + base_reg_def_imm->getIntValue();

    //     if (Visitor::isValidImmediateOffset(new_offset_val)) {
    //         auto old_inst_string = inst->toString();
    //         // 合并成一个新的指令
    //         inst->clearOperands();
    //         // inst->setOpcode(inst->getOpcode() == LOAD ? LOAD : STORE);
    //         inst->addOperand_(Visitor::cloneRegister(
    //             dynamic_cast<RegisterOperand*>(base_reg_def_src_op)));
    //         auto new_imm_operand =
    //             std::make_unique<ImmediateOperand>(new_offset_val);
    //         auto new_mem_operand = std::make_unique<MemoryOperand>(
    //             Visitor::cloneRegister(base_reg), std::move(new_imm_operand));

    //         DEBUG_OUT() << "fold memory access: '" << old_inst_string
    //                     << "' -> '" << inst->toString() << "'" << std::endl;
    //     }
    // }
}

Module& RISCV64Target::foldMemoryAccessPass(riscv64::Module& module) {
    for (auto& func : module) {
        if (func->empty()) {
            continue;  // Skip empty functions
        }
        for (auto& bb : *func) {
            DEBUG_OUT() << "Running FoldMemoryAccessPass on basic block: "
                        << bb->getLabel() << std::endl;
            for (auto& inst : *bb) {
                // 处理每条指令
                handleInstruction(inst.get(), bb.get());
            }
        }
    }
    return module;
}

}  // namespace riscv64