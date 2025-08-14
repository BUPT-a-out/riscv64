#include "ConstantFoldingPass.h"

#include "Visit.h"

namespace riscv64 {

void ConstantFolding::runOnFunction(Function* function) {
    // Perform constant folding on the given function
}

void ConstantFolding::runOnBasicBlock(BasicBlock* basicBlock) {
    // Perform constant folding on the given basic block
    // init
    virtualRegisterConstants.clear();
    instructionsToRemove.clear();

    for (auto& inst : *basicBlock) {
        handleInstruction(inst.get(), basicBlock);
    }

    for (auto* inst : instructionsToRemove) {
        // Remove the instruction from the basic block
        basicBlock->removeInstruction(inst);
        std::cout << "Removed instruction: " << inst->toString() << std::endl;
    }
}

void ConstantFolding::handleInstruction(Instruction* inst,
                                        BasicBlock* parent_bb) {
    // 处理重定义
    if (inst->getOprandCount() >= 1) {
        auto* defined_operand = inst->getOperand(0);
        if (defined_operand->isReg()) {
            // 旧值失效
            virtualRegisterConstants.erase(defined_operand->getRegNum());
        }
    }

    // 尝试折叠
    foldInstruction(inst, parent_bb);
    // 尝试窥孔优化
    peepholeOptimize(inst, parent_bb);
    // 尝试常量传播
    constantPropagate(inst, parent_bb);
}

void ConstantFolding::foldInstruction(Instruction* inst,
                                      BasicBlock* parent_bb) {
    std::vector<int64_t> source_constants;
    for (const auto& operand : inst->getOperands()) {
        if (operand->isImm()) {
            source_constants.push_back(operand->getValue());
        } else if (operand->isReg()) {
            auto it = virtualRegisterConstants.find(operand->getRegNum());
            if (it != virtualRegisterConstants.end()) {
                source_constants.push_back(it->second);
            } else {
                return;  // 无法折叠
            }
        } else {
            return;  // 不支持的操作数类型
        }
    }

    // 那么，可以被折叠
    auto result =
        calculateInstructionValue(inst->getOpcode(), source_constants);
    if (!result.has_value()) {
        return;  // 算不出来
    }

    auto* dest_reg = inst->getOperand(0);
    virtualRegisterConstants[dest_reg->getRegNum()] = result.value();

    // 替换指令
    auto inst_backup = *inst;
    inst->clearOperands();
    inst->setOpcode(Opcode::LI);
    inst->addOperand(std::make_unique<ImmediateOperand>(result.value()));

    std::cout << "Folded instruction: '" << inst_backup.toString() << "' to '"
              << inst->toString() << "'" << std::endl;
}

void ConstantFolding::peepholeOptimize(Instruction* inst,
                                       BasicBlock* parent_bb) {
    foldToITypeInst(inst, parent_bb);
    algebraicIdentitySimplify(inst, parent_bb);
    strengthReduction(inst, parent_bb);
    bitwiseOperationSimplify(inst, parent_bb);
    instructionReassociateAndCombine(inst, parent_bb);
}

void ConstantFolding::foldToITypeInst(Instruction* inst,
                                      BasicBlock* parent_bb) {
    ;
}

void ConstantFolding::algebraicIdentitySimplify(Instruction* inst,
                                                BasicBlock* parent_bb) {
    ;
}

void ConstantFolding::strengthReduction(Instruction* inst,
                                        BasicBlock* parent_bb) {
    ;
}

void ConstantFolding::bitwiseOperationSimplify(Instruction* inst,
                                               BasicBlock* parent_bb) {
    ;
}

void ConstantFolding::instructionReassociateAndCombine(Instruction* inst,
                                                       BasicBlock* parent_bb) {
    ;
}

void ConstantFolding::constantPropagate(Instruction* inst,
                                        BasicBlock* parent_bb) {
    ;
}

std::optional<int64_t> ConstantFolding::calculateInstructionValue(
    Opcode op, std::vector<int64_t>& source_operands) {
    // TODO(rikka): impl
    return 114514;
}

}  // namespace riscv64