#include "ConstantFoldingPass.h"

#include <string>

#include "Visit.h"

namespace riscv64 {

void ConstantFolding::runOnFunction(Function* function) {
    // Perform constant folding on the given function
    for (auto& bb : *function) {
        runOnBasicBlock(bb.get());
    }
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

    // 若最终形态为 LI，确保记录常量 (放在最后避免被后续修改覆盖)
    if (inst->getOpcode() == Opcode::LI && inst->getOprandCount() == 2) {
        auto* rd = inst->getOperand(0);
        auto* imm = inst->getOperand(1);
        if (rd && imm && rd->isReg() && imm->isImm()) {
            virtualRegisterConstants[rd->getRegNum()] = imm->getValue();
        }
    }
}

std::optional<int64_t> ConstantFolding::getConstant(MachineOperand& operand) {
    if (operand.isImm()) {
        return operand.getValue();
    }
    if (operand.isReg()) {
        auto it = virtualRegisterConstants.find(operand.getRegNum());
        if (it != virtualRegisterConstants.end()) {
            return it->second;
        }
    }
    return std::nullopt;  // 不是常量
}

void ConstantFolding::foldInstruction(Instruction* inst,
                                      BasicBlock* parent_bb) {
    // 目前假定第0个操作数是目的寄存器，后续的是源操作数
    if (inst->getOprandCount() <= 1) {
        return;  // 没有源操作数，无需折叠
    }

    if (inst->getOpcode() == Opcode::LI) {
        virtualRegisterConstants[inst->getOperand(0)->getRegNum()] =
            inst->getOperand(1)->getValue();
        return;
    }

    std::vector<int64_t> source_constants;
    for (size_t i = 1; i < inst->getOprandCount(); ++i) {
        auto* operand = inst->getOperand(i);
        auto operand_constant = getConstant(*operand);
        if (!operand_constant.has_value()) {
            return;  // 任一源不是常量，无法折叠
        }
        source_constants.push_back(operand_constant.value());
    }

    auto result =
        calculateInstructionValue(inst->getOpcode(), source_constants);
    if (!result.has_value()) {
        return;  // 运算不支持
    }

    auto* dest_reg_operand = inst->getOperand(0);
    if (!dest_reg_operand->isReg()) {
        return;  // 目的不是寄存器，不处理
    }
    virtualRegisterConstants[dest_reg_operand->getRegNum()] = result.value();

    // 记录原始字符串用于日志
    std::string original = inst->toString();

    // 克隆目的寄存器
    auto* dest_reg_raw = dynamic_cast<RegisterOperand*>(dest_reg_operand);
    auto dest_clone = Visitor::cloneRegister(dest_reg_raw);

    // 重写指令为 LI rd, imm
    inst->clearOperands();
    inst->setOpcode(Opcode::LI);
    inst->addOperand(std::move(dest_clone));
    inst->addOperand(std::make_unique<ImmediateOperand>(result.value()));

    std::cout << "Folded instruction: '" << original << "' to '"
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
    switch (inst->getOpcode()) {
        case Opcode::ADD:
        case Opcode::ADDW:
        case Opcode::AND:
        case Opcode::OR:
        case Opcode::XOR:
        case Opcode::SLL:
        case Opcode::SLLW:
        case Opcode::SRA:
        case Opcode::SRAW:
        case Opcode::SRL:
        case Opcode::SRLW:
        case Opcode::SLT: {
            static const std::unordered_map<Opcode, Opcode> rTypeToITypeMap = {
                {Opcode::ADD, Opcode::ADDI},   {Opcode::ADDW, Opcode::ADDIW},
                {Opcode::AND, Opcode::ANDI},   {Opcode::OR, Opcode::ORI},
                {Opcode::XOR, Opcode::XORI},   {Opcode::SLL, Opcode::SLLI},
                {Opcode::SLLW, Opcode::SLLIW}, {Opcode::SRA, Opcode::SRAI},
                {Opcode::SRAW, Opcode::SRAIW}, {Opcode::SRL, Opcode::SRLI},
                {Opcode::SRLW, Opcode::SRLIW}, {Opcode::SLT, Opcode::SLTI},
                {Opcode::SUB, Opcode::ADDI},   {Opcode::SUBW, Opcode::ADDIW}};

            // Check if any source operand is a constant
            auto* operand1 = inst->getOperand(1);  // First source operand
            auto* operand2 = inst->getOperand(2);  // Second source operand

            auto const1 = getConstant(*operand1);
            auto const2 = getConstant(*operand2);

            // If either operand is constant, convert to I-type instruction
            if (const1.has_value() || const2.has_value()) {
                auto opcode = inst->getOpcode();
                auto it = rTypeToITypeMap.find(opcode);
                if (it != rTypeToITypeMap.end()) {
                    Opcode newOpcode = it->second;

                    // Determine which operand is constant and which is the
                    // register
                    RegisterOperand* regOperand;
                    int64_t immValue;

                    if (const2.has_value()) {
                        // Second operand is constant, keep first operand as
                        // register
                        regOperand = dynamic_cast<RegisterOperand*>(operand1);
                        immValue = const2.value();
                    } else {
                        // First operand is constant, keep second operand as
                        // register
                        regOperand = dynamic_cast<RegisterOperand*>(operand2);
                        immValue = const1.value();
                    }

                    if (opcode == Opcode::SUB || opcode == Opcode::SUBW) {
                        // 对于减法指令，立即数需要取反
                        immValue = -immValue;
                    }

                    // Check if immediate value is within valid range for I-type
                    // instructions RISC-V I-type instructions have 12-bit
                    // signed immediate field
                    if (immValue >= -2048 && immValue <= 2047) {
                        // Store the destination operand
                        auto destOperand = Visitor::cloneRegister(
                            dynamic_cast<RegisterOperand*>(
                                inst->getOperand(0)));
                        auto regOperandClone =
                            Visitor::cloneRegister(regOperand);

                        // Clear existing operands and set new opcode
                        inst->clearOperands();
                        inst->setOpcode(newOpcode);

                        // Add operands: destination, register, immediate
                        inst->addOperand(std::move(destOperand));
                        inst->addOperand(std::move(regOperandClone));
                        inst->addOperand(
                            std::make_unique<ImmediateOperand>(immValue));

                        std::cout << "Converted R-type to I-type instruction: "
                                  << inst->toString() << std::endl;
                    }
                }
            }
        } break;

        default:
            return;
    }
}

void ConstantFolding::algebraicIdentitySimplify(Instruction* inst,
                                                BasicBlock* parent_bb) {
    // Only handle simple 3-operand integer-like instructions: rd, rs1, rs2
    if (inst->getOprandCount() != 3) {
        return;
    }

    auto opcode = inst->getOpcode();
    auto* destOperand = inst->getOperand(0);
    auto* srcOperand1 = inst->getOperand(1);
    auto* srcOperand2 = inst->getOperand(2);

    if (!destOperand->isReg()) {
        return;  // We only care when we define a register
    }

    auto* destReg = dynamic_cast<RegisterOperand*>(destOperand);

    auto constOp1 = getConstant(*srcOperand1);
    auto constOp2 = getConstant(*srcOperand2);

    auto cloneReg = [](MachineOperand* operand) {
        return Visitor::cloneRegister(dynamic_cast<RegisterOperand*>(operand));
    };

    auto emitMovFrom = [&](MachineOperand* sourceRegOperand) {
        // ADDI rd, rs, 0  (canonical MOV form we use elsewhere)
        if (!sourceRegOperand->isReg()) {
            return;  // Defensive: patterns ensure reg, but guard anyway
        }
        std::string original = inst->toString();
        unsigned destRegNum = destReg->getRegNum();
        unsigned srcRegNum = sourceRegOperand->getRegNum();
        auto destClone = Visitor::cloneRegister(destReg);  // clone before clear
        auto srcClone = cloneReg(sourceRegOperand);        // clone before clear
        inst->clearOperands();
        inst->setOpcode(Opcode::ADDI);
        inst->addOperand(std::move(destClone));
        inst->addOperand(std::move(srcClone));
        inst->addOperand(std::make_unique<ImmediateOperand>(0));
        // Constant propagation update
        auto itConst = virtualRegisterConstants.find(srcRegNum);
        if (itConst != virtualRegisterConstants.end()) {
            virtualRegisterConstants[destRegNum] = itConst->second;
        } else {
            virtualRegisterConstants.erase(destRegNum);
        }
        std::cout << "Algebraic simplify: '" << original << "' -> '"
                  << inst->toString() << "'" << std::endl;
    };

    auto emitLiZero = [&]() {
        std::string original = inst->toString();
        unsigned destRegNum = destReg->getRegNum();
        auto destClone = Visitor::cloneRegister(destReg);
        inst->clearOperands();
        inst->setOpcode(Opcode::LI);
        inst->addOperand(std::move(destClone));
        inst->addOperand(std::make_unique<ImmediateOperand>(0));
        virtualRegisterConstants[destRegNum] = 0;
        std::cout << "Algebraic simplify: '" << original << "' -> '"
                  << inst->toString() << "'" << std::endl;
    };

    switch (opcode) {
        case Opcode::ADD:
        case Opcode::ADDW: {  // Pattern 1.1
            if (constOp1 && *constOp1 == 0 && srcOperand2->isReg()) {
                emitMovFrom(srcOperand2);
            } else if (constOp2 && *constOp2 == 0 && srcOperand1->isReg()) {
                emitMovFrom(srcOperand1);
            }
            break;
        }
        case Opcode::SUB:     // Pattern 1.2 & 1.3
        case Opcode::SUBW: {  // 32-bit variant
            if (srcOperand1->isReg() && srcOperand2->isReg() &&
                srcOperand1->getRegNum() == srcOperand2->getRegNum()) {
                // x - x
                emitLiZero();  // Pattern 1.3
            } else if (constOp2 && *constOp2 == 0 && srcOperand1->isReg()) {
                // x - 0 = x (Pattern 1.2)
                emitMovFrom(srcOperand1);
            }
            break;
        }
        case Opcode::MUL:
        case Opcode::MULW: {  // Pattern 1.4 & 1.5 (only handle MUL variant
                              // here)
            if ((constOp1 && *constOp1 == 0) || (constOp2 && *constOp2 == 0)) {
                emitLiZero();  // x * 0 = 0 (Pattern 1.5)
            } else if (constOp1 && *constOp1 == 1 && srcOperand2->isReg()) {
                emitMovFrom(srcOperand2);  // 1 * x = x (Pattern 1.4)
            } else if (constOp2 && *constOp2 == 1 && srcOperand1->isReg()) {
                emitMovFrom(srcOperand1);  // x * 1 = x (Pattern 1.4)
            }
            break;
        }
        case Opcode::DIV:      // Pattern 1.6
        case Opcode::DIVU:     // Unsigned variant
        case Opcode::DIVW:     // 32-bit signed
        case Opcode::DIVUW: {  // 32-bit unsigned
            if (constOp2 && *constOp2 == 1 && srcOperand1->isReg()) {
                emitMovFrom(srcOperand1);  // x / 1 = x
            }
            break;
        }
        default:
            break;  // other opcodes not handled here
    }
}

void ConstantFolding::strengthReduction(Instruction* inst,
                                        BasicBlock* parent_bb) {
    (void)inst;
    (void)parent_bb;
}

void ConstantFolding::bitwiseOperationSimplify(Instruction* inst,
                                               BasicBlock* parent_bb) {
    (void)inst;
    (void)parent_bb;
}

void ConstantFolding::instructionReassociateAndCombine(Instruction* inst,
                                                       BasicBlock* parent_bb) {
    (void)inst;
    (void)parent_bb;
}

void ConstantFolding::constantPropagate(Instruction* inst,
                                        BasicBlock* parent_bb) {
    (void)inst;
    (void)parent_bb;
}

std::optional<int64_t> ConstantFolding::calculateInstructionValue(
    Opcode op, std::vector<int64_t>& source_operands) {
    // TODO(rikka): impl
    (void)op;
    (void)source_operands;
    return 114514;
}

}  // namespace riscv64