#include "ConstantFoldingPass.h"

#include <cstdint>
#include <limits>
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
    // Only try to fold canonical 3-op R-type forms: rd, rs1, rs2
    if (inst->getOprandCount() != 3) {
        return;
    }

    auto opcode = inst->getOpcode();

    // Handle constant-first non-commutative comparisons by flipping predicate.
    // Pattern: SLT rd, imm, rs  ==>  SGT rd, rs, imm   (because imm < rs <=> rs
    // > imm) Keep it simple: only signed variant (i32) per requirement.
    // Unsigned left untouched.
    if (opcode == Opcode::SLT && inst->getOprandCount() == 3) {
        auto* op1 = inst->getOperand(1);
        auto* op2 = inst->getOperand(2);
        if (op1->isImm() && op2->isReg()) {
            auto* rd = inst->getOperand(0);
            if (rd->isReg()) {
                // Clone operands in new order: rd, rs(/*op2*/), imm(/*op1*/)
                auto rdClone =
                    Visitor::cloneRegister(dynamic_cast<RegisterOperand*>(rd));
                auto rsClone =
                    Visitor::cloneRegister(dynamic_cast<RegisterOperand*>(op2));
                auto immVal = op1->getValue();
                std::string original = inst->toString();
                inst->clearOperands();
                inst->setOpcode(Opcode::SGT);  // use pseudo greater-than
                inst->addOperand(std::move(rdClone));
                inst->addOperand(std::move(rsClone));
                inst->addOperand(std::make_unique<ImmediateOperand>(immVal));
                std::cout << "Flip predicate (const-first SLT): '" << original
                          << "' -> '" << inst->toString() << "'" << std::endl;
                // After rewriting, proceed with possible further peephole in
                // later passes (do not continue here to avoid double-processing
                // now)
            }
        }
    }

    // Mapping for direct R->I conversions (same semantics, rs2 -> imm)
    static const std::unordered_map<Opcode, Opcode> r2i = {
        {Opcode::ADD, Opcode::ADDI},   {Opcode::ADDW, Opcode::ADDIW},
        {Opcode::AND, Opcode::ANDI},   {Opcode::OR, Opcode::ORI},
        {Opcode::XOR, Opcode::XORI},   {Opcode::SLL, Opcode::SLLI},
        {Opcode::SLLW, Opcode::SLLIW}, {Opcode::SRA, Opcode::SRAI},
        {Opcode::SRAW, Opcode::SRAIW}, {Opcode::SRL, Opcode::SRLI},
        {Opcode::SRLW, Opcode::SRLIW}, {Opcode::SLT, Opcode::SLTI},
        {Opcode::SLTU, Opcode::SLTIU}, {Opcode::SUB, Opcode::ADDI},
        {Opcode::SUBW, Opcode::ADDIW}};  // SUB uses ADDI with negated imm

    auto it_map = r2i.find(opcode);
    if (it_map == r2i.end()) {
        return;  // Not a supported opcode
    }

    auto* rs1 = inst->getOperand(1);
    auto* rs2 = inst->getOperand(2);
    auto constOpRs1 = getConstant(*rs1);
    auto constOpRs2 = getConstant(*rs2);

    // Classification of op commutativity (with regard to source operands)
    auto is_commutative = [&](Opcode opcodeCandidate) {
        switch (opcodeCandidate) {
            case Opcode::ADD:
            case Opcode::ADDW:
            case Opcode::AND:
            case Opcode::OR:
            case Opcode::XOR:
                return true;  // safe to swap rs1/rs2
            default:
                return false;
        }
    };

    // For non-commutative ops (SUB, shifts, SLT/SLTU) we only proceed if rs2 is
    // constant. If rs1 is constant we abort (to avoid semantic inversion) -
    // KISS principle.
    if (!is_commutative(opcode)) {
        if (!constOpRs2.has_value()) {
            return;  // need rs2 const
        }
        if (!rs1->isReg()) {
            return;  // rs1 must stay a reg
        }
    } else {
        // Commutative: prefer rs2 const. If only rs1 const, swap logically.
        if (!constOpRs2.has_value() && constOpRs1.has_value()) {
            // We can transform: op rd, const, reg  ==> op (swap) rd, reg, const
            // Just treat as if rs2 const by swapping rs1/rs2 pointers and
            // constants.
            std::swap(rs1, rs2);
            std::swap(constOpRs1, constOpRs2);
        }
        if (!constOpRs2.has_value()) {
            return;  // still no immediate candidate
        }
        if (!rs1->isReg()) {
            return;  // rs1 must be register in I-type
        }
    }

    // Now rs2 is the constant to fold.
    auto immValue = constOpRs2.value();

    // Special case: SUB / SUBW become ADDI / ADDIW with negated immediate
    if (opcode == Opcode::SUB || opcode == Opcode::SUBW) {
        immValue = -immValue;
    }

    // Range check for 12-bit signed immediates
    constexpr int IMM12_MIN = -2048;
    constexpr int IMM12_MAX = 2047;
    if (immValue < IMM12_MIN || immValue > IMM12_MAX) {
        return;
    }

    // Clone destination & rs1 register
    auto* rd_orig = inst->getOperand(0);
    if (!rd_orig->isReg()) {
        return;  // defensive
    }
    auto rdClone =
        Visitor::cloneRegister(dynamic_cast<RegisterOperand*>(rd_orig));
    auto rs1Clone = Visitor::cloneRegister(dynamic_cast<RegisterOperand*>(rs1));

    auto newOpcode = it_map->second;
    std::string original = inst->toString();

    inst->clearOperands();
    inst->setOpcode(newOpcode);
    inst->addOperand(std::move(rdClone));
    inst->addOperand(std::move(rs1Clone));
    inst->addOperand(std::make_unique<ImmediateOperand>(immValue));

    std::cout << "Converted R-type to I-type instruction: '" << original
              << "' -> '" << inst->toString() << "'" << std::endl;
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
    // Only handle simple integer (i32) semantics for now.
    auto as_i32 = [](int64_t v) -> int32_t { return static_cast<int32_t>(v); };
    auto sign_extend_i32 = [&](int64_t v) -> int64_t {
        return static_cast<int64_t>(static_cast<int32_t>(v));
    };

    auto needN = [&](std::size_t n) -> bool {
        return source_operands.size() == n;
    };

    switch (op) {
        // Binary arithmetic (signed) wrap in 32-bit
        case Opcode::ADD:
        case Opcode::ADDW:
        case Opcode::ADDI:
        case Opcode::ADDIW: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = sign_extend_i32(static_cast<int64_t>(a) +
                                          static_cast<int64_t>(b));
            std::cout << "Calculate Inst value: " << a << " + " << b << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::SUB:
        case Opcode::SUBW: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = sign_extend_i32(static_cast<int64_t>(a) -
                                          static_cast<int64_t>(b));
            std::cout << "Calculate Inst value: " << a << " - " << b << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::MUL:
        case Opcode::MULW: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = sign_extend_i32(static_cast<int64_t>(a) *
                                          static_cast<int64_t>(b));
            std::cout << "Calculate Inst value: " << a << " * " << b << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::DIV:
        case Opcode::DIVW: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            if (b == 0) return std::nullopt;  // avoid div-by-zero fold
            if (a == std::numeric_limits<int32_t>::min() && b == -1)
                return std::nullopt;  // avoid overflow UB
            auto result = sign_extend_i32(static_cast<int64_t>(a / b));
            std::cout << "Calculate Inst value: " << a << " / " << b << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::DIVU:
        case Opcode::DIVUW: {
            if (!needN(2)) return std::nullopt;
            uint32_t a = static_cast<uint32_t>(as_i32(source_operands[0]));
            uint32_t b = static_cast<uint32_t>(as_i32(source_operands[1]));
            if (b == 0) return std::nullopt;
            auto result = sign_extend_i32(
                static_cast<int64_t>(static_cast<int32_t>(a / b)));
            std::cout << "Calculate Inst value: " << a << " /u " << b << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::REM:
        case Opcode::REMW: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            if (b == 0) return std::nullopt;
            if (a == std::numeric_limits<int32_t>::min() && b == -1) {
                auto result =
                    sign_extend_i32(0);  // per RISC-V spec rem of this is 0
                std::cout << "Calculate Inst value: " << a << " % " << b
                          << " (special) -> " << result << std::endl;
                return result;
            }
            auto result = sign_extend_i32(static_cast<int64_t>(a % b));
            std::cout << "Calculate Inst value: " << a << " % " << b << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::REMU:
        case Opcode::REMUW: {
            if (!needN(2)) return std::nullopt;
            uint32_t a = static_cast<uint32_t>(as_i32(source_operands[0]));
            uint32_t b = static_cast<uint32_t>(as_i32(source_operands[1]));
            if (b == 0) return std::nullopt;
            auto result = sign_extend_i32(
                static_cast<int64_t>(static_cast<int32_t>(a % b)));
            std::cout << "Calculate Inst value: " << a << " %u " << b << " -> "
                      << result << std::endl;
            return result;
        }

        // Bitwise / logic
        case Opcode::AND:
        case Opcode::ANDI: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = sign_extend_i32(a & b);
            std::cout << "Calculate Inst value: " << a << " & " << b << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::OR:
        case Opcode::ORI: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = sign_extend_i32(a | b);
            std::cout << "Calculate Inst value: " << a << " | " << b << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::XOR:
        case Opcode::XORI: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = sign_extend_i32(a ^ b);
            std::cout << "Calculate Inst value: " << a << " ^ " << b << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::SLL:
        case Opcode::SLLW:
        case Opcode::SLLI:
        case Opcode::SLLIW: {
            if (!needN(2)) return std::nullopt;
            uint32_t a = static_cast<uint32_t>(as_i32(source_operands[0]));
            uint32_t sh =
                static_cast<uint32_t>(as_i32(source_operands[1])) & 31u;
            auto result = sign_extend_i32(
                static_cast<int64_t>(static_cast<int32_t>(a << sh)));
            std::cout << "Calculate Inst value: " << a << " << " << sh << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::SRL:
        case Opcode::SRLW:
        case Opcode::SRLI:
        case Opcode::SRLIW: {
            if (!needN(2)) return std::nullopt;
            uint32_t a = static_cast<uint32_t>(as_i32(source_operands[0]));
            uint32_t sh =
                static_cast<uint32_t>(as_i32(source_operands[1])) & 31u;
            auto result = sign_extend_i32(
                static_cast<int64_t>(static_cast<int32_t>(a >> sh)));
            std::cout << "Calculate Inst value: " << a << " >>u " << sh
                      << " -> " << result << std::endl;
            return result;
        }
        case Opcode::SRA:
        case Opcode::SRAW:
        case Opcode::SRAI:
        case Opcode::SRAIW: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            uint32_t sh =
                static_cast<uint32_t>(as_i32(source_operands[1])) & 31u;
            auto result = sign_extend_i32(static_cast<int64_t>(a >> sh));
            std::cout << "Calculate Inst value: " << a << " >> " << sh << " -> "
                      << result << std::endl;
            return result;
        }

        // Comparisons (return 0/1)
        case Opcode::SLT:
        case Opcode::SLTI: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = static_cast<int64_t>(a < b ? 1 : 0);
            std::cout << "Calculate Inst value: " << a << " < " << b << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::SGT: {  // pseudo >
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = static_cast<int64_t>(a > b ? 1 : 0);
            std::cout << "Calculate Inst value: " << a << " > " << b << " -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::SLTU:
        case Opcode::SLTIU: {
            if (!needN(2)) return std::nullopt;
            uint32_t a = static_cast<uint32_t>(as_i32(source_operands[0]));
            uint32_t b = static_cast<uint32_t>(as_i32(source_operands[1]));
            auto result = static_cast<int64_t>(a < b ? 1 : 0);
            std::cout << "Calculate Inst value: " << a << " <u " << b << " -> "
                      << result << std::endl;
            return result;
        }

        // Unary pseudos
        case Opcode::NEG:
        case Opcode::NEGW: {
            if (!needN(1)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            // Avoid overflow case -INT32_MIN (leave for later lowering)
            if (a == std::numeric_limits<int32_t>::min()) return std::nullopt;
            auto result = sign_extend_i32(-a);
            std::cout << "Calculate Inst value: -" << a << " -> " << result
                      << std::endl;
            return result;
        }
        case Opcode::NOT: {
            if (!needN(1)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            auto result = sign_extend_i32(~a);
            std::cout << "Calculate Inst value: ~" << a << " -> " << result
                      << std::endl;
            return result;
        }
        case Opcode::SEQZ: {
            if (!needN(1)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            auto result = static_cast<int64_t>(a == 0 ? 1 : 0);
            std::cout << "Calculate Inst value: (" << a << " == 0) -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::SNEZ: {
            if (!needN(1)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            auto result = static_cast<int64_t>(a != 0 ? 1 : 0);
            std::cout << "Calculate Inst value: (" << a << " != 0) -> "
                      << result << std::endl;
            return result;
        }
        case Opcode::SLTZ: {
            if (!needN(1)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            auto result = static_cast<int64_t>(a < 0 ? 1 : 0);
            std::cout << "Calculate Inst value: (" << a << " < 0) -> " << result
                      << std::endl;
            return result;
        }
        case Opcode::SGTZ: {
            if (!needN(1)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            auto result = static_cast<int64_t>(a > 0 ? 1 : 0);
            std::cout << "Calculate Inst value: (" << a << " > 0) -> " << result
                      << std::endl;
            return result;
        }

        default:
            return std::nullopt;  // Not (yet) supported
    }
}

}  // namespace riscv64