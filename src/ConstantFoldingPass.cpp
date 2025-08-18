#include "ConstantFoldingPass.h"

#include <cstdint>
#include <limits>
#include <string>

#include "Visit.h"

// Debug output macro - only outputs when A_OUT_DEBUG is defined
#ifdef A_OUT_DEBUG
#define DEBUG_OUT() std::cout
#else
#define DEBUG_OUT() \
    if constexpr (false) std::cout
#endif

namespace riscv64 {

void ConstantFolding::runOnFunction(Function* function) {
    // Perform constant folding on the given function
    for (auto& bb : *function) {
        for (int i = 0; i < 3; i++) {
            DEBUG_OUT() << "(" << i + 1
                        << " / 3) Running constant folding on basic block: "
                        << bb->getLabel() << std::endl;
            runOnBasicBlock(bb.get());
        }
    }
}

void ConstantFolding::runOnBasicBlock(BasicBlock* basicBlock) {
    // Perform constant folding on the given basic block
    // init
    virtualRegisterConstants.clear();
    instructionsToRemove.clear();

    // init: x0 -> 0
    mapRegToConstant(0, 0);

    for (auto& inst : *basicBlock) {
        handleInstruction(inst.get(), basicBlock);
    }

    for (auto* inst : instructionsToRemove) {
        // Remove the instruction from the basic block
        basicBlock->removeInstruction(inst);
        DEBUG_OUT() << "Removed instruction: " << inst->toString() << std::endl;
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
            mapRegToConstant(rd->getRegNum(), imm->getValue());
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
        mapRegToConstant(inst->getOperand(0)->getRegNum(),
                         inst->getOperand(1)->getValue());
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
    mapRegToConstant(dest_reg_operand->getRegNum(), result.value());

    // 记录原始字符串用于日志
    std::string original = inst->toString();

    // 克隆目的寄存器
    auto* dest_reg_raw = dynamic_cast<RegisterOperand*>(dest_reg_operand);
    auto dest_clone = Visitor::cloneRegister(dest_reg_raw);

    // 重写指令为 LI rd, imm
    inst->clearOperands();
    inst->setOpcode(Opcode::LI);
    inst->addOperand_(std::move(dest_clone));
    inst->addOperand_(std::make_unique<ImmediateOperand>(result.value()));

    DEBUG_OUT() << "Folded instruction: '" << original << "' to '"
                << inst->toString() << "'" << std::endl;
}

void ConstantFolding::peepholeOptimize(Instruction* inst,
                                       BasicBlock* parent_bb) {
    for (int i = 0; i < 3; i++) {
        foldToITypeInst(inst, parent_bb);
        algebraicIdentitySimplify(inst, parent_bb);
        strengthReduction(inst, parent_bb);
        bitwiseOperationSimplify(inst, parent_bb);
        // mvToAddiw(inst, parent_bb);
        instructionReassociateAndCombine(inst, parent_bb);
        useZeroReg(inst, parent_bb);
    }
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
                inst->addOperand_(std::move(rdClone));
                inst->addOperand_(std::move(rsClone));
                inst->addOperand_(std::make_unique<ImmediateOperand>(immVal));
                DEBUG_OUT() << "Flip predicate (const-first SLT): '" << original
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
    inst->addOperand_(std::move(rdClone));
    inst->addOperand_(std::move(rs1Clone));
    inst->addOperand_(std::make_unique<ImmediateOperand>(immValue));

    DEBUG_OUT() << "Converted R-type to I-type instruction: '" << original
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
        inst->addOperand_(std::move(destClone));
        inst->addOperand_(std::move(srcClone));
        inst->addOperand_(std::make_unique<ImmediateOperand>(0));
        // Constant propagation update
        auto itConst = virtualRegisterConstants.find(srcRegNum);
        if (itConst != virtualRegisterConstants.end()) {
            mapRegToConstant(destRegNum, itConst->second);
        } else {
            virtualRegisterConstants.erase(destRegNum);
        }
        DEBUG_OUT() << "Algebraic simplify: '" << original << "' -> '"
                    << inst->toString() << "'" << std::endl;
    };

    auto emitLiZero = [&]() {
        std::string original = inst->toString();
        unsigned destRegNum = destReg->getRegNum();
        auto destClone = Visitor::cloneRegister(destReg);
        inst->clearOperands();
        inst->setOpcode(Opcode::LI);
        inst->addOperand_(std::move(destClone));
        inst->addOperand_(std::make_unique<ImmediateOperand>(0));
        mapRegToConstant(destRegNum, 0);
        DEBUG_OUT() << "Algebraic simplify: '" << original << "' -> '"
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
    (void)parent_bb;
    // Pattern: mul/div by power-of-two constant -> shift
    // Support opcodes: MUL, MULW, DIVU, DIVUW, DIV, DIVW (signed div only if
    // dividend known non-negative)
    if (inst->getOprandCount() != 3) {
        return;  // rd, rs1, rs2
    }

    const auto opcode = inst->getOpcode();
    auto* destReg = inst->getOperand(0);
    auto* srcReg1 = inst->getOperand(1);
    auto* srcReg2 = inst->getOperand(2);
    if (!destReg->isReg()) {
        return;
    }

    auto getConstValue =
        [&](MachineOperand* operand) -> std::optional<int64_t> {
        return getConstant(*operand);
    };
    const auto constVal1 = getConstValue(srcReg1);
    const auto constVal2 = getConstValue(srcReg2);

    auto isPowerOfTwo = [](int64_t value) -> bool {
        if (value <= 0) {
            return false;
        }
        uint64_t u = static_cast<uint64_t>(value);
        return (u & (u - 1ULL)) == 0ULL;
    };
    auto log2Exact = [](int64_t value) -> unsigned {
        unsigned shiftAmount = 0U;
        uint64_t u = static_cast<uint64_t>(value);
        while ((1ULL << shiftAmount) < u) {
            ++shiftAmount;
        }
        return shiftAmount;
    };

    auto cloneRegOperand = [](MachineOperand* operand) {
        return Visitor::cloneRegister(dynamic_cast<RegisterOperand*>(operand));
    };

    auto applyShiftImm = [&](Opcode newOpcode, MachineOperand* fromReg,
                             unsigned shiftAmount) {
        std::string original = inst->toString();
        unsigned destRegNum = destReg->getRegNum();
        unsigned sourceRegNum = fromReg->getRegNum();
        auto destClone = cloneRegOperand(destReg);
        auto sourceClone = cloneRegOperand(fromReg);
        inst->clearOperands();
        inst->setOpcode(newOpcode);
        inst->addOperand_(std::move(destClone));
        inst->addOperand_(std::move(sourceClone));
        inst->addOperand_(std::make_unique<ImmediateOperand>(
            static_cast<int64_t>(shiftAmount)));
        // Constant propagation update
        auto itConst = virtualRegisterConstants.find(sourceRegNum);
        if (itConst != virtualRegisterConstants.end()) {
            int64_t srcVal = itConst->second;
            int64_t newVal = 0;
            switch (newOpcode) {
                case Opcode::SLLI:
                case Opcode::SLLIW:
                    newVal = static_cast<int64_t>(static_cast<int32_t>(
                        static_cast<int32_t>(srcVal) << shiftAmount));
                    break;
                case Opcode::SRLI:
                case Opcode::SRLIW:
                    newVal = static_cast<int64_t>(static_cast<int32_t>(
                        static_cast<uint32_t>(static_cast<int32_t>(srcVal)) >>
                        shiftAmount));
                    break;
                case Opcode::SRAI:
                case Opcode::SRAIW:
                    newVal = static_cast<int64_t>(static_cast<int32_t>(
                        static_cast<int32_t>(srcVal) >> shiftAmount));
                    break;
                default:
                    break;
            }
            mapRegToConstant(destRegNum, newVal);
        } else {
            virtualRegisterConstants.erase(destRegNum);
        }
        DEBUG_OUT() << "Strength reduction: '" << original << "' -> '"
                    << inst->toString() << "'" << std::endl;
    };

    constexpr unsigned SHIFT_WIDTH_32 = 32U;  // 5-bit shift amount range
    auto shiftFits = [&](unsigned shiftAmount, bool isWord) -> bool {
        if (isWord) {
            return shiftAmount < SHIFT_WIDTH_32;
        }
        return shiftAmount <
               SHIFT_WIDTH_32;  // Currently treating as 32-bit ops
    };

    auto tryReduceMul = [&]() {
        bool isWord = (opcode == Opcode::MULW);
        MachineOperand* sourceRegForShift = nullptr;
        std::optional<int64_t> constFactor;
        if (constVal1 && isPowerOfTwo(*constVal1) && srcReg2->isReg()) {
            sourceRegForShift = srcReg2;
            constFactor = constVal1;
        } else if (constVal2 && isPowerOfTwo(*constVal2) && srcReg1->isReg()) {
            sourceRegForShift = srcReg1;
            constFactor = constVal2;
        } else {
            return false;
        }
        if (*constFactor == 1) {
            return false;  // handled elsewhere
        }
        unsigned shiftAmount = log2Exact(*constFactor);
        if (!shiftFits(shiftAmount, isWord)) {
            return false;
        }
        applyShiftImm(isWord ? Opcode::SLLIW : Opcode::SLLI, sourceRegForShift,
                      shiftAmount);
        return true;
    };

    auto tryReduceDivU = [&]() {
        bool isWord = (opcode == Opcode::DIVUW);
        if (!constVal2 || !isPowerOfTwo(*constVal2) || *constVal2 == 1) {
            return false;
        }
        if (!srcReg1->isReg()) {
            return false;
        }
        unsigned shiftAmount = log2Exact(*constVal2);
        if (!shiftFits(shiftAmount, isWord)) {
            return false;
        }
        applyShiftImm(isWord ? Opcode::SRLIW : Opcode::SRLI, srcReg1,
                      shiftAmount);
        return true;
    };

    auto tryReduceDivSigned = [&]() {
        bool isWord = (opcode == Opcode::DIVW);
        if (!constVal2 || !isPowerOfTwo(*constVal2) || *constVal2 == 1) {
            return false;
        }
        // Only when dividend known non-negative (conservative)
        if (!(constVal1 && *constVal1 >= 0)) {
            return false;
        }
        if (!srcReg1->isReg()) {
            return false;
        }
        unsigned shiftAmount = log2Exact(*constVal2);
        if (!shiftFits(shiftAmount, isWord)) {
            return false;
        }
        applyShiftImm(isWord ? Opcode::SRAIW : Opcode::SRAI, srcReg1,
                      shiftAmount);
        return true;
    };

    switch (opcode) {
        case Opcode::MUL:
        case Opcode::MULW:
            (void)tryReduceMul();
            break;
        case Opcode::DIVU:
        case Opcode::DIVUW:
            (void)tryReduceDivU();
            break;
        case Opcode::DIV:
        case Opcode::DIVW:
            (void)tryReduceDivSigned();
            break;
        default:
            break;  // not handled
    }
}

void ConstantFolding::bitwiseOperationSimplify(Instruction* inst,
                                               BasicBlock* parent_bb) {
    if (inst->getOprandCount() != 3) {
        return;  // 只处理三操作数形式: rd, rs1, rs2
    }

    auto opcode = inst->getOpcode();
    if (opcode != Opcode::AND && opcode != Opcode::OR &&
        opcode != Opcode::XOR && opcode != Opcode::ANDI &&
        opcode != Opcode::ORI && opcode != Opcode::XORI) {
        return;
    }

    auto* rd = inst->getOperand(0);
    auto* op1 = inst->getOperand(1);
    auto* op2 = inst->getOperand(2);
    if (!rd->isReg()) return;

    auto cloneReg = [](MachineOperand* r) {
        return Visitor::cloneRegister(dynamic_cast<RegisterOperand*>(r));
    };

    auto emitLi = [&](int32_t imm) {
        std::string original = inst->toString();
        unsigned rdNum = rd->getRegNum();
        auto rdClone = cloneReg(rd);
        inst->clearOperands();
        inst->setOpcode(Opcode::LI);
        inst->addOperand_(std::move(rdClone));
        inst->addOperand_(
            std::make_unique<ImmediateOperand>(static_cast<int64_t>(imm)));
        mapRegToConstant(rdNum, imm);
        DEBUG_OUT() << "Bitwise simplify: '" << original << "' -> '"
                    << inst->toString() << "'" << std::endl;
    };

    auto emitMovFrom = [&](MachineOperand* src) {
        if (!src->isReg()) return;
        std::string original = inst->toString();
        unsigned rdNum = rd->getRegNum();
        unsigned srcNum = src->getRegNum();
        auto rdClone = cloneReg(rd);
        auto srcClone = cloneReg(src);
        inst->clearOperands();
        inst->setOpcode(Opcode::ADDI);
        inst->addOperand_(std::move(rdClone));
        inst->addOperand_(std::move(srcClone));
        inst->addOperand_(std::make_unique<ImmediateOperand>(0));
        auto it = virtualRegisterConstants.find(srcNum);
        if (it != virtualRegisterConstants.end()) {
            mapRegToConstant(rdNum, it->second);
        } else {
            virtualRegisterConstants.erase(rdNum);
        }
        DEBUG_OUT() << "Bitwise simplify: '" << original << "' -> '"
                    << inst->toString() << "'" << std::endl;
    };

    auto emitNotFrom = [&](MachineOperand* src) {
        if (!src->isReg()) return;
        std::string original = inst->toString();
        unsigned rdNum = rd->getRegNum();
        unsigned srcNum = src->getRegNum();
        auto rdClone = cloneReg(rd);
        auto srcClone = cloneReg(src);
        inst->clearOperands();
        inst->setOpcode(Opcode::NOT);  // 伪指令：后续可降为 XORI rd, rs, -1
        inst->addOperand_(std::move(rdClone));
        inst->addOperand_(std::move(srcClone));
        auto it = virtualRegisterConstants.find(srcNum);
        if (it != virtualRegisterConstants.end()) {
            int32_t v = static_cast<int32_t>(it->second);
            int32_t res = static_cast<int32_t>(~v);
            mapRegToConstant(rdNum, res);
        } else {
            virtualRegisterConstants.erase(rdNum);
        }
        DEBUG_OUT() << "Bitwise simplify: '" << original << "' -> '"
                    << inst->toString() << "'" << std::endl;
    };

    auto constVal1 = getConstant(*op1);
    auto constVal2 = getConstant(*op2);

    auto isZero = [](const std::optional<int64_t>& v) {
        return v && static_cast<int32_t>(*v) == 0;
    };
    auto isMinusOne = [](const std::optional<int64_t>& v) {
        return v && static_cast<int32_t>(*v) == -1;
    };

    // 模式 3.1: 与自身
    if (op1->isReg() && op2->isReg() && op1->getRegNum() == op2->getRegNum()) {
        switch (opcode) {
            case Opcode::AND:
            case Opcode::OR:
                emitMovFrom(op1);
                return;
            case Opcode::XOR:
                emitLi(0);
                return;
            default:
                break;
        }
    }

    // 模式 3.2: 与常量（0 / -1）
    // 对于 R-Type 可交换操作（AND/OR/XOR），若常量在左侧尝试交换到右侧，
    // I-Type (ANDI/ORI/XORI) 不能交换操作数次序（语义不同），保持原状。
    auto isRTypeCommutative = (opcode == Opcode::AND || opcode == Opcode::OR ||
                               opcode == Opcode::XOR);
    if (isRTypeCommutative) {
        if (!constVal2 && constVal1) {
            std::swap(op1, op2);
            std::swap(constVal1, constVal2);
        }
    }

    // 现在 constVal2 若存在是优先考虑的常量
    if (constVal2) {
        if (opcode == Opcode::AND || opcode == Opcode::ANDI) {
            if (isZero(constVal2)) {  // x & 0 -> 0
                emitLi(0);
                return;
            }
            if (isMinusOne(constVal2) && op1->isReg()) {  // x & -1 -> x
                emitMovFrom(op1);
                return;
            }
        } else if (opcode == Opcode::OR || opcode == Opcode::ORI) {
            if (isZero(constVal2) && op1->isReg()) {  // x | 0 -> x
                emitMovFrom(op1);
                return;
            }
            if (isMinusOne(constVal2)) {  // x | -1 -> -1
                emitLi(-1);
                return;
            }
        } else if (opcode == Opcode::XOR || opcode == Opcode::XORI) {
            if (isMinusOne(constVal2) && op1->isReg()) {  // x ^ -1 -> ~x
                emitNotFrom(op1);
                return;
            }
            if (isZero(constVal2) && op1->isReg()) {  // x ^ 0 -> x
                emitMovFrom(op1);
                return;
            }
        }
    }

    // 若常量在左 (constVal1) 且右侧不是常量但没有被处理（例如 0 |
    // x），再处理一次对称
    if (constVal1 && !constVal2 && op2->isReg()) {
        // 仅对 R-Type 进行对称处理；I-Type 不调整
        if (opcode == Opcode::OR && isZero(constVal1)) {  // 0 | x -> x
            emitMovFrom(op2);
            return;
        }
        if (opcode == Opcode::AND && isMinusOne(constVal1)) {  // -1 & x -> x
            emitMovFrom(op2);
            return;
        }
        if (opcode == Opcode::XOR && isMinusOne(constVal1)) {  // -1 ^ x -> ~x
            emitNotFrom(op2);
            return;
        }
        if (opcode == Opcode::AND && isZero(constVal1)) {  // 0 & x -> 0
            emitLi(0);
            return;
        }
        if (opcode == Opcode::OR && isMinusOne(constVal1)) {  // -1 | x -> -1
            emitLi(-1);
            return;
        }
    }
}

void ConstantFolding::mvToAddiw(Instruction* inst, BasicBlock* parent_bb) {
    if (inst->getOpcode() != MV) {
        return;
    }

    // 把 MV 替换为 ADDI，便于后续传播
    auto dest_op = Visitor::cloneRegister(
        dynamic_cast<RegisterOperand*>(inst->getOperand(0)));
    auto src_op = Visitor::cloneRegister(
        dynamic_cast<RegisterOperand*>(inst->getOperand(1)));

    inst->clearOperands();
    inst->setOpcode(ADDIW);
    inst->addOperand_(std::move(dest_op));
    inst->addOperand_(std::move(src_op));
    inst->addOperand_(std::make_unique<ImmediateOperand>(0));
}

void ConstantFolding::instructionReassociateAndCombine(Instruction* inst,
                                                       BasicBlock* parent_bb) {
    auto isVReg = [](MachineOperand* operand) {
        return operand->isReg() && operand->getRegNum() >= 100;
    };

    // 1. 合并 ADDI 指令（前面已经把常量加减都变成了 ADDI）
    if (inst->getOpcode() == ADDI || inst->getOpcode() == ADDIW) {
        // src_reg_def: ADDI src_op, src_reg_def_src_op, imm1
        // inst: ADDI result, src_op, imm2
        auto* src_op = inst->getOperand(1);
        if (!isVReg(src_op)) {
            return;  // 需要是虚拟寄存器
        }

        auto* src_reg_def = parent_bb->getIntVRegDef(src_op->getRegNum());
        if (src_reg_def == nullptr) {
            return;  // 目标指令未定义，或者不在这个基本块
        }

        if (src_reg_def->getOpcode() != ADDI &&
            src_reg_def->getOpcode() != ADDIW) {
            return;  // 不是 ADDI 指令
        }

        auto* src_reg_def_src_op = src_reg_def->getOperand(1);
        if (!isVReg(src_reg_def_src_op)) {
            return;  // 需要是虚拟寄存器
        }

        if (inst->getOperand(0)->getRegNum() ==
            inst->getOperand(1)->getRegNum()) {
            return;
        }

        auto new_imm_val = getConstant(*src_reg_def->getOperand(2)).value() +
                           getConstant(*inst->getOperand(2)).value();
        if (Visitor::isValidImmediateOffset(new_imm_val)) {
            // 合并成一个新的 ADDI 指令
            auto original = inst->toString();
            unsigned dest_reg_num = inst->getOperand(0)->getRegNum();
            auto dest_clone = Visitor::cloneRegister(
                dynamic_cast<RegisterOperand*>(inst->getOperand(0)));
            // auto src_clone = Visitor::cloneRegister(
            //     dynamic_cast<RegisterOperand*>(src_reg_def->getOperand(0)));
            auto new_src = Visitor::cloneRegister(
                dynamic_cast<RegisterOperand*>(src_reg_def_src_op));

            inst->clearOperands();
            // inst->setOpcode(inst->getOpcode() == ADDI ? ADDI : ADDIW);
            inst->addOperand_(std::move(dest_clone));
            inst->addOperand_(std::move(new_src));
            inst->addOperand_(std::make_unique<ImmediateOperand>(new_imm_val));
            DEBUG_OUT() << "Reassociate and combine: '" << original << "' -> '"
                        << inst->toString() << "'" << std::endl;
        }

    } else if (inst->getOpcode() == MUL || inst->getOpcode() == MULW) {
    }
}

void ConstantFolding::useZeroReg(Instruction* inst, BasicBlock* parent_bb) {
    if (inst->getOpcode() != LI) {
        return;
    }

    if (inst->getOpcode() == LI && inst->getOperand(1)->getValue() == 0) {
        auto dest_reg = Visitor::cloneRegister(
            dynamic_cast<RegisterOperand*>(inst->getOperand(0)));
        DEBUG_OUT() << "Replace reg " << dest_reg->toString() << " with zero"
                    << std::endl;
        inst->clearOperands();
        inst->setOpcode(MV);
        inst->addOperand_(std::move(dest_reg));                        // rd
        inst->addOperand_(std::make_unique<RegisterOperand>("zero"));  // zero
    }
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
        case Opcode::MV: {
            if (source_operands.size() != 1) {
                return std::nullopt;  // MV should have one source operand
            }
            int32_t value = as_i32(source_operands[0]);
            return value;
        }
        case Opcode::ADD:
        case Opcode::ADDW:
        case Opcode::ADDI:
        case Opcode::ADDIW: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = sign_extend_i32(static_cast<int64_t>(a) +
                                          static_cast<int64_t>(b));
            DEBUG_OUT() << "Calculate Inst value: " << a << " + " << b << " -> "
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
            DEBUG_OUT() << "Calculate Inst value: " << a << " - " << b << " -> "
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
            DEBUG_OUT() << "Calculate Inst value: " << a << " * " << b << " -> "
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
            DEBUG_OUT() << "Calculate Inst value: " << a << " / " << b << " -> "
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
            DEBUG_OUT() << "Calculate Inst value: " << a << " /u " << b
                        << " -> " << result << std::endl;
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
                DEBUG_OUT() << "Calculate Inst value: " << a << " % " << b
                            << " (special) -> " << result << std::endl;
                return result;
            }
            auto result = sign_extend_i32(static_cast<int64_t>(a % b));
            DEBUG_OUT() << "Calculate Inst value: " << a << " % " << b << " -> "
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
            DEBUG_OUT() << "Calculate Inst value: " << a << " %u " << b
                        << " -> " << result << std::endl;
            return result;
        }

        // Bitwise / logic
        case Opcode::AND:
        case Opcode::ANDI: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = sign_extend_i32(a & b);
            DEBUG_OUT() << "Calculate Inst value: " << a << " & " << b << " -> "
                        << result << std::endl;
            return result;
        }
        case Opcode::OR:
        case Opcode::ORI: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = sign_extend_i32(a | b);
            DEBUG_OUT() << "Calculate Inst value: " << a << " | " << b << " -> "
                        << result << std::endl;
            return result;
        }
        case Opcode::XOR:
        case Opcode::XORI: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = sign_extend_i32(a ^ b);
            DEBUG_OUT() << "Calculate Inst value: " << a << " ^ " << b << " -> "
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
            DEBUG_OUT() << "Calculate Inst value: " << a << " << " << sh
                        << " -> " << result << std::endl;
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
            DEBUG_OUT() << "Calculate Inst value: " << a << " >>u " << sh
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
            DEBUG_OUT() << "Calculate Inst value: " << a << " >> " << sh
                        << " -> " << result << std::endl;
            return result;
        }

        // Comparisons (return 0/1)
        case Opcode::SLT:
        case Opcode::SLTI: {
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = static_cast<int64_t>(a < b ? 1 : 0);
            DEBUG_OUT() << "Calculate Inst value: " << a << " < " << b << " -> "
                        << result << std::endl;
            return result;
        }
        case Opcode::SGT: {  // pseudo >
            if (!needN(2)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            int32_t b = as_i32(source_operands[1]);
            auto result = static_cast<int64_t>(a > b ? 1 : 0);
            DEBUG_OUT() << "Calculate Inst value: " << a << " > " << b << " -> "
                        << result << std::endl;
            return result;
        }
        case Opcode::SLTU:
        case Opcode::SLTIU: {
            if (!needN(2)) return std::nullopt;
            uint32_t a = static_cast<uint32_t>(as_i32(source_operands[0]));
            uint32_t b = static_cast<uint32_t>(as_i32(source_operands[1]));
            auto result = static_cast<int64_t>(a < b ? 1 : 0);
            DEBUG_OUT() << "Calculate Inst value: " << a << " <u " << b
                        << " -> " << result << std::endl;
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
            DEBUG_OUT() << "Calculate Inst value: -" << a << " -> " << result
                        << std::endl;
            return result;
        }
        case Opcode::NOT: {
            if (!needN(1)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            auto result = sign_extend_i32(~a);
            DEBUG_OUT() << "Calculate Inst value: ~" << a << " -> " << result
                        << std::endl;
            return result;
        }
        case Opcode::SEQZ: {
            if (!needN(1)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            auto result = static_cast<int64_t>(a == 0 ? 1 : 0);
            DEBUG_OUT() << "Calculate Inst value: (" << a << " == 0) -> "
                        << result << std::endl;
            return result;
        }
        case Opcode::SNEZ: {
            if (!needN(1)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            auto result = static_cast<int64_t>(a != 0 ? 1 : 0);
            DEBUG_OUT() << "Calculate Inst value: (" << a << " != 0) -> "
                        << result << std::endl;
            return result;
        }
        case Opcode::SLTZ: {
            if (!needN(1)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            auto result = static_cast<int64_t>(a < 0 ? 1 : 0);
            DEBUG_OUT() << "Calculate Inst value: (" << a << " < 0) -> "
                        << result << std::endl;
            return result;
        }
        case Opcode::SGTZ: {
            if (!needN(1)) return std::nullopt;
            int32_t a = as_i32(source_operands[0]);
            auto result = static_cast<int64_t>(a > 0 ? 1 : 0);
            DEBUG_OUT() << "Calculate Inst value: (" << a << " > 0) -> "
                        << result << std::endl;
            return result;
        }

        default:
            return std::nullopt;  // Not (yet) supported
    }
}

}  // namespace riscv64