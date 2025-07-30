#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "Instructions/All.h"

namespace riscv64 {
// Machine Operand 相关的 toString 实现
std::string MachineOperand::toString() const {
    switch (getType()) {
        case OperandType::Register:
            return dynamic_cast<const RegisterOperand*>(this)->toString();
        case OperandType::Immediate:
            return dynamic_cast<const ImmediateOperand*>(this)->toString();
        case OperandType::Label:
            return dynamic_cast<const LabelOperand*>(this)->toString();
        case OperandType::Memory:
            return dynamic_cast<const MemoryOperand*>(this)->toString();
        case OperandType::FrameIndex:
            return dynamic_cast<const FrameIndexOperand*>(this)->toString();
        default:
            throw std::runtime_error("Unknown operand type");
    }
}

std::string RegisterOperand::toString(bool use_abi) const {
    if (isFloatRegister()) {
        if (isVirtual()) {
            return "%freg_" + std::to_string(regNum);
        }
        if (use_abi) {
            return ABI::getABINameFromRegNum(regNum);
        }
        return "f" + std::to_string(regNum);
    }
    if (isVirtual()) {
        return "%vreg_" + std::to_string(regNum);
    }
    if (use_abi) {
        return ABI::getABINameFromRegNum(regNum);
    }
    return "x" + std::to_string(regNum);
}

std::string ImmediateOperand::toString() const {
    if (type == RegisterType::Float) {
        return std::to_string(getFloatValue());
    }
    return std::to_string(value);
}

std::string LabelOperand::toString() const { return getLabelName(); }

std::string MemoryOperand::toString() const {
    return std::to_string(getOffset()->getValue()) + "(" +
           getBaseReg()->toString() + ")";
}

std::string FrameIndexOperand::toString() const {
    return "FI(" + std::to_string(getIndex()) + ")";
}

std::string getInstructionName(Opcode opcode) {
    static const std::unordered_map<Opcode, std::string> opcodeNames = {
        {Opcode::ADD, "add"},
        {Opcode::SUB, "sub"},
        {Opcode::MUL, "mul"},
        {Opcode::DIV, "div"},
        {Opcode::REM, "rem"},
        {Opcode::XOR, "xor"},
        {Opcode::AND, "and"},
        {Opcode::RET, "ret"},
        {Opcode::LI, "li"},
        {Opcode::MV, "mv"},
        {Opcode::ADDI, "addi"},
        {Opcode::BNEZ, "bnez"},
        {Opcode::SEQZ, "seqz"},
        {Opcode::SNEZ, "snez"},
        {Opcode::SLTZ, "sltz"},
        {Opcode::SGTZ, "sgtz"},
        {Opcode::BEQZ, "beqz"},
        {Opcode::BGTZ, "bgtz"},
        {Opcode::BGEZ, "bgez"},
        {Opcode::BLTZ, "bltz"},
        {Opcode::BGT, "bgt"},
        {Opcode::BLE, "ble"},
        {Opcode::BGTU, "bgtu"},
        {Opcode::J, "j"},
        {Opcode::SLT, "slt"},
        {Opcode::SGT, "sgt"},
        {Opcode::SLTI, "slti"},
        {Opcode::CALL, "call"},
        {Opcode::FRAMEADDR, "frameaddr"},
        {Opcode::SW, "sw"},
        {Opcode::LW, "lw"},
        {Opcode::LA, "la"},
        {Opcode::BEQZ, "beqz"},
        {Opcode::SD, "sd"},
        {Opcode::LD, "ld"},
        {Opcode::SLL, "sll"},
        {Opcode::SRL, "srl"},
        {Opcode::SLLI, "slli"},
        {Opcode::SRLI, "srli"},
        {Opcode::SRA, "sra"},
        {Opcode::OR, "or"},
        {Opcode::MULH, "mulh"},
        {Opcode::MULHSU, "mulhsu"},
        {Opcode::MULHU, "mulhu"},
        {Opcode::DIVU, "divu"},
        {Opcode::REMU, "remu"},
        {Opcode::SLTU, "sltu"},
        {Opcode::BEQ, "beq"},
        {Opcode::BNE, "bne"},
        {Opcode::BLT, "blt"},
        {Opcode::BGE, "bge"},
        {Opcode::BLTU, "bltu"},
        {Opcode::BGEU, "bgeu"},
        {Opcode::ORI, "ori"},
        {Opcode::XORI, "xori"},
        {Opcode::ANDI, "andi"},
        {Opcode::LUI, "lui"},
        {Opcode::SLTIU, "sltiu"},
        {Opcode::ADD, "add"},
        {Opcode::SUB, "sub"},
        {Opcode::XOR, "xor"},
        {Opcode::AND, "and"},
        {Opcode::ADDW, "addw"},
        {Opcode::SUBW, "subw"},
        {Opcode::SLLW, "sllw"},
        {Opcode::SRLW, "srlw"},
        {Opcode::SRAW, "sraw"},
        {Opcode::SRAI, "srai"},
        {Opcode::ADDIW, "addiw"},
        {Opcode::SLLIW, "slliw"},
        {Opcode::SRLIW, "srliw"},
        {Opcode::SRAIW, "sraiw"},
        {Opcode::LB, "lb"},
        {Opcode::LH, "lh"},
        {Opcode::LBU, "lbu"},
        {Opcode::LHU, "lhu"},
        {Opcode::LWU, "lwu"},
        {Opcode::JALR, "jalr"},
        {Opcode::SB, "sb"},
        {Opcode::SH, "sh"},
        {Opcode::AUIPC, "auipc"},
        {Opcode::JAL, "jal"},
        {Opcode::ECALL, "ecall"},
        {Opcode::EBREAK, "ebreak"},
        {Opcode::FENCE, "fence"},
        {Opcode::MUL, "mul"},
        {Opcode::DIV, "div"},
        {Opcode::REM, "rem"},
        {Opcode::MULW, "mulw"},
        {Opcode::DIVW, "divw"},
        {Opcode::DIVUW, "divuw"},
        {Opcode::REMW, "remw"},
        {Opcode::REMUW, "remuw"},
        {Opcode::FLW, "flw"},
        {Opcode::FSW, "fsw"},
        {Opcode::FMADD_S, "fmadd.s"},
        {Opcode::FMSUB_S, "fmsub.s"},
        {Opcode::FNMSUB_S, "fnmsub.s"},
        {Opcode::FNMADD_S, "fnmadd.s"},
        {Opcode::FADD_S, "fadd.s"},
        {Opcode::FSUB_S, "fsub.s"},
        {Opcode::FMUL_S, "fmul.s"},
        {Opcode::FDIV_S, "fdiv.s"},
        {Opcode::FSQRT_S, "fsqrt.s"},
        {Opcode::FSGNJ_S, "fsgnj.s"},
        {Opcode::FSGNJN_S, "fsgnjn.s"},
        {Opcode::FSGNJX_S, "fsgnjx.s"},
        {Opcode::FMIN_S, "fmin.s"},
        {Opcode::FMAX_S, "fmax.s"},
        {Opcode::FCVT_W_S, "fcvt.w.s"},
        {Opcode::FCVT_WU_S, "fcvt.wu.s"},
        {Opcode::FMV_X_W, "fmv.x.w"},
        {Opcode::FEQ_S, "feq.s"},
        {Opcode::FLT_S, "flt.s"},
        {Opcode::FLE_S, "fle.s"},
        {Opcode::FCLASS_S, "fclass.s"},
        {Opcode::FCVT_S_W, "fcvt.s.w"},
        {Opcode::FCVT_S_WU, "fcvt.s.wu"},
        {Opcode::FMV_W_X, "fmv.w.x"},
        {Opcode::FCVT_L_S, "fcvt.l.s"},
        {Opcode::FCVT_LU_S, "fcvt.lu.s"},
        {Opcode::FCVT_S_L, "fcvt.s.l"},
        {Opcode::FCVT_S_LU, "fcvt.s.lu"},
        {Opcode::FLD, "fld"},
        {Opcode::FSD, "fsd"},
        {Opcode::FMADD_D, "fmadd.d"},
        {Opcode::FMSUB_D, "fmsub.d"},
        {Opcode::FNMSUB_D, "fnmsub.d"},
        {Opcode::FNMADD_D, "fnmadd.d"},
        {Opcode::FADD_D, "fadd.d"},
        {Opcode::FSUB_D, "fsub.d"},
        {Opcode::FMUL_D, "fmul.d"},
        {Opcode::FDIV_D, "fdiv.d"},
        {Opcode::FSQRT_D, "fsqrt.d"},
        {Opcode::FSGNJ_D, "fsgnj.d"},
        {Opcode::FSGNJN_D, "fsgnjn.d"},
        {Opcode::FSGNJX_D, "fsgnjx.d"},
        {Opcode::FMIN_D, "fmin.d"},
        {Opcode::FMAX_D, "fmax.d"},
        {Opcode::FCVT_S_D, "fcvt.s.d"},
        {Opcode::FCVT_D_S, "fcvt.d.s"},
        {Opcode::FEQ_D, "feq.d"},
        {Opcode::FLT_D, "flt.d"},
        {Opcode::FLE_D, "fle.d"},
        {Opcode::FCLASS_D, "fclass.d"},
        {Opcode::FCVT_W_D, "fcvt.w.d"},
        {Opcode::FCVT_WU_D, "fcvt.wu.d"},
        {Opcode::FCVT_D_W, "fcvt.d.w"},
        {Opcode::FCVT_D_WU, "fcvt.d.wu"},
        {Opcode::FCVT_L_D, "fcvt.l.d"},
        {Opcode::FCVT_LU_D, "fcvt.lu.d"},
        {Opcode::FMV_X_D, "fmv.x.d"},
        {Opcode::FCVT_D_L, "fcvt.d.l"},
        {Opcode::FCVT_D_LU, "fcvt.d.lu"},
        {Opcode::FMV_D_X, "fmv.d.x"},
        {Opcode::NOT, "not"},
        {Opcode::NEG, "neg"},
        {Opcode::NEGW, "negw"},
        {Opcode::SEXT_W, "sext.w"},
        {Opcode::BLEZ, "blez"},
        {Opcode::BLEU, "bleu"},
        {Opcode::JR, "jr"},
        {Opcode::RET, "ret"},
        {Opcode::TAIL, "tail"},
        {Opcode::FMOV_S, "fmv.s"},
        {Opcode::FABS_S, "fabs.s"},
        {Opcode::FNEG_S, "fneg.s"},
        {Opcode::FMOV_D, "fmov.d"},
        {Opcode::FABS_D, "fabs.d"},
        {Opcode::FNEG_D, "fneg.d"},
        {Opcode::NOP, "nop"},
        {Opcode::COPY, "copy"},
        // TODO(rikka): 添加其他操作码的名称...
    };

    auto it = opcodeNames.find(opcode);
    if (it != opcodeNames.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown opcode: " +
                             std::to_string(static_cast<int>(opcode)));
}

std::string Instruction::toString() const {
    auto result = getInstructionName(opcode);
    auto operand_count = getOprandCount();
    if (operand_count == 0) {
        return result;
    }

    result += ' ';
    for (size_t index = 0; index < operand_count; ++index) {
        if (index > 0) {
            result += ", ";
        }
        result += getOperand(index)->toString();
    }

    return result;
}

std::string BasicBlock::toString() const {
    std::string result = label + ":\n";
    for (const auto& inst : instructions) {
        result += "  " + inst->toString() + "\n";
    }
    return result;
}

std::string Function::toString() const {
    std::string result = name + ":\n";
    for (const auto& bb : basic_blocks) {
        result += bb->toString();
    }
    return result;
}

std::string Module::toString() const {
    std::string result;
    // 输出3个 Segment
    result += rodata_segment_.toString();
    result += data_segment_.toString();
    result += bss_segment_.toString();

    result += "  .text\n";
    // for (const auto& global : global_vars) {
    //     result += "  " + global->toString() + "\n";
    // }
    // TODO(rikka): 处理全局变量的输出，修改数据结构
    for (const auto& func : functions) {
        result += "  .globl " + func->getName() + "\n";
        result += func->toString() + "\n";
    }
    return result;
}

// Helper to get section name
const char* getSectionName(SegmentKind kind) {
    switch (kind) {
        case SegmentKind::DATA:
            return ".data";
        case SegmentKind::RODATA:
            return ".rodata";
        case SegmentKind::BSS:
            return ".bss";
    }
    return "";  // Should not happen
}

// Implementation of DataSegment::generateAsm
std::string DataSegment::toString() const {
    if (items_.empty()) {
        return "";
    }

    std::string result =
        "\n\t.section " + std::string(getSectionName(kind_)) + "\n";

    for (const auto& var : items_) {
        result += "  .globl " + var.name + "\n";
        result += "  .align 2\n";  // 4-byte alignment
        result += var.name + ":\n";

        if (kind_ == SegmentKind::BSS) {
            result +=
                "  .space " + std::to_string(var.type.getSizeInBytes()) + "\n";
        } else {
            // Visitor to handle different initializer types
            auto initializer_visitor = [&](const auto& value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, int32_t>) {
                    result += "  .word " + std::to_string(value) + "\n";
                } else if constexpr (std::is_same_v<T, float>) {
                    // Note: Emitting floats might require converting to hex
                    // representation for gas, but for simplicity we'll just
                    // print the value.
                    result += "  .float " + std::to_string(value) + "\n";
                } else if constexpr (std::is_same_v<T, std::vector<int32_t>>) {
                    // Optimize consecutive zeros with .space directive (only
                    // for 2+ consecutive zeros)
                    size_t i = 0;
                    while (i < value.size()) {
                        if (value[i] != 0) {
                            result +=
                                "  .word " + std::to_string(value[i]) + "\n";
                            i++;
                        } else {
                            // Count consecutive zeros
                            size_t zero_start = i;
                            while (i < value.size() && value[i] == 0) {
                                i++;
                            }
                            size_t zero_count = i - zero_start;

                            // Only use .space for 2+ consecutive zeros,
                            // otherwise use .word 0
                            if (zero_count >= 2) {
                                result += "  .space " +
                                          std::to_string(zero_count * 4) + "\n";
                            } else {
                                // Single zero, use .word 0 for clarity
                                result += "  .word 0\n";
                            }
                        }
                    }
                } else if constexpr (std::is_same_v<T, std::vector<float>>) {
                    // Optimize consecutive zeros with .space directive (only
                    // for 2+ consecutive zeros)
                    size_t i = 0;
                    while (i < value.size()) {
                        if (value[i] != 0.0f) {
                            result +=
                                "  .float " + std::to_string(value[i]) + "\n";
                            i++;
                        } else {
                            // Count consecutive zeros
                            size_t zero_start = i;
                            while (i < value.size() && value[i] == 0.0f) {
                                i++;
                            }
                            size_t zero_count = i - zero_start;

                            // Only use .space for 2+ consecutive zeros,
                            // otherwise use .float 0.0
                            if (zero_count >= 2) {
                                result += "  .space " +
                                          std::to_string(zero_count * 4) + "\n";
                            } else {
                                // Single zero, use .float 0.0 for clarity
                                result += "  .float 0.0\n";
                            }
                        }
                    }
                }
                // ZeroInitializer is handled by BSS logic, so it shouldn't be
                // visited here.
            };

            if (var.initializer.has_value()) {
                std::visit(initializer_visitor, var.initializer.value());
            }
        }
    }

    return result;
}

}  // namespace riscv64