#pragma once

// #include "BasicBlock.h"
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "MachineOperand.h"

namespace riscv64 {

class BasicBlock;  // 前向声明

// RISC-V 指令操作码枚举
enum Opcode {
    // RV64I Base Integer Instruction Set
    // R-Type
    ADD,
    SUB,
    SLL,
    SLT,
    SGT,
    SLTU,
    XOR,
    SRL,
    SRA,
    OR,
    AND,
    ADDW,
    SUBW,
    SLLW,
    SRLW,
    SRAW,

    // I-Type
    ADDI,
    SLTI,
    SLTIU,
    XORI,
    ORI,
    ANDI,
    SLLI,
    SRLI,
    SRAI,
    ADDIW,
    SLLIW,
    SRLIW,
    SRAIW,
    LB,
    LH,
    LW,
    LD,
    LBU,
    LHU,
    LWU,
    JALR,

    // S-Type
    SB,
    SH,
    SW,
    SD,

    // B-Type
    BEQ,
    BNE,
    BLT,
    BGE,
    BLTU,
    BGEU,

    // U-Type
    LUI,
    AUIPC,

    // J-Type
    JAL,

    // System
    ECALL,
    EBREAK,
    FENCE,

    // RV64M Standard Extension for Integer Multiplication and Division
    MUL,
    MULH,
    MULHSU,
    MULHU,
    DIV,
    DIVU,
    REM,
    REMU,
    MULW,
    DIVW,
    DIVUW,
    REMW,
    REMUW,

    // RV64F Standard Extension for Single-Precision Floating-Point
    FLW,
    FSW,
    FMADD_S,
    FMSUB_S,
    FNMSUB_S,
    FNMADD_S,
    FADD_S,
    FSUB_S,
    FMUL_S,
    FDIV_S,
    FSQRT_S,
    FSGNJ_S,
    FSGNJN_S,
    FSGNJX_S,
    FMIN_S,
    FMAX_S,
    FCVT_W_S,
    FCVT_WU_S,
    FMV_X_W,
    FEQ_S,
    FLT_S,
    FLE_S,
    FCLASS_S,
    FCVT_S_W,
    FCVT_S_WU,
    FMV_W_X,
    FCVT_L_S,
    FCVT_LU_S,
    FCVT_S_L,
    FCVT_S_LU,

    // RV64D Standard Extension for Double-Precision Floating-Point
    FLD,
    FSD,
    FMADD_D,
    FMSUB_D,
    FNMSUB_D,
    FNMADD_D,
    FADD_D,
    FSUB_D,
    FMUL_D,
    FDIV_D,
    FSQRT_D,
    FSGNJ_D,
    FSGNJN_D,
    FSGNJX_D,
    FMIN_D,
    FMAX_D,
    FCVT_S_D,
    FCVT_D_S,
    FEQ_D,
    FLT_D,
    FLE_D,
    FCLASS_D,
    FCVT_W_D,
    FCVT_WU_D,
    FCVT_D_W,
    FCVT_D_WU,
    FCVT_L_D,
    FCVT_LU_D,
    FMV_X_D,
    FCVT_D_L,
    FCVT_D_LU,
    FMV_D_X,

    // Pseudo Instructions
    LI,
    MV,
    NOT,
    NEG,
    NEGW,
    SEXT_W,
    SEQZ,
    SNEZ,
    SLTZ,
    SGTZ,
    BEQZ,
    BNEZ,
    BLEZ,
    BGEZ,
    BLTZ,
    BGTZ,
    BGT,
    BLE,
    BGTU,
    BLEU,
    J,
    JR,
    RET,
    CALL,
    TAIL,
    FMOV_S,
    FABS_S,
    FNEG_S,
    FMOV_D,
    FABS_D,
    FNEG_D,
    NOP,

    // 特殊控制流伪指令
    COPY  // 表示约束“这个虚拟寄存器的值，必须被放入那个特定的物理寄存器中”
};
using DestSourcePair = std::pair<MachineOperand*, MachineOperand*>;

class Instruction {
   public:
    explicit Instruction(Opcode op) : opcode(op) {}
    explicit Instruction(Opcode op, BasicBlock* parent)
        : opcode(op), parent(parent) {}
    explicit Instruction(Opcode op, std::vector<std::unique_ptr<MachineOperand>>&& operands_vec, BasicBlock* parent = nullptr)
    : opcode(op), operands(std::move(operands_vec)), parent(parent) {}

    // 添加操作数
    void addOperand(std::unique_ptr<MachineOperand> operand) {
        operands.push_back(std::move(operand));
    }

    Opcode getOpcode() const { return opcode; }
    const std::vector<std::unique_ptr<MachineOperand>>& getOperands() const {
        return operands;
    }
    auto getOprandCount() const { return operands.size(); }

    // 为了方便，可以提供一些辅助函数
    // 例如：获取第 n 个操作数
    MachineOperand* getOperand(std::size_t n) const {
        return operands[n].get();
    }

    BasicBlock* getParent() const { return parent; }
    void setParent(BasicBlock* bb) { parent = bb; }

    // 生成文本表示
    std::string toString() const;

    std::optional<DestSourcePair> isCopyInstr() const;
    bool isCallInstr() const {
        return opcode == CALL || opcode == JAL;
    }

   private:
    Opcode opcode;
    std::vector<std::unique_ptr<MachineOperand>> operands;
    std::optional<DestSourcePair> isCopyInstrImpl() const;

    // 指向其所在的基本块 (可选，但非常有用) - 弱引用
    BasicBlock* parent{};
};

}  // namespace riscv64