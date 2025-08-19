#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "Instructions/Function.h"
#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"
#include "StackFrameManager.h"

namespace riscv64 {

// 第三阶段：Frame Index Elimination Pass
// 负责计算最终栈帧布局并消除所有Frame Index伪指令
class FrameIndexElimination {
   private:
    Function* function;
    StackFrameManager* stackManager;

    // 最终的栈帧布局信息
    struct FinalFrameLayout {
        int totalFrameSize = 0;
        int returnAddressOffset = 0;  // ra相对于sp的偏移
        int framePointerOffset = 0;   // s0相对于sp的偏移

        // FI到最终偏移的映射(相对于s0/fp)
        std::unordered_map<int, int> frameIndexToOffset;
    };

    FinalFrameLayout layout;

   public:
    explicit FrameIndexElimination(Function* func)
        : function(func), stackManager(func->getStackFrameManager()) {}

    // 主要接口
    void run();

   private:
    // 第三阶段：计算最终栈帧布局
    void computeFinalFrameLayout();

    // 为所有栈对象分配最终偏移
    void assignFinalOffsets();

    // 生成最终的序言和尾声代码
    void generateFinalPrologueEpilogue();

    // 消除所有frameaddr伪指令
    void eliminateFrameIndices();

    // 替换单个frameaddr指令
    void eliminateFrameIndexInstruction(
        BasicBlock* bb, std::list<std::unique_ptr<Instruction>>::iterator& it);

    // 辅助函数
    int alignTo(int value, int alignment) const;
    void printFinalLayout() const;

    // 计算保存寄存器所需的空间
    int calculateSavedRegisterSize();

    // 计算最大调用参数所需的栈空间
    int calculateMaxCallArgSize();

    // 收集需要保存的寄存器列表
    std::vector<int> collectSavedIntegerRegisters();
    std::vector<int> collectSavedFloatRegisters();

    // 处理大偏移量的辅助函数
    bool isValidImmediateOffset(int64_t offset) const;

    // 当偏移量超出范围时，生成地址计算指令
    std::vector<std::unique_ptr<Instruction>> generateAddressComputation(
        int offset);

    // 恢复寄存器
    std::vector<std::unique_ptr<Instruction>> restoreRegister(int regNum,
                                                              int offset,
                                                              Opcode opcode);

    // 生成内存操作数
    std::unique_ptr<MemoryOperand> generateMemoryOperand(int offset,
                                                         bool useDirectOffset);

    // 生成寄存器保存指令
    std::vector<std::unique_ptr<Instruction>> generateRegisterSave(
        int regNum, int offset, Opcode opcode);

    // 生成栈指针调整指令
    std::vector<std::unique_ptr<Instruction>> generateStackAdjustment(
        int adjustment) {
        // 栈指针减法：sp = sp - adjustment
        return generateSubImm(2, 2, adjustment);  // 2是sp寄存器
    }

    std::vector<std::unique_ptr<Instruction>> generateFramePointerSetup(
        int frameSize) {
        // 帧指针设置：s0 = sp + frameSize
        return generateAddImm(8, 2, frameSize);  // 8是s0寄存器，2是sp寄存器
    }

    // 生成立即数加法指令 (dst = src + imm)
    std::vector<std::unique_ptr<Instruction>> generateAddImm(
        int dstReg, int srcReg, int imm, bool useT0 = true) {
        std::vector<std::unique_ptr<Instruction>> insts;

        if (isValidImmediateOffset(imm)) {
            auto addInst = std::make_unique<Instruction>(Opcode::ADDI);
            addInst->addOperand_(
                std::make_unique<RegisterOperand>(dstReg, false));
            addInst->addOperand_(
                std::make_unique<RegisterOperand>(srcReg, false));
            addInst->addOperand_(std::make_unique<ImmediateOperand>(imm));
            insts.push_back(std::move(addInst));
        } else {
            unsigned tmpReg = useT0 ? 5 : dstReg;
            // 使用临时寄存器处理大立即数
            auto liInst = std::make_unique<Instruction>(Opcode::LI);
            liInst->addOperand_(
                std::make_unique<RegisterOperand>(tmpReg, false));
            liInst->addOperand_(std::make_unique<ImmediateOperand>(imm));
            insts.push_back(std::move(liInst));

            auto addInst = std::make_unique<Instruction>(Opcode::ADD);
            addInst->addOperand_(
                std::make_unique<RegisterOperand>(dstReg, false));
            addInst->addOperand_(
                std::make_unique<RegisterOperand>(srcReg, false));
            addInst->addOperand_(
                std::make_unique<RegisterOperand>(tmpReg, false));
            insts.push_back(std::move(addInst));
        }

        return insts;
    }

    // 生成立即数减法指令 (dst = src - imm)
    std::vector<std::unique_ptr<Instruction>> generateSubImm(int dstReg,
                                                             int srcReg,
                                                             int imm) {
        std::vector<std::unique_ptr<Instruction>> insts;

        if (isValidImmediateOffset(-imm)) {
            // 可以用ADDI实现减法: dst = src + (-imm)
            auto addInst = std::make_unique<Instruction>(Opcode::ADDI);
            addInst->addOperand_(
                std::make_unique<RegisterOperand>(dstReg, false));
            addInst->addOperand_(
                std::make_unique<RegisterOperand>(srcReg, false));
            addInst->addOperand_(std::make_unique<ImmediateOperand>(-imm));
            insts.push_back(std::move(addInst));
        } else {
            // 使用临时寄存器处理大立即数
            auto liInst = std::make_unique<Instruction>(Opcode::LI);
            liInst->addOperand_(
                std::make_unique<RegisterOperand>(5, false));  // t0
            liInst->addOperand_(std::make_unique<ImmediateOperand>(imm));
            insts.push_back(std::move(liInst));

            auto subInst = std::make_unique<Instruction>(Opcode::SUB);
            subInst->addOperand_(
                std::make_unique<RegisterOperand>(dstReg, false));
            subInst->addOperand_(
                std::make_unique<RegisterOperand>(srcReg, false));
            subInst->addOperand_(
                std::make_unique<RegisterOperand>(5, false));  // t0
            insts.push_back(std::move(subInst));
        }

        return insts;
    }

    // Peephole: fold address materialization (addi rd, rs, imm) followed by
    // memory ops using 0(rd) into direct memory ops imm(rs). We attempt this
    // aggressively for lw/ld/sw/sd/flw/fsw and their 8/16/32-bit variants as
    // long as the final immediate stays within the signed 12-bit range and
    // 'rd' has no other non-memory uses before being redefined.
    void foldAddressIntoMemory();
    static bool isMemoryAccessOpcode(Opcode opc) {
        switch (opc) {
            case LW:
            case LD:
            case LH:
            case LHU:
            case LB:
            case LBU:
            case LWU:
            case SW:
            case SD:
            case SH:
            case SB:
            case FLW:
            case FSW:
            case FLD:
            case FSD:
                return true;
            default:
                return false;
        }
    }
};

}  // namespace riscv64