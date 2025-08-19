#include "FoldMemoryAccessPass.h"

#include "Instructions/All.h"
#include "Target.h"
#include "Visit.h"

#ifdef A_OUT_DEBUG
#define DEBUG_OUT() std::cout
#else
#define DEBUG_OUT() \
    if constexpr (false) std::cout
#endif

namespace riscv64 {

void FoldMemoryAccessPass::processInstr(Instruction* instr) {
    auto opCode = instr->getOpcode();
    if (opCode == Opcode::ADDI) {
        // 处理 addi rd, rs1, imm 指令
        // addi t0, s0, -10
        auto rd = instr->getOperand(0)->getRegNum();
        auto rs1 = instr->getOperand(1)->getRegNum();
        auto imm = instr->getOperand(2)->getValue();

        // 检查 rs1 是否已经跟踪了offset信息
        if (offsetMap.count(rs1)) {
            // rs1 = base + old_offset, 现在 rd = rs1 + imm = base + (old_offset
            // + imm)
            auto old_offset = offsetMap[rs1];
            int new_offset = old_offset + imm;

            // 检查新的offset是否在有效范围内
            if (Visitor::isValidImmediateOffset(new_offset)) {
                offsetMap[rd] = new_offset;
            }
            return;
        }
        if (rs1 == 8) {  // s0
            offsetMap[rd] = imm;
            return;
        }

        // 跟踪常数值
        if (valueMap.count(rs1)) {
            valueMap[rd] = valueMap[rs1] + imm;
        } else {
            // 休矣, 无法武断.
            valueMap.erase(rd);
            offsetMap.erase(rd);
        }
        return;
    }

    if (opCode == Opcode::LI) {
        auto rd = instr->getOperand(0)->getRegNum();
        auto imm = instr->getOperand(1)->getValue();
        valueMap[rd] = imm;
        return;
    }

    if (opCode == Opcode::MV) {
        auto dst = instr->getOperand(0)->getRegNum();
        auto src = instr->getOperand(1)->getRegNum();
        if (offsetMap.count(src)) {
            offsetMap[dst] = offsetMap[src];
            valueMap.erase(dst);
            return;
        } else {
            offsetMap.erase(dst);
            return;
        }
    
        if (valueMap.count(src)) {
            valueMap[dst] = valueMap[src];
            offsetMap.erase(dst);
            return;
        } else {
            valueMap.erase(dst);
        }
    }

    if (instr->isMemoryInstr()) {
        // 处理内存访问指令：lw, sw, lb, sb, lh, sh 等
        auto val_reg = instr->getOperand(0)->getRegNum();

        auto memOp = static_cast<MemoryOperand*>(instr->getOperand(1));
        auto addr_reg = memOp->getBaseReg()->getRegNum();
        auto current_offset = memOp->getOffset()->getValue();

        // 检查地址寄存器是否可以被优化
        if (offsetMap.count(addr_reg)) {
            auto tracked_offset = offsetMap[addr_reg];
            int total_offset = current_offset + tracked_offset;

            // 检查总offset是否在有效范围内
            if (Visitor::isValidImmediateOffset(total_offset)) {
                // 执行优化：替换基址寄存器和offset
                DEBUG_OUT() << instr->toString() << " folded to ";
                memOp->getBaseReg()->setPhysicalRegForce(8);  // s0
                memOp->getOffset()->setIntValue(total_offset);

                DEBUG_OUT() << instr->toString() << std::endl;
            }
        }

        // TODO: extract func, avoid mis-operation
        // 内存值不可预料.
        if (opCode == LW || opCode == LD) {
            valueMap.erase(val_reg);
            offsetMap.erase(val_reg);
        }
        return;
    }


    // 其他指令可能会修改寄存器，需要清理对应的跟踪信息
    // 只需要关心整数
    auto defs = instr->getDefinedIntegerRegs();

    for (auto def : defs) {
        offsetMap.erase(def);
        valueMap.erase(def);
    }
}

void FoldMemoryAccessPass::runOnBlock(BasicBlock* bb) {
    offsetMap.clear();
    valueMap.clear();
    for (auto& instr : *bb) {
        processInstr(instr.get());
    }
}

Module& RISCV64Target::foldMemoryAccessPass(riscv64::Module& module) {
    DEBUG_OUT() << "\n=== FoldMem ===" << std::endl;
    for (auto& func : module) {
        if (func->empty()) {
            continue;  // Skip empty functions
        }
        auto pass = FoldMemoryAccessPass();
        pass.runOnFunction(func.get());
    }
    return module;
}

}  // namespace riscv64