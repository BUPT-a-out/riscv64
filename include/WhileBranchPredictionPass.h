#pragma once

#include "Instructions/Function.h"
#include "Pass/Analysis/LoopInfo.h"

// A simple pass to invert while-loop branch conditions so that the loop body
// becomes the fallthrough path (more likely taken) and the merge/exit block is
// reached via a conditional branch with the inverted predicate. This reduces
// the number of taken unconditional jumps inside typical while loops.
//
// Pattern before:
//   bnez cond, while.loop   ; cond true -> jump to loop body
//   j while.merge           ; cond false -> fallthrough after reordering
//   ...
// After (desired):
//   beqz cond, while.merge  ; cond false -> branch to merge (less likely)
//   while.loop:             ; fallthrough is loop body (hot path)
// (The redundant unconditional jump is removed and block order adjusted.)
//
// We detect blocks ending with (COND_BR to *.while.loop) + (J to *.while.merge)
// and rewrite them accordingly. Only simple single-register zero compare
// pseudo branches are handled (BNEZ/BEQZ and a few variants mapping).

namespace riscv64 {

class WhileBranchPredictionPass {
   public:
    void runOnFunction(Function* func, const midend::Function* mid_func,
                       const midend::LoopInfo* loopInfo);

   private:
    Opcode invertSimpleBranch(Opcode op) {
        switch (op) {
            case Opcode::BNEZ:
                return Opcode::BEQZ;
            case Opcode::BEQZ:
                return Opcode::BNEZ;  // (not expected input)
            case Opcode::BLEZ:
                return Opcode::BGTZ;
            case Opcode::BGTZ:
                return Opcode::BLEZ;
            case Opcode::BGEZ:
                return Opcode::BLTZ;
            case Opcode::BLTZ:
                return Opcode::BGEZ;
            default:
                return op;  // unsupported
        }
    }
};

}  // namespace riscv64
