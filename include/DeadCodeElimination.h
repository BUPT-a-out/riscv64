// DeadCodeElimination.h
// Post-regalloc dead code elimination based on liveness analysis.
// Removes instructions that (1) define a virtual register, (2) that value
// is never used, and (3) the instruction has no observable side effects.
#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Instructions/Function.h"
#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"

// Debug output macro - only outputs when A_OUT_DEBUG is defined
#ifdef A_OUT_DEBUG
#define DCE_DEBUG() std::cout
#else
#define DCE_DEBUG() \
    if constexpr (false) std::cout
#endif

namespace riscv64 {

class DeadCodeElimination {
   public:
    struct BlockLiveness {
        std::unordered_set<unsigned> use;      // uses before def in block
        std::unordered_set<unsigned> def;      // defs in block
        std::unordered_set<unsigned> liveIn;   // live at entry
        std::unordered_set<unsigned> liveOut;  // live at exit
    };

    // Run DCE on a function. Returns true if any instruction removed.
    bool runOnFunction(Function* func);

   private:
    // Phase 1: build per-basic-block def/use sets.
    void computeDefUse(Function* func);
    // Phase 2: iterative data-flow to convergence.
    void livenessFixpoint(Function* func);
    // Phase 3: sweep instructions removing dead defs.
    bool eliminate(Function* func);

    bool isVirtualReg(unsigned reg) const {
        // Physical integer: 0-31, physical float: 32-63. Convention in backend
        // uses virtual regs with numbers >= 64 (e.g. %vreg_100). After
        // register allocation some may already be rewritten to physical regs;
        // we only eliminate still-virtual ones. This heuristic matches current
        // backend behavior. If future backend changes, adjust accordingly.
        return reg >= 64;  // conservative: never treat phys as virtual
    }

    bool hasSideEffects(Instruction* inst) const;

    std::vector<unsigned> getDefinedRegs(Instruction* inst) const;
    std::vector<unsigned> getUsedRegs(Instruction* inst) const;

   private:
    std::unordered_map<BasicBlock*, BlockLiveness> info_;
};

}  // namespace riscv64
