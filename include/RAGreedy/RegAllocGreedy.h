#pragma once

#include "Instructions/Function.h"
#include "RAGreedy/LiveIntervals.h"

namespace riscv64 {
class RegAllocGreedy {
   private:
    Function *function;

   public:
    explicit RegAllocGreedy(Function *func) : function(func) {};
    void run(void);

   private:
    void allocateRegisters();
    // void selectOrSplit(LI, Regs);

    void enqueue(LiveInterval* li);
    LiveInterval* dequeue();

    // MCRegister tryAssign(const LiveInterval &, AllocationOrder &,
    //                      SmallVectorImpl<Register> &, const SmallVirtRegSet &);
    // MCRegister tryEvict(const LiveInterval &, AllocationOrder &,
    //                     SmallVectorImpl<Register> &, uint8_t,
    //                     const SmallVirtRegSet &);
    // MCRegister tryRegionSplit(const LiveInterval &, AllocationOrder &,
    //                           SmallVectorImpl<Register> &);
    // MCRegister tryBlockSplit(const LiveInterval &, AllocationOrder &,
    //                          SmallVectorImpl<Register> &);
    // MCRegister tryLocalSplit(const LiveInterval &, AllocationOrder &,
    //                          SmallVectorImpl<Register> &);
    // MCRegister tryLastChanceRecoloring(const LiveInterval &, AllocationOrder &,
    //                                    SmallVectorImpl<Register> &,
    //                                    SmallVirtRegSet &, RecoloringStack &,
    //                                    unsigned);

};

}  // namespace riscv64