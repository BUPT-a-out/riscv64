#include "Instructions/BasicBlock.h"
#include "Instructions/Function.h"

namespace riscv64 {

class SpillCodeOptimizer {
   public:
    static void optimizeSpillCode(Function* function);

   private:
    static void removeRedundantFrameAddr(Function* function);
    static void replaceIntegerRegisterInBasicBlock(BasicBlock* bb,
                                                   unsigned oldReg,
                                                   unsigned newReg);
    static bool isFrameAddrInstruction(Instruction* inst, int& frameIndex,
                                       unsigned& dstReg);
};
}  // namespace riscv64