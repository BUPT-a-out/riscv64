#pragma once

#include "Instructions/Function.h"
#include "flat_hash_map/unordered_map.hpp"
namespace riscv64 {

class FoldMemoryAccessPass {
   private:
    ska::unordered_map<unsigned, int> offsetMap;
    ska::unordered_map<unsigned, int> valueMap;

    void runOnBlock(BasicBlock* bb);
    void processInstr(Instruction* instr);

   public:
    void runOnFunction(Function* fn) {
        for (auto& bb : *fn) {
            runOnBlock(bb.get());
        }
    }
};

}  // namespace riscv64