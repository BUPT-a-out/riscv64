#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Instructions/Instruction.h"

namespace riscv64 {

// 溢出临时寄存器管理器
class SpillChainManager {
   public:
    SpillChainManager(unsigned tempRegCounter, bool isFloat = false);
    // 为spill操作分配临时寄存器
    unsigned selectAvailablePhysicalDataReg(Instruction* inst);

   private:
    bool isFloat;
    unsigned tempRegCounter;

    std::vector<unsigned> availablePhysicalRegs;  // 可用的物理寄存器

    // 选择可用的物理寄存器
};

}  // namespace riscv64
