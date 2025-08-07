// SpillChainManager.cpp
#include "SpillChainManager.h"

#include <algorithm>
#include <iostream>

#include "Instructions/Instruction.h"

namespace riscv64 {

SpillChainManager::SpillChainManager(bool isFloat)
    : isFloat(isFloat) {
    // 初始化临时寄存器优先级顺序

    // 整数优先使用临时寄存器 t0-t6
    std::vector<unsigned> tempIntegerPriority = {6,  7,  28,
                                                 29, 30, 31};  // t1-t2, t3-t6
    // t0 固定作为地址寄存器, 先避开

    // 浮点优先使用临时寄存器 ft0-ft7
    std::vector<unsigned> tempFloatPriority = {32, 33, 34, 35,
                                               36, 37, 38,39, 
                                               60, 61, 62, 63};  // ft0-ft7 ft8-ft11

    auto tempPriority = isFloat ? tempFloatPriority : tempIntegerPriority;

    // 重新排序可用寄存器，临时寄存器优先
    availablePhysicalRegs.clear();
    for (unsigned reg : tempPriority) {
        availablePhysicalRegs.push_back(reg);
    }
}

unsigned SpillChainManager::selectAvailablePhysicalDataReg(Instruction* inst) {
    auto usedInInst =
        isFloat ? inst->getUsedFloatRegs() : inst->getUsedIntegerRegs();

    // 选择第一个不冲突且未使用的物理寄存器
    for (unsigned physReg : availablePhysicalRegs) {
        if (std::find(usedInInst.begin(), usedInInst.end(), physReg) ==
            usedInInst.end()) {
            return physReg;
        }
    }

    return 0;  // 没有可用寄存器
}

}  // namespace riscv64
