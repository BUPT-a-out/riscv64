// SpillChainManager.cpp
#include "SpillChainManager.h"

#include <algorithm>
#include <iostream>

#include "Instructions/Instruction.h"

namespace riscv64 {

SpillChainManager::SpillChainManager(const std::vector<unsigned>& availableRegs)
    : availablePhysicalRegs(availableRegs) {
    // 初始化临时寄存器优先级顺序
    // 优先使用临时寄存器 t0-t6
    std::vector<unsigned> tempPriority = {5,  6,  7, 28,
                                          29, 30, 31};  // t0-t2, t3-t6

    // 重新排序可用寄存器，临时寄存器优先
    availablePhysicalRegs.clear();
    for (unsigned reg : tempPriority) {
        if (std::find(availableRegs.begin(), availableRegs.end(), reg) !=
            availableRegs.end()) {
            availablePhysicalRegs.push_back(reg);
        }
    }

    // 添加其他可用寄存器
    for (unsigned reg : availableRegs) {
        if (std::find(tempPriority.begin(), tempPriority.end(), reg) ==
            tempPriority.end()) {
            availablePhysicalRegs.push_back(reg);
        }
    }
}

unsigned SpillChainManager::allocateTempRegister(unsigned spilledReg,
                                                 Instruction* inst) {
    // 检查是否超过最大spill链深度
    int currentDepth = getSpillChainDepth(spilledReg);
    if (currentDepth >= MAX_SPILL_CHAIN_DEPTH) {
        std::cerr << "Error: Maximum spill chain depth exceeded for register "
                  << spilledReg << std::endl;
        // 强制选择一个物理寄存器，即使可能冲突
        return selectAvailablePhysicalReg(inst);
    }

    // 尝试为spill操作分配临时寄存器
    unsigned physReg = selectAvailablePhysicalReg(inst);
    if (physReg == 0) {
        std::cerr << "Error: No available physical register for spill operation"
                  << std::endl;
        return 5;  // 默认使用t0
    }

    // 创建临时寄存器信息
    TempRegInfo tempInfo;
    tempInfo.physicalReg = physReg;
    tempInfo.originalReg = spilledReg;
    tempInfo.isInUse = true;
    tempInfo.chainDepth = currentDepth + 1;

    // 分配新的临时寄存器ID（虚拟寄存器ID范围之外）
    static unsigned tempRegCounter = 100000;
    unsigned tempRegId = tempRegCounter++;

    tempRegMap[tempRegId] = tempInfo;
    usedPhysicalRegs.insert(physReg);

    // 更新spill链深度
    updateSpillChainDepth(tempRegId, tempInfo.chainDepth);

    return physReg;  // 直接返回物理寄存器
}

void SpillChainManager::releaseTempRegister(unsigned tempReg) {
    auto it = tempRegMap.find(tempReg);
    if (it != tempRegMap.end()) {
        usedPhysicalRegs.erase(it->second.physicalReg);
        tempRegMap.erase(it);
    }
}

bool SpillChainManager::isTempRegister(unsigned reg) const {
    return tempRegMap.find(reg) != tempRegMap.end();
}

void SpillChainManager::resetTempRegisters() {
    tempRegMap.clear();
    usedPhysicalRegs.clear();
    spillChainDepth.clear();
}

int SpillChainManager::getSpillChainDepth(unsigned reg) const {
    auto it = spillChainDepth.find(reg);
    return it != spillChainDepth.end() ? it->second : 0;
}

bool SpillChainManager::canSpillRegister(unsigned reg) const {
    return getSpillChainDepth(reg) < MAX_SPILL_CHAIN_DEPTH;
}

unsigned SpillChainManager::selectAvailablePhysicalReg(Instruction* inst) {
    // 分析指令中已使用的寄存器
    std::unordered_set<unsigned> usedInInst;

    if (inst) {
        const auto& operands = inst->getOperands();
        for (const auto& operand : operands) {
            if (operand->isReg()) {
                RegisterOperand* regOp =
                    static_cast<RegisterOperand*>(operand.get());
                unsigned regNum = regOp->getRegNum();
                if (regNum < 32) {  // 物理寄存器
                    usedInInst.insert(regNum);
                }
            }
        }
    }

    // 选择第一个不冲突且未使用的物理寄存器
    for (unsigned physReg : availablePhysicalRegs) {
        if (usedPhysicalRegs.find(physReg) == usedPhysicalRegs.end() &&
            usedInInst.find(physReg) == usedInInst.end()) {
            return physReg;
        }
    }

    // 如果没有完全可用的，选择不在当前指令中使用的
    for (unsigned physReg : availablePhysicalRegs) {
        if (usedInInst.find(physReg) == usedInInst.end()) {
            return physReg;
        }
    }

    return 0;  // 没有可用寄存器
}

void SpillChainManager::updateSpillChainDepth(unsigned reg, int depth) {
    spillChainDepth[reg] = depth;
}

}  // namespace riscv64
