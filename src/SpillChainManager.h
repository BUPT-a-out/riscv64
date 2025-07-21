#pragma once

#include "Instructions/Instruction.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace riscv64 {

// 溢出临时寄存器管理器
class SpillChainManager {
public:
    SpillChainManager(const std::vector<unsigned>& availableRegs);
    
    // 为spill操作分配临时寄存器
    unsigned allocateTempRegister(unsigned spilledReg, Instruction* inst);
    
    // 释放临时寄存器
    void releaseTempRegister(unsigned tempReg);
    
    // 检查是否是临时寄存器
    bool isTempRegister(unsigned reg) const;
    
    // 重置所有临时寄存器状态
    void resetTempRegisters();
    
    // 获取spill链深度
    int getSpillChainDepth(unsigned reg) const;
    
    // 防止递归spill
    bool canSpillRegister(unsigned reg) const;
    
private:
    struct TempRegInfo {
        unsigned physicalReg;  // 分配的物理寄存器
        unsigned originalReg;  // 原始被spill的寄存器
        bool isInUse;         // 是否正在使用
        int chainDepth;       // spill链深度
    };
    
    std::vector<unsigned> availablePhysicalRegs;  // 可用的物理寄存器
    std::unordered_map<unsigned, TempRegInfo> tempRegMap;  // 临时寄存器映射
    std::unordered_set<unsigned> usedPhysicalRegs;  // 已使用的物理寄存器
    std::unordered_map<unsigned, int> spillChainDepth;  // spill链深度
    
    static const int MAX_SPILL_CHAIN_DEPTH = 3;  // 最大spill链深度
    
    // 选择可用的物理寄存器
    unsigned selectAvailablePhysicalReg(Instruction* inst);
    
    // 更新spill链深度
    void updateSpillChainDepth(unsigned reg, int depth);
};

} // namespace riscv64
