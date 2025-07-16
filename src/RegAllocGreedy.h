#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <memory>
#include <optional>
#include <functional>

#include "Instructions/Module.h"
#include "Instructions/Function.h"
#include "Instructions/BasicBlock.h"
#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"

namespace riscv64 {

// 活跃区间表示
struct LiveInterval {
    RegisterOperand* virtReg;           // 虚拟寄存器
    std::vector<std::pair<int, int>> ranges; // 活跃范围 [start, end]
    int weight;                         // 权重 (使用频率)
    bool spillable;                     // 是否可以溢出
    
    LiveInterval(RegisterOperand* reg) : virtReg(reg), weight(0), spillable(true) {}
    
    bool interferesWith(const LiveInterval& other) const;
    bool contains(int point) const;
    int getStartPoint() const { return ranges.empty() ? 0 : ranges.front().first; }
    int getEndPoint() const { return ranges.empty() ? 0 : ranges.back().second; }
};

// 寄存器分配阶段
enum class RegAllocStage {
    RS_New,         // 新创建的区间
    RS_Assign,      // 尝试分配
    RS_Split,       // 尝试分割
    RS_Spill        // 溢出
};

// 物理寄存器信息
struct PhysicalRegister {
    unsigned regNum;
    bool allocated;
    LiveInterval* currentInterval;
    
    PhysicalRegister(unsigned num) : regNum(num), allocated(false), currentInterval(nullptr) {}
};

class RegAllocGreedy {
public:
    explicit RegAllocGreedy(Module& module);
    
    // 主要接口
    bool run();
    
private:
    Module& module_;
    
    // 寄存器相关
    std::vector<PhysicalRegister> physicalRegs_;
    std::unordered_map<RegisterOperand*, std::unique_ptr<LiveInterval>> liveIntervals_;
    std::unordered_map<RegisterOperand*, RegAllocStage> regStages_;
    std::unordered_map<RegisterOperand*, unsigned> virtualToPhysical_;

    std::unordered_map<Instruction*, int> instructionMap_;
    std::unordered_map<RegisterOperand*, std::vector<Instruction*>> defMap_;
    std::unordered_map<RegisterOperand*, std::vector<Instruction*>> useMap_;
    
    // 分配队列
    std::priority_queue<LiveInterval*, std::vector<LiveInterval*>, 
                       std::function<bool(LiveInterval*, LiveInterval*)>> allocQueue_;
    
    // 溢出相关
    std::unordered_set<RegisterOperand*> spilledRegs_;
    std::unordered_map<RegisterOperand*, int> spillSlots_;
    int nextSpillSlot_;
    
    // 统计信息
    int numSpills_;
    int numReloads_;

    int instructionNumber_;
    float csrCost_;
    
    // 初始化
    void initialize();
    void initializePhysicalRegs();
    void calculateLiveIntervals();
    void calculateCSRCosts();
    
    // 主要算法
    void selectOrSplit(LiveInterval* virtReg);
    std::optional<unsigned> tryDirectAssign(LiveInterval* virtReg);
    std::optional<unsigned> tryEviction(LiveInterval* virtReg);
    bool trySplit(LiveInterval* virtReg);
    std::optional<unsigned> tryLastChanceRecoloring(LiveInterval* virtReg);
    void spillInterval(LiveInterval* virtReg);
    
    // 分割策略
    bool tryRegionSplit(LiveInterval* virtReg);
    bool tryBlockSplit(LiveInterval* virtReg);
    bool tryInstructionSplit(LiveInterval* virtReg);
    
    // 辅助函数
    std::vector<LiveInterval*> getInterferingIntervals(LiveInterval* virtReg, unsigned physReg);
    bool canEvictInterval(LiveInterval* interval);
    int calculateEvictionCost(LiveInterval* interval);
    bool canRecolorInterval(LiveInterval* interval, unsigned newReg);
    
    // 分配和释放
    void assignRegister(LiveInterval* virtReg, unsigned physReg);
    void freeRegister(unsigned physReg);
    
    // 溢出代码生成
    void generateSpillCode(LiveInterval* virtReg);
    void insertSpillAtDef(RegisterOperand* virtReg, Instruction* defInst);
    void insertReloadAtUse(RegisterOperand* virtReg, Instruction* useInst);
    
    // 重着色
    bool tryHintRecoloring();
    bool recolorInterval(LiveInterval* interval, unsigned newReg);
    
    // 优化
    void postOptimization();
    void coalesceRegisters();
    bool tryCoalesceCopy(Instruction* copyInst);
    bool canEliminateCopy(Instruction* copyInst);
    
    // 工具函数
    bool hasAvailableReg() const;
    unsigned findBestPhysicalReg(LiveInterval* virtReg) const;
    std::vector<unsigned> getAllocationOrder(LiveInterval* virtReg) const;
    
    // 调试和统计
    void reportStatistics();
    void verifyAllocation();
    
    // 需要实现的辅助类和函数 (在实现文件中)
    void updateInstruction(Instruction* inst, RegisterOperand* oldReg, unsigned newPhysReg);
    int getInstructionNumber(Instruction* inst) const;
    std::vector<Instruction*> getUsesAndDefs(RegisterOperand* virtReg) const;
    bool isDefinition(Instruction* inst, RegisterOperand* virtReg) const;
    bool isUse(Instruction* inst, RegisterOperand* virtReg) const;
};

// 辅助函数
namespace RegAllocUtils {
    // 计算两个活跃区间是否冲突
    bool interferes(const LiveInterval& a, const LiveInterval& b);
    
    // 获取寄存器的分配顺序 (根据调用约定)
    std::vector<unsigned> getRegisterAllocationOrder();
    
    // 判断是否是调用者保存寄存器
    bool isCallerSaved(unsigned physReg);
    
    // 判断是否是被调用者保存寄存器
    bool isCalleeSaved(unsigned physReg);
    
    // 获取寄存器类别 (整数/浮点)
    enum class RegClass { Integer, Float };
    RegClass getRegisterClass(unsigned physReg);
}

} // namespace riscv64