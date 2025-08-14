#pragma once

#include <queue>
#include <set>

#include "Instructions/Function.h"
#include "RAGreedy/LiveIntervals.h"
#include "RAGreedy/LiveRegMatrix.h"
#include "RAGreedy/VirtRegMap.h"

namespace riscv64 {
class RegAllocGreedy {
   private:
    Function *function;
    LiveIntervals *LIS;
    VirtRegMap *VRM;
    LiveRegMatrix *Matrix;
    bool assigningFloat = false;

    unsigned stackSlotNum_ = 0;

    std::priority_queue<std::pair<unsigned, LiveInterval*>> Queue;  // 分配队列
    std::vector<LiveInterval *> unassignedRegs;                // 未分配的寄存器

   public:
    explicit RegAllocGreedy(Function *func, LiveIntervals *LIS, bool assigningFloat = false)
        : function(func), LIS(LIS), assigningFloat(assigningFloat) {};
    void run(void);
    VirtRegMap* getVRM() { return VRM; }

   private:
    // 初始化方法
    void init();
    void seedLiveRegs();

    // 寄存器分配辅助方法
    std::vector<unsigned> getAllocationOrder(const LiveInterval &VirtReg);
    bool checkInterference(const LiveInterval &VirtReg, unsigned PhysReg);
    unsigned calculateSpillCost(const LiveInterval &VirtReg);

    // 分割相关方法
    bool canSplit(const LiveInterval &VirtReg);
    std::vector<LiveInterval *> splitInterval(const LiveInterval &VirtReg);

    // 驱逐相关方法
    std::vector<LiveInterval *> getEvictionCandidates(unsigned PhysReg);
    bool shouldEvict(const LiveInterval &Evictee, const LiveInterval &Evictor);

    void allocateRegisters();
    void selectOrSplit(LiveInterval *LI, std::vector<unsigned> regs);

    void enqueue(LiveInterval *li);
    LiveInterval *dequeue();

    unsigned tryAssign(const LiveInterval &,
                       std::vector<unsigned> allocationOrder,
                       std::vector<unsigned> &newVRegs,
                       std::set<unsigned> fixedRegs);

    unsigned tryEvict(const LiveInterval &,
                      std::vector<unsigned> allocationOrder,
                      std::vector<unsigned> &newVRegs, unsigned costPerUseLimit,
                      std::set<unsigned> fixedRegs);

    unsigned tryRegionSplit(const LiveInterval &,
                            std::vector<unsigned> allocationOrder,
                            std::vector<unsigned> &newVRegs);

    unsigned tryBlockSplit(const LiveInterval &,
                           std::vector<unsigned> allocationOrder,
                           std::vector<unsigned> &newVRegs);

    unsigned tryLocalSplit(const LiveInterval &,
                           std::vector<unsigned> allocationOrder,
                           std::vector<unsigned> &newVRegs);

    unsigned tryLastChanceRecoloring(
        const LiveInterval &VirtReg, std::vector<unsigned> allocationOrder,
        std::vector<unsigned> &NewVRegs,
        std::unordered_set<unsigned> &FixedRegisters,
        std::vector<std::pair<const LiveInterval *, unsigned>> &RecolorStack,
        unsigned Depth);

    bool isUsedAsArgument(unsigned virtualReg) const;
    bool isUsedAcrossCalls(unsigned virtualReg) const;

   public:
    void print(std::ostream &OS) const;
};

}  // namespace riscv64