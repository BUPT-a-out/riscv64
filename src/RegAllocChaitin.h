#pragma once

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Instructions/Function.h"
#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"
#include "SpillChainManager.h"

namespace riscv64 {

// 活跃性分析结果
struct LivenessInfo {
    std::unordered_set<unsigned> liveIn;   // 基本块入口处的活跃变量
    std::unordered_set<unsigned> liveOut;  // 基本块出口处的活跃变量
    std::unordered_set<unsigned> def;      // 基本块内定义的变量
    std::unordered_set<unsigned> use;      // 基本块内使用的变量
};

// 冲突图中的节点
struct InterferenceNode {
    unsigned regNum;
    std::unordered_set<unsigned> neighbors;  // 邻接节点
    int color = -1;                          // 分配的颜色（物理寄存器）
    bool isPrecolored = false;               // 是否已经预着色（物理寄存器）

    unsigned coalesceParent;  // 合并的代表元

    InterferenceNode(unsigned reg) : regNum(reg) {}
};

// 合并信息结构
struct CoalesceInfo {
    unsigned src;
    unsigned dst;
    bool canCoalesce;
    int priority;  // 合并优先级，基于指令频次等因素
};

// 图着色寄存器分配器
class RegAllocChaitin {
   private:
    static const int NUM_COLORS = 32;  // RISC-V有32个通用寄存器

    // TODO: do stuff
    int assigningFloat = false;

    // 可用于分配的寄存器 (排除保留寄存器)
    std::vector<unsigned> availableRegs;

    Function* function;
    std::unordered_map<BasicBlock*, LivenessInfo> livenessInfo;
    std::unordered_map<unsigned, std::unique_ptr<InterferenceNode>>
        interferenceGraph;
    std::unordered_map<unsigned, unsigned> virtualToPhysical;
    std::unordered_set<unsigned> spilledRegs;  // 需要溢出的寄存器

    std::unordered_map<unsigned, std::unordered_set<unsigned>>
        physicalConstraints;
    void addPhysicalConstraint(unsigned virtualReg, unsigned physicalReg);

    // 强约束：必须分配到指定的物理寄存器
    std::unordered_map<unsigned, unsigned> strongConstraints;
    // 保留的物理寄存器：不能分配给任何虚拟寄存器
    std::unordered_set<unsigned> reservedPhysicalRegs;

    std::vector<CoalesceInfo> coalesceCandidates;
    std::unordered_set<unsigned> coalescedRegs;
    std::unordered_map<unsigned, unsigned> coalesceMap;  // 映射到代表元

    std::unique_ptr<SpillChainManager> spillChainManager;

    StackFrameManager stackManager;

    // 维护度数缓存，避免重复计算
    std::unordered_map<unsigned, int> degreeCache;
    void initializeDegreeCache();
    void updateDegreeAfterRemoval(unsigned removedReg);
    void updateDegreeAfterCoalesce(unsigned merged, unsigned eliminated);
    void invalidateDegreeCache(unsigned reg);
    int getCachedDegree(unsigned reg);
    void clearDegreeCache() { degreeCache.clear(); }

   public:
    explicit RegAllocChaitin(Function* func, bool assigningFloat=false)
        : assigningFloat(assigningFloat), function(func), stackManager(StackFrameManager(function)) {}

    // 主要的寄存器分配接口
    void allocateRegisters();

   private:
    // 活跃性分析
    void computeLiveness();
    void computeDefUse(BasicBlock* bb, LivenessInfo& info);

    // 构建冲突图
    void buildInterferenceGraph();
    void addInterference(unsigned reg1, unsigned reg2);

    // 图着色算法
    bool colorGraph();
    std::vector<unsigned> getSimplificationOrder();
    bool attemptColoring(const std::vector<unsigned>& order);

    // 溢出处理
    void handleSpills();
    std::vector<unsigned> selectSpillCandidates();
    void insertSpillCode(unsigned reg);
    void handleLoadStoreSpill(Instruction* inst, unsigned spilledReg,
                              int frameIndex, BasicBlock::iterator& it,
                              BasicBlock* bb);

    void insertLoadStoreReload(Instruction* inst, unsigned spilledReg,
                               int frameIndex, BasicBlock::iterator& it,
                               BasicBlock* bb);
    void insertLoadStoreSpill(Instruction* inst, unsigned spilledReg,
                              int frameIndex, BasicBlock::iterator& it,
                              BasicBlock* bb);
    void updateLoadStoreOperands(Instruction* inst, unsigned oldReg,
                                 unsigned addrTempReg, unsigned dataTempReg);
    void updateLoadStoreOperands(Instruction* inst, unsigned oldReg,
                                 unsigned addrTempReg);

    // 重写指令中的寄存器
    void rewriteInstructions();
    void rewriteInstruction(Instruction* inst);
    void rewriteOperand(MachineOperand* operand);
    unsigned getFinalCoalescedReg(unsigned reg);
    void updateRegisterInInstruction(Instruction* inst, unsigned oldReg,
                                     unsigned newReg);

    // 寄存器合并方法
    void performCoalescing();
    void identifyCoalesceCandidates();
    bool canCoalesce(unsigned src, unsigned dst);
    void coalesceRegisters(unsigned src, unsigned dst);
    unsigned findCoalesceRoot(unsigned reg);
    void unionCoalesce(unsigned reg1, unsigned reg2);
    void updateInterferenceAfterCoalesce(unsigned merged, unsigned eliminated);
    void removeCoalescedCopies();
    void removeRebundantCopies();

    bool canCoalesceWithABI(unsigned src, unsigned dst) const;
    bool crossesFunctionCall(unsigned src, unsigned dst) const;

    // 计算合并权重
    int calculateCoalescePriority(unsigned src, unsigned dst, BasicBlock* bb,
                                  Instruction* inst);
    int getBasicBlockFrequency(BasicBlock* bb);
    int calculateABIPriority(unsigned src, unsigned dst) const;
    int getRegisterUsageCount(unsigned reg);
    int getRegisterDegree(unsigned reg);
    int calculateLifetimeOverlap(unsigned src, unsigned dst);
    int getRegisterPressure(BasicBlock* bb);
    int calculatePhysicalRegPreference(unsigned src, unsigned dst);

    // 辅助函数
    bool isPhysicalReg(unsigned reg) const;
    unsigned getPhysicalReg(unsigned virtualReg) const;

    // ABI约束
    void initializeABIConstraints();
    void setFunctionSpecificConstraints();
    void setParameterConstraints();
    void setReturnValueConstraints();
    void setCallSiteConstraints();
    void setPreCallConstraints(BasicBlock* bb, Instruction* callInst);
    void setPostCallConstraints(BasicBlock* bb, Instruction* callInst);

    void addStrongPhysicalConstraint(unsigned virtualReg,
                                     unsigned physicalReg) {
        strongConstraints[virtualReg] = physicalReg;
    }

    void addReservedPhysicalReg(unsigned physicalReg) {
        reservedPhysicalRegs.insert(physicalReg);
    }

    std::vector<unsigned> getABIPreferredRegs(unsigned virtualReg) const;
    bool isUsedAsArgument(unsigned virtualReg) const;
    bool isUsedAcrossCalls(unsigned virtualReg) const;

    std::vector<unsigned> getDefinedRegs(Instruction* inst) const;
    std::vector<unsigned> getUsedRegs(Instruction* inst) const;


    // 调试和统计
    void printInterferenceGraph() const;
    void printAllocationResult() const;
    void printCoalesceResult() const;
};

}  // namespace riscv64