#pragma once

#include <algorithm>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ABI.h"
#include "Instructions/Function.h"
#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"

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

    InterferenceNode(unsigned reg) : regNum(reg) {}
};

// 图着色寄存器分配器
class RegAllocChaitin {
   private:
    static const int NUM_COLORS = 32;  // RISC-V有32个通用寄存器

    // 可用于分配的寄存器 (排除保留寄存器)
    std::vector<unsigned> availableRegs = {
        5,  6,  7,  28, 29, 30, 31,             // t0-t2, t3-t6
        10, 11, 12, 13, 14, 15, 16, 17,         // a0-a7
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27  // s2-s11
    };

    Function* function;
    std::unordered_map<BasicBlock*, LivenessInfo> livenessInfo;
    std::unordered_map<unsigned, std::unique_ptr<InterferenceNode>>
        interferenceGraph;
    std::unordered_map<unsigned, unsigned> virtualToPhysical;
    std::unordered_set<unsigned> spilledRegs;  // 需要溢出的寄存器

   public:
    explicit RegAllocChaitin(Function* func) : function(func) {}

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

    // 重写指令中的寄存器
    void rewriteInstructions();
    void rewriteInstruction(Instruction* inst);

    // 辅助函数
    bool isPhysicalReg(unsigned reg) const;
    unsigned getPhysicalReg(unsigned virtualReg) const;
    std::vector<unsigned> getUsedRegs(const Instruction* inst) const;
    std::vector<unsigned> getDefinedRegs(const Instruction* inst) const;

    // 调试和统计
    void printInterferenceGraph() const;
    void printAllocationResult() const;
};

}  // namespace riscv64