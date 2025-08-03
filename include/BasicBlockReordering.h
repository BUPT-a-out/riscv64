#pragma once

#include <memory>
#include <set>
#include <vector>

#include "Instructions/BasicBlock.h"
#include "Instructions/Function.h"

namespace riscv64 {

class BasicBlockReordering {
   public:
    explicit BasicBlockReordering(Function* function);
    ~BasicBlockReordering() = default;

    // 禁止复制，允许移动
    BasicBlockReordering(const BasicBlockReordering&) = delete;
    BasicBlockReordering& operator=(const BasicBlockReordering&) = delete;
    BasicBlockReordering(BasicBlockReordering&&) = default;
    BasicBlockReordering& operator=(BasicBlockReordering&&) = default;

    // 主要的优化函数
    void run();

   private:
    Function* function_;

    // 核心算法：基于贪心的链构建算法
    void optimizeBlockLayout();

    // 辅助函数
    bool isUnconditionalJump(BasicBlock* block, BasicBlock*& target) const;
    bool isConditionalBranch(BasicBlock* block, BasicBlock*& jump_target,
                             BasicBlock*& fallthrough_target) const;
    BasicBlock* findBestSuccessor(
        BasicBlock* current_block,
        const std::set<BasicBlock*>& unplaced_blocks) const;
    void removeRedundantJumps();

    // 调试输出
    void printOriginalLayout() const;
    void printOptimizedLayout() const;
    void printStatistics() const;
};

}  // namespace riscv64
