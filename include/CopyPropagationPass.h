#pragma once

#include <map>
#include <set>
#include <unordered_set>
#include <variant>
#include <vector>

#include "Instructions/Function.h"
#include "flat_hash_map/unordered_map.hpp"

namespace riscv64 {

// 表示一个复写关系 <dest, src>，即 mv dest, src
struct CopyPair {
   private:
    unsigned dest_;  // 目标寄存器
    unsigned src_;   // 源寄存器

   public:
    explicit CopyPair(unsigned destination, unsigned source)
        : dest_(destination), src_(source) {}

    [[nodiscard]] auto operator<(const CopyPair& other) const -> bool {
        if (dest_ != other.dest_) return dest_ < other.dest_;
        return src_ < other.src_;
    }

    [[nodiscard]] auto operator==(const CopyPair& other) const -> bool {
        return dest_ == other.dest_ && src_ == other.src_;
    }

    [[nodiscard]] auto involves(unsigned reg) const -> bool {
        return dest_ == reg || src_ == reg;
    }

    [[nodiscard]] auto dest() const -> unsigned { return dest_; }
    [[nodiscard]] auto src() const -> unsigned { return src_; }
};

// 复写信息集合
using CopyInfo = std::set<CopyPair>;

using GenOrKill = std::variant<unsigned, CopyPair>;
static const GenOrKill currentGenOrKill = static_cast<unsigned>(1);

class CopyPropagationPass {
   private:
    // 每个基本块的gen和kill集合
    ska::unordered_map<BasicBlock*, CopyInfo> gen;
    ska::unordered_map<BasicBlock*, std::unordered_set<unsigned>> kill;

    // 每个基本块的CopyIn和CopyOut集合
    ska::unordered_map<BasicBlock*, CopyInfo> copyIn;
    ska::unordered_map<BasicBlock*, CopyInfo> copyOut;

    // 标记要删除的指令
    std::vector<Instruction*> instructionsToRemove;

    // 阶段1：计算gen和kill集合
    void computeGenKill(Function* function);
    void processBasicBlock(BasicBlock* basicBlock);

    // 阶段2：迭代数据流分析
    void iterativeDataFlow(Function* function);
    static auto intersectCopyInfo(const std::vector<CopyInfo>& sets)
        -> CopyInfo;
    static auto applyCopyKill(const CopyInfo& copySet,
                              const std::unordered_set<unsigned>& killSet)
        -> CopyInfo;

    // 阶段3：代码转换
    void transformCode(Function* function);
    void transformBasicBlock(BasicBlock* basicBlock);

    // 辅助方法
    static auto isCopyInstruction(Instruction* inst) -> bool;
    static auto isVirtualRegister(unsigned reg) -> bool;
    void markInstructionForRemoval(Instruction* inst);
    void removeMarkedInstructions(Function* function);

    static constexpr unsigned VIRTUAL_REG_START = 100;

   public:
    void runOnFunction(Function* function);
    void clear();
};

}  // namespace riscv64