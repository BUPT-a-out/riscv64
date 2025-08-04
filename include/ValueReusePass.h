#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

// Forward declarations
namespace midend {
class Value;
class BasicBlock;
class Function;
class AnalysisManager;
template <bool>
class DominanceInfoBase;
using DominanceInfo = DominanceInfoBase<false>;
template <bool>
class DominatorTreeBase;
using DominatorTree = DominatorTreeBase<false>;
}  // namespace midend

namespace riscv64 {
// Forward declarations - full definitions in Instructions/All.h
class Function;
class BasicBlock;
class Instruction;
class RegisterOperand;

class ValueReusePass {
   public:
    // Statistics
    struct PassStatistics {
        int loadsAnalyzed = 0;
        int optimizationOpportunities = 0;
        int valuesReused = 0;
        int loadsEliminated = 0;
        int virtualRegsReused = 0;
        int storesProcessed = 0;
        int callsProcessed = 0;
        int invalidations = 0;
    };

    ValueReusePass() = default;
    ~ValueReusePass() = default;

    // Main entry point for the pass
    bool runOnFunction(Function* riscv_function,
                       const midend::Function* midend_function,
                       const midend::AnalysisManager* analysisManager);

    // Access to statistics
    const PassStatistics& getStatistics() const { return stats_; }

   private:
    // Core optimization algorithm using dominator tree
    bool traverseDominatorTree(
        const void* node,  // DominatorTree::Node pointer
        Function* riscv_function, const midend::Function* midend_function,
        std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap);

    // Process individual basic blocks
    bool processBasicBlock(
        BasicBlock* riscv_bb, const midend::BasicBlock* midend_bb,
        std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap,
        std::vector<const midend::Value*>& definitionsInThisBlock);

    // Process individual instructions
    bool processInstruction(
        Instruction* inst, const midend::BasicBlock* midend_bb,
        std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap,
        std::vector<const midend::Value*>& definitionsInThisBlock);

    // Helper methods
    bool canReuseValue(Instruction* inst, RegisterOperand* existing_reg);
    const midend::Value* findCorrespondingMidendInstruction(
        Instruction* inst, const midend::BasicBlock* midend_bb);
    const midend::Value* findCorrespondingValue(
        Instruction* inst, const midend::BasicBlock* midend_bb);
    void invalidateMemoryValues(
        std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap,
        std::vector<const midend::Value*>& definitionsInThisBlock);
    std::unordered_map<const midend::BasicBlock*, BasicBlock*>
    createBasicBlockMapping(Function* riscv_function,
                            const midend::Function* midend_function);
    bool replaceWithMove(Instruction* inst, RegisterOperand* source_reg);
    void resetState();

   private:
    PassStatistics stats_;
};

}  // namespace riscv64
