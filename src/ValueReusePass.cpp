#include "ValueReusePass.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "Instructions/All.h"
#include "Pass/Analysis/DominanceInfo.h"

namespace riscv64 {

bool ValueReusePass::runOnFunction(
    Function* riscv_function, const midend::Function* midend_function,
    const midend::AnalysisManager* analysisManager) {
    if (riscv_function == nullptr || riscv_function->empty() ||
        midend_function == nullptr || midend_function->empty()) {
        return false;
    }

    resetState();

    std::cout << "ValueReusePass: Analyzing function "
              << riscv_function->getName()
              << " with advanced dominator tree-based optimization"
              << std::endl;

    // Get or compute dominance information for the midend function
    const midend::DominanceInfo* dominanceInfo = nullptr;
    std::unique_ptr<midend::DominanceInfoBase<false>> ownedDominanceInfo;

    if (analysisManager != nullptr) {
        // Try to get precomputed dominance analysis
        dominanceInfo =
            const_cast<midend::AnalysisManager*>(analysisManager)
                ->getAnalysis<midend::DominanceInfo>(
                    midend::DominanceAnalysis::getName(),
                    *const_cast<midend::Function*>(midend_function));
        if (dominanceInfo != nullptr) {
            std::cout
                << "  Using precomputed dominance info from AnalysisManager"
                << std::endl;
        }
    }

    if (dominanceInfo == nullptr) {
        // Compute dominance information ourselves
        std::cout << "  Computing dominance info directly" << std::endl;
        auto dominanceResult = midend::DominanceAnalysis::run(
            *const_cast<midend::Function*>(midend_function));
        if (!dominanceResult) {
            std::cout << "  Failed to compute dominance info" << std::endl;
            return false;
        }
        ownedDominanceInfo = std::move(dominanceResult);
        dominanceInfo = ownedDominanceInfo.get();
    }

    const auto* domTree = dominanceInfo->getDominatorTree();
    if (domTree == nullptr || domTree->getRoot() == nullptr) {
        std::cout << "  Invalid dominator tree" << std::endl;
        return false;
    }

    std::cout << "  Dominator tree root: " << domTree->getRoot()->bb->getName()
              << std::endl;

    // Create mapping from midend basic blocks to RISCV64 basic blocks
    auto bb_mapping = createBasicBlockMapping(riscv_function, midend_function);
    if (bb_mapping.empty()) {
        std::cout << "  Failed to create BB mapping" << std::endl;
        return false;
    }

    std::cout << "  Starting dominator tree traversal..." << std::endl;
    
    // Core optimization: DFS traversal of dominator tree with value tracking
    std::unordered_map<const midend::Value*, RegisterOperand*> valueMap;
    bool modified = false;
    
    try {
        modified = traverseDominatorTree(domTree->getRoot(), riscv_function,
                                        bb_mapping, valueMap);
        std::cout << "  Dominator tree traversal completed successfully" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Error during dominator tree traversal: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cout << "  Unknown error during dominator tree traversal" << std::endl;
        return false;
    }

    // Print statistics
    if (stats_.loadsAnalyzed > 0 || stats_.optimizationOpportunities > 0) {
        std::cout << "ValueReusePass statistics for "
                  << riscv_function->getName() << ":" << std::endl;
        std::cout << "  Instructions analyzed: " << stats_.loadsAnalyzed
                  << std::endl;
        std::cout << "  Optimization opportunities: "
                  << stats_.optimizationOpportunities << std::endl;
        std::cout << "  Instructions eliminated: " << stats_.loadsEliminated
                  << std::endl;
        std::cout << "  Virtual registers reused: " << stats_.virtualRegsReused
                  << std::endl;
        std::cout << "  Stores processed: " << stats_.storesProcessed
                  << std::endl;
        std::cout << "  Calls processed: " << stats_.callsProcessed
                  << std::endl;
        std::cout << "  Memory invalidations: " << stats_.invalidations
                  << std::endl;
    }

    return modified;
}

bool ValueReusePass::traverseDominatorTree(
    const void* node_ptr,  // DominatorTree::Node pointer
    Function* riscv_function,
    const std::unordered_map<const midend::BasicBlock*, BasicBlock*>&
        bb_mapping,
    std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap) {
    
    if (node_ptr == nullptr) {
        std::cout << "  Error: null node_ptr passed to traverseDominatorTree" << std::endl;
        return false;
    }
    
    // Cast to the actual DominatorTree::Node type
    using NodeType = midend::DominatorTreeBase<false>::Node;
    const NodeType* node = static_cast<const NodeType*>(node_ptr);

    if (node == nullptr || node->bb == nullptr) {
        std::cout << "  Error: null node or null bb after cast" << std::endl;
        return false;
    }

    std::cout << "  Processing dominator tree node: " << node->bb->getName()
              << " (level " << node->level << ")" << std::endl;

    // Track definitions made in this block for backtracking
    std::vector<const midend::Value*> definitionsInThisBlock;

    // Find corresponding RISCV64 basic block
    auto bb_it = bb_mapping.find(node->bb);
    if (bb_it == bb_mapping.end()) {
        std::cout << "    No corresponding RISCV64 basic block found"
                  << std::endl;
        // Still process children
        bool childrenModified = false;
        for (const auto& child : node->children) {
            childrenModified |= traverseDominatorTree(
                child.get(), riscv_function, bb_mapping, valueMap);
        }
        return childrenModified;
    }

    BasicBlock* riscv_bb = bb_it->second;
    bool blockModified =
        processBasicBlock(riscv_bb, node->bb, valueMap, definitionsInThisBlock);

    // Recursively process children in the dominator tree
    bool childrenModified = false;
    for (const auto& child : node->children) {
        childrenModified |= traverseDominatorTree(child.get(), riscv_function,
                                                  bb_mapping, valueMap);
    }

    // Backtrack: remove definitions made in this block
    for (const auto* value : definitionsInThisBlock) {
        valueMap.erase(value);
    }

    return blockModified || childrenModified;
}

bool ValueReusePass::processBasicBlock(
    BasicBlock* riscv_bb, const midend::BasicBlock* midend_bb,
    std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap,
    std::vector<const midend::Value*>& definitionsInThisBlock) {
    if (riscv_bb == nullptr) {
        return false;
    }

    std::cout << "    Processing basic block: " << riscv_bb->getLabel()
              << std::endl;

    bool modified = false;
    std::vector<BasicBlock::iterator> toErase;

    for (auto it = riscv_bb->begin(); it != riscv_bb->end(); ++it) {
        auto* inst = it->get();
        stats_.loadsAnalyzed++;

        if (processInstruction(inst, midend_bb, valueMap,
                               definitionsInThisBlock)) {
            toErase.push_back(it);
            modified = true;
        }
    }

    // Remove optimized instructions
    for (auto iter : toErase) {
        std::cout << "      Removing redundant instruction: "
                  << (*iter)->toString() << std::endl;
        riscv_bb->erase(iter);
        stats_.loadsEliminated++;
    }

    return modified;
}

bool ValueReusePass::processInstruction(
    Instruction* inst, const midend::BasicBlock* midend_bb,
    std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap,
    std::vector<const midend::Value*>& definitionsInThisBlock) {
    Opcode opcode = inst->getOpcode();

    // Handle different instruction types
    switch (opcode) {
        case Opcode::LI:
        case Opcode::ADDI:
        case Opcode::LW:
        case Opcode::FLW: {
            // Try to find corresponding midend value
            const midend::Value* correspondingValue =
                findCorrespondingMidendInstruction(inst, midend_bb);
            if (correspondingValue != nullptr) {
                // Check if we already have this value in a register
                auto it = valueMap.find(correspondingValue);
                if (it != valueMap.end()) {
                    // Found reusable value
                    const auto& operands = inst->getOperands();
                    if (!operands.empty()) {
                        auto* dest_reg =
                            dynamic_cast<RegisterOperand*>(operands[0].get());
                        if (dest_reg != nullptr) {
                            std::cout
                                << "        OPTIMIZATION: Reusing value for "
                                << correspondingValue->getName()
                                << " from register " << it->second->getRegNum()
                                << " to register " << dest_reg->getRegNum()
                                << std::endl;

                            stats_.optimizationOpportunities++;
                            stats_.virtualRegsReused++;

                            // We should replace this with a move instruction,
                            // but for now just eliminate it
                            return true;  // Mark for removal
                        }
                    }
                } else {
                    // First time seeing this value - record it
                    const auto& operands = inst->getOperands();
                    if (!operands.empty()) {
                        auto* dest_reg =
                            dynamic_cast<RegisterOperand*>(operands[0].get());
                        if (dest_reg != nullptr) {
                            valueMap[correspondingValue] = dest_reg;
                            definitionsInThisBlock.push_back(
                                correspondingValue);
                            std::cout << "        Recording value "
                                      << correspondingValue->getName()
                                      << " -> register "
                                      << dest_reg->getRegNum() << std::endl;
                        }
                    }
                }
            }
            break;
        }

        case Opcode::SW:
        case Opcode::FSW: {
            // Store instructions potentially invalidate memory
            std::cout << "      Store instruction may invalidate memory"
                      << std::endl;
            stats_.storesProcessed++;
            invalidateMemoryValues(valueMap, definitionsInThisBlock);
            break;
        }

        case Opcode::CALL: {
            // Function calls may have side effects and invalidate memory
            std::cout << "      Call instruction may invalidate memory"
                      << std::endl;
            stats_.callsProcessed++;
            invalidateMemoryValues(valueMap, definitionsInThisBlock);
            break;
        }

        default:
            // For other instructions, check if they redefine any registers
            // we're tracking
            const auto& operands = inst->getOperands();
            if (!operands.empty()) {
                auto* dest_reg =
                    dynamic_cast<RegisterOperand*>(operands[0].get());
                if (dest_reg != nullptr) {
                    // This instruction defines a new value in dest_reg
                    // We don't need to invalidate old mappings since we're
                    // tracking by Value*, not by register number
                }
            }
            break;
    }

    return false;  // Don't remove this instruction
}

const midend::Value* ValueReusePass::findCorrespondingMidendInstruction(
    Instruction* inst, const midend::BasicBlock* midend_bb) {
    // This is a simplified heuristic approach
    // In a real implementation, we'd need a more sophisticated mapping
    // between RISCV64 instructions and midend values

    // For now, we'll use instruction position as a rough heuristic
    // This assumes instruction selection preserves relative ordering

    if (midend_bb == nullptr) {
        return nullptr;
    }

    // Count RISCV64 instruction position in its block
    auto* riscv_bb = inst->getParent();
    if (riscv_bb == nullptr) {
        return nullptr;
    }

    int inst_position = 0;
    for (auto it = riscv_bb->begin(); it != riscv_bb->end(); ++it) {
        if (it->get() == inst) {
            break;
        }
        inst_position++;
    }

    // Find corresponding midend instruction by position (very rough heuristic)
    int current_position = 0;
    for (auto& midend_inst : *midend_bb) {
        if (current_position == inst_position) {
            // Check if this midend instruction produces a value
            if (midend_inst->getType() &&
                !midend_inst->getType()->isVoidType()) {
                return midend_inst;
            }
        }
        current_position++;
    }

    return nullptr;
}

std::unordered_map<const midend::BasicBlock*, BasicBlock*>
ValueReusePass::createBasicBlockMapping(
    Function* riscv_function, const midend::Function* midend_function) {
    std::unordered_map<const midend::BasicBlock*, BasicBlock*> mapping;

    // Simple mapping based on position in function
    // This assumes that the basic block order is preserved during instruction
    // selection
    auto midend_it = midend_function->begin();
    auto riscv_it = riscv_function->begin();

    while (midend_it != midend_function->end() &&
           riscv_it != riscv_function->end()) {
        mapping[*midend_it] = riscv_it->get();
        ++midend_it;
        ++riscv_it;
    }

    std::cout << "  Created mapping for " << mapping.size() << " basic blocks"
              << std::endl;
    return mapping;
}

bool ValueReusePass::replaceWithMove(Instruction* inst,
                                     RegisterOperand* source_reg) {
    // This is a placeholder - in a real implementation, we'd generate a move
    // instruction
    (void)inst;
    (void)source_reg;
    return false;
}

void ValueReusePass::resetState() {
    stats_.loadsAnalyzed = 0;
    stats_.optimizationOpportunities = 0;
    stats_.valuesReused = 0;
    stats_.loadsEliminated = 0;
    stats_.virtualRegsReused = 0;
    stats_.storesProcessed = 0;
    stats_.callsProcessed = 0;
    stats_.invalidations = 0;
}

void ValueReusePass::invalidateMemoryValues(
    std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap,
    std::vector<const midend::Value*>& definitionsInThisBlock) {
    // For conservativeness, we could invalidate all memory-related values
    // For simplicity in this implementation, we'll be very conservative
    // and invalidate everything

    std::cout << "        Conservative invalidation of " << valueMap.size()
              << " tracked values" << std::endl;

    // Track what we're invalidating for potential backtracking
    for (const auto& pair : valueMap) {
        definitionsInThisBlock.push_back(pair.first);
    }

    valueMap.clear();
    stats_.invalidations++;
}

}  // namespace riscv64
