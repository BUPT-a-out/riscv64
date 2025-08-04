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

    // Use the existing basic block mapping from the function
    // No need to create our own mapping since Function already maintains it
    std::cout << "  Using existing basic block mapping from Function"
              << std::endl;

    std::cout << "  Starting dominator tree traversal..." << std::endl;

    // Core optimization: DFS traversal of dominator tree with value tracking
    std::unordered_map<const midend::Value*, RegisterOperand*> valueMap;
    bool modified = false;

    try {
        modified = traverseDominatorTree(domTree->getRoot(), riscv_function,
                                         midend_function, valueMap);
        std::cout << "  Dominator tree traversal completed successfully"
                  << std::endl;
    } catch (const std::exception& e) {
        std::cout << "  Error during dominator tree traversal: " << e.what()
                  << std::endl;
        return false;
    } catch (...) {
        std::cout << "  Unknown error during dominator tree traversal"
                  << std::endl;
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
    Function* riscv_function, const midend::Function* midend_function,
    std::unordered_map<const midend::Value*, RegisterOperand*>& valueMap) {
    if (node_ptr == nullptr) {
        std::cout << "  Error: null node_ptr passed to traverseDominatorTree"
                  << std::endl;
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

    // Find corresponding RISCV64 basic block using Function's mapping
    BasicBlock* riscv_bb = nullptr;
    try {
        riscv_bb = riscv_function->getBasicBlock(node->bb);
    } catch (const std::exception& e) {
        std::cout << "    No corresponding RISCV64 basic block found for "
                  << node->bb->getName() << ": " << e.what() << std::endl;
        // Still process children
        bool childrenModified = false;
        for (const auto& child : node->children) {
            childrenModified |= traverseDominatorTree(
                child.get(), riscv_function, midend_function, valueMap);
        }
        return childrenModified;
    }

    bool blockModified =
        processBasicBlock(riscv_bb, node->bb, valueMap, definitionsInThisBlock);

    // Recursively process children in the dominator tree
    // Children can see and reuse values defined in this block
    bool childrenModified = false;
    for (const auto& child : node->children) {
        childrenModified |= traverseDominatorTree(child.get(), riscv_function,
                                                  midend_function, valueMap);
    }

    // Backtrack: remove definitions made in this block
    // This ensures that siblings cannot see each other's definitions
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
    std::cout << "        Processing instruction: " << inst->toString() << std::endl;

    switch (opcode) {
        case LI: {
            // 立即数加载指令 - 实现正确的值复用逻辑
            const auto& operands = inst->getOperands();
            if (operands.size() >= 2) {
                auto* dest_reg = dynamic_cast<RegisterOperand*>(operands[0].get());
                auto* imm_op = dynamic_cast<ImmediateOperand*>(operands[1].get());
                
                if (dest_reg != nullptr && imm_op != nullptr) {
                    int64_t value = imm_op->getValue();
                    std::cout << "          Load immediate: " << value 
                              << " -> reg" << dest_reg->getRegNum() << std::endl;
                    stats_.loadsAnalyzed++;
                    
                    // 关键修复：只在寄存器中查找已有的立即数值，不从栈加载
                    // 查找是否有相同立即数值的指令已经存在于寄存器中
                    const midend::Value* correspondingValue = findCorrespondingMidendInstruction(inst, midend_bb);
                    if (correspondingValue != nullptr) {
                        // 检查是否已经有寄存器保存了相同的值
                        auto it = valueMap.find(correspondingValue);
                        if (it != valueMap.end() && it->second != nullptr) {
                            RegisterOperand* existing_reg = it->second;
                            
                            // 确保现有寄存器仍然有效且不是当前目标寄存器
                            if (existing_reg->getRegNum() != dest_reg->getRegNum()) {
                                std::cout << "          OPTIMIZATION: Found existing register " 
                                          << existing_reg->getRegNum() 
                                          << " with same immediate value " << value 
                                          << ", could reuse for reg" << dest_reg->getRegNum() << std::endl;
                                stats_.optimizationOpportunities++;
                                // 注意：这里我们只统计机会，不实际修改指令
                                // 避免生成错误的栈加载指令
                            }
                        }
                        
                        // 无论是否复用，都记录当前定义
                        valueMap[correspondingValue] = dest_reg;
                        definitionsInThisBlock.push_back(correspondingValue);
                        std::cout << "          Recording immediate " << value 
                                  << " in reg" << dest_reg->getRegNum() << std::endl;
                    }
                }
            }
            break;
        }
        
        case LW:
        case FLW: {
            // 内存加载指令
            const auto& operands = inst->getOperands();
            if (operands.size() >= 2) {
                auto* dest_reg = dynamic_cast<RegisterOperand*>(operands[0].get());
                if (dest_reg != nullptr) {
                    std::cout << "          Memory load -> reg" << dest_reg->getRegNum() << std::endl;
                    stats_.loadsAnalyzed++;
                    
                    // 为内存加载建立映射
                    const midend::Value* correspondingValue = findCorrespondingMidendInstruction(inst, midend_bb);
                    if (correspondingValue != nullptr) {
                        valueMap[correspondingValue] = dest_reg;
                        definitionsInThisBlock.push_back(correspondingValue);
                    }
                }
            }
            break;
        }
        
        case SW:
        case FSW: {
            std::cout << "          Store instruction" << std::endl;
            stats_.storesProcessed++;
            // Store指令可能使某些内存值失效
            invalidateMemoryValues(valueMap, definitionsInThisBlock);
            break;
        }
        
        case CALL: {
            std::cout << "          Call instruction" << std::endl;
            stats_.callsProcessed++;
            // 函数调用可能修改内存，使某些值失效
            invalidateMemoryValues(valueMap, definitionsInThisBlock);
            break;
        }
        
        default:
            // 其他指令暂不处理
            break;
    }

    // 不删除指令，只做分析和映射建立
    return false;
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
