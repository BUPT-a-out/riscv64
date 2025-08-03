#include "BasicBlockReordering.h"

#include <algorithm>
#include <iostream>
#include <list>
#include <map>

#include "Instructions/Instruction.h"

namespace riscv64 {

BasicBlockReordering::BasicBlockReordering(Function* function)
    : function_(function) {}

void BasicBlockReordering::run() {
    if (!function_ || function_->empty()) {
        std::cout << "BasicBlockReordering: Skipping empty function"
                  << std::endl;
        return;
    }

    std::cout << "\n=== Basic Block Reordering for function: "
              << function_->getName() << " ===" << std::endl;

    printOriginalLayout();
    optimizeBlockLayout();
    removeRedundantJumps();
    printOptimizedLayout();
    printStatistics();
}

void BasicBlockReordering::optimizeBlockLayout() {
    std::cout << "Starting greedy chain-building algorithm..." << std::endl;

    // 收集所有基本块到临时vector（因为我们需要修改function的basic_blocks）
    std::vector<BasicBlock*> all_blocks;
    for (auto& bb_ptr : *function_) {
        all_blocks.push_back(bb_ptr.get());
    }

    if (all_blocks.empty()) {
        return;
    }

    // 1. 初始化
    std::vector<BasicBlock*> new_layout;
    std::set<BasicBlock*> unplaced_blocks(all_blocks.begin(), all_blocks.end());

    // 2. 主循环：构建所有链
    while (!unplaced_blocks.empty()) {
        // 2a. 选择链的起点
        BasicBlock* current_block;
        if (new_layout.empty()) {
            // 第一次选择入口块
            current_block = function_->getEntryBlock();
            if (!current_block ||
                unplaced_blocks.find(current_block) == unplaced_blocks.end()) {
                // 入口块不存在或已被放置，选择任意一个
                current_block = *unplaced_blocks.begin();
            }
        } else {
            // 后续选择任意未放置的块
            current_block = *unplaced_blocks.begin();
        }

        std::cout << "Starting new chain with block: "
                  << current_block->getLabel() << std::endl;

        // 2b. 内循环：延伸当前链
        while (current_block &&
               unplaced_blocks.find(current_block) != unplaced_blocks.end()) {
            // 2b.i 将当前块从未放置集合移除，添加到新布局
            unplaced_blocks.erase(current_block);
            new_layout.push_back(current_block);
            std::cout << "  Added block: " << current_block->getLabel()
                      << " to layout" << std::endl;

            // 2b.ii 寻找最佳后继
            BasicBlock* next_block =
                findBestSuccessor(current_block, unplaced_blocks);

            if (next_block) {
                std::cout << "  Next block in chain: " << next_block->getLabel()
                          << std::endl;
                current_block = next_block;
            } else {
                std::cout << "  Chain ended (no suitable successor)"
                          << std::endl;
                break;
            }
        }
    }

    // 3. 重新构建function的basic_blocks
    // 直接访问friend class的私有成员
    auto& func_blocks = function_->basic_blocks;

    // 创建一个映射来保存unique_ptr
    std::map<BasicBlock*, std::unique_ptr<BasicBlock>> block_map;

    // 将所有blocks移动到临时map
    for (auto& bb_ptr : func_blocks) {
        BasicBlock* raw_ptr = bb_ptr.get();
        block_map[raw_ptr] = std::move(bb_ptr);
    }

    // 清空原vector
    func_blocks.clear();

    // 按新顺序重新添加
    for (BasicBlock* bb : new_layout) {
        auto it = block_map.find(bb);
        if (it != block_map.end()) {
            func_blocks.push_back(std::move(it->second));
        }
    }

    std::cout << "Block layout optimization completed. New order: ";
    for (size_t i = 0; i < new_layout.size(); ++i) {
        std::cout << new_layout[i]->getLabel();
        if (i < new_layout.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;
}

BasicBlock* BasicBlockReordering::findBestSuccessor(
    BasicBlock* current_block,
    const std::set<BasicBlock*>& unplaced_blocks) const {
    if (current_block->getSuccessors().empty()) {
        return nullptr;
    }

    // 优先级1：消除无条件跳转
    BasicBlock* jump_target = nullptr;
    if (isUnconditionalJump(current_block, jump_target)) {
        if (unplaced_blocks.find(jump_target) != unplaced_blocks.end()) {
            // 检查jump_target是否只有一个前驱（当前块）
            const auto& predecessors = jump_target->getPredecessors();
            if (predecessors.size() == 1 && predecessors[0] == current_block) {
                std::cout
                    << "    Priority 1: Eliminating unconditional jump to "
                    << jump_target->getLabel() << std::endl;
                return jump_target;
            }
        }
    }

    // 优先级2：优化条件分支（选择fallthrough路径）
    BasicBlock* jump_tgt = nullptr;
    BasicBlock* fallthrough_target = nullptr;
    if (isConditionalBranch(current_block, jump_tgt, fallthrough_target)) {
        if (fallthrough_target &&
            unplaced_blocks.find(fallthrough_target) != unplaced_blocks.end()) {
            std::cout << "    Priority 2: Following fallthrough path to "
                      << fallthrough_target->getLabel() << std::endl;
            return fallthrough_target;
        }
    }

    // 优先级3：任意合法后继
    for (BasicBlock* successor : current_block->getSuccessors()) {
        if (successor &&
            unplaced_blocks.find(successor) != unplaced_blocks.end()) {
            std::cout << "    Priority 3: Arbitrary successor "
                      << successor->getLabel() << std::endl;
            return successor;
        }
    }

    return nullptr;
}

bool BasicBlockReordering::isUnconditionalJump(BasicBlock* block,
                                               BasicBlock*& target) const {
    if (block->size() == 0) return false;

    // 获取最后一条指令
    auto it = block->rbegin();
    if (it == block->rend()) return false;

    const Instruction* last_inst = it->get();
    if (last_inst->getOpcode() == Opcode::J) {
        // 对于 J 指令，目标通常在第一个操作数中
        const auto& operands = last_inst->getOperands();
        if (!operands.empty()) {
            // 我们需要从操作数中提取目标标签并找到对应的基本块
            // 这里我们通过successors来查找
            const auto& successors = block->getSuccessors();
            if (successors.size() == 1) {
                target = successors[0];
                return true;
            }
        }
    }

    return false;
}

bool BasicBlockReordering::isConditionalBranch(
    BasicBlock* block, BasicBlock*& jump_target,
    BasicBlock*& fallthrough_target) const {
    if (block->size() == 0) return false;

    auto it = block->rbegin();
    if (it == block->rend()) return false;

    const Instruction* last_inst = it->get();
    Opcode opcode = last_inst->getOpcode();

    // 检查是否是条件分支指令
    bool is_conditional = (opcode == Opcode::BEQ || opcode == Opcode::BNE ||
                           opcode == Opcode::BLT || opcode == Opcode::BGE ||
                           opcode == Opcode::BLTU || opcode == Opcode::BGEU ||
                           opcode == Opcode::BEQZ || opcode == Opcode::BNEZ ||
                           opcode == Opcode::BLEZ || opcode == Opcode::BGEZ ||
                           opcode == Opcode::BLTZ || opcode == Opcode::BGTZ);

    if (is_conditional) {
        const auto& successors = block->getSuccessors();
        if (successors.size() == 2) {
            // 按约定：successors[0] 是跳转目标，successors[1] 是fallthrough目标
            jump_target = successors[0];
            fallthrough_target = successors[1];
            return true;
        }
    }

    return false;
}

void BasicBlockReordering::removeRedundantJumps() {
    std::cout << "Removing redundant jumps..." << std::endl;

    int removed_jumps = 0;
    auto& blocks = function_->basic_blocks;

    for (size_t i = 0; i < blocks.size() - 1; ++i) {
        BasicBlock* current = blocks[i].get();
        BasicBlock* next = blocks[i + 1].get();

        if (current->size() == 0) continue;

        auto it = current->rbegin();
        if (it == current->rend()) continue;

        Instruction* last_inst = it->get();
        if (last_inst->getOpcode() == Opcode::J) {
            // 检查这个J指令是否跳转到紧邻的下一个块
            const auto& successors = current->getSuccessors();
            if (successors.size() == 1 && successors[0] == next) {
                std::cout << "  Removing redundant jump from "
                          << current->getLabel() << " to " << next->getLabel()
                          << std::endl;

                // 删除这条指令
                auto forward_it = current->end();
                --forward_it;  // 指向最后一个元素
                current->erase(forward_it);
                removed_jumps++;
            }
        }
    }

    std::cout << "Removed " << removed_jumps << " redundant jump instructions"
              << std::endl;
}

void BasicBlockReordering::printOriginalLayout() const {
    std::cout << "Original layout: ";
    bool first = true;
    for (const auto& bb : *function_) {
        if (!first) std::cout << " -> ";
        std::cout << bb->getLabel();
        first = false;
    }
    std::cout << std::endl;
}

void BasicBlockReordering::printOptimizedLayout() const {
    std::cout << "Optimized layout: ";
    bool first = true;
    for (const auto& bb : *function_) {
        if (!first) std::cout << " -> ";
        std::cout << bb->getLabel();
        first = false;
    }
    std::cout << std::endl;
}

void BasicBlockReordering::printStatistics() const {
    std::cout << "Block reordering statistics:" << std::endl;
    std::cout << "  Total blocks: " << function_->getBasicBlockCount()
              << std::endl;

    // 统计跳转指令数量
    int jump_count = 0;
    int conditional_branch_count = 0;

    for (const auto& bb : *function_) {
        if (bb->size() > 0) {
            auto it = bb->rbegin();
            if (it != bb->rend()) {
                const Instruction* last_inst = it->get();
                Opcode opcode = last_inst->getOpcode();

                if (opcode == Opcode::J) {
                    jump_count++;
                } else if (opcode == Opcode::BEQ || opcode == Opcode::BNE ||
                           opcode == Opcode::BLT || opcode == Opcode::BGE ||
                           opcode == Opcode::BLTU || opcode == Opcode::BGEU ||
                           opcode == Opcode::BEQZ || opcode == Opcode::BNEZ ||
                           opcode == Opcode::BLEZ || opcode == Opcode::BGEZ ||
                           opcode == Opcode::BLTZ || opcode == Opcode::BGTZ) {
                    conditional_branch_count++;
                }
            }
        }
    }

    std::cout << "  Unconditional jumps: " << jump_count << std::endl;
    std::cout << "  Conditional branches: " << conditional_branch_count
              << std::endl;
}

}  // namespace riscv64
