#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "BasicBlock.h"
#include "StackFrameManager.h"

namespace riscv64 {

class StackFrameManager;     // 前向声明
class BasicBlockReordering;  // 前向声明

class Function {
   public:
    // 友元声明，允许BasicBlockReordering访问私有成员
    friend class BasicBlockReordering;

    explicit Function(std::string name)
        : name(std::move(name)),
          stackFrameManager_(std::make_unique<StackFrameManager>(this)) {}

    void addBasicBlock(std::unique_ptr<BasicBlock> block) {
        basic_blocks.push_back(std::move(block));
    }

    // 提供访问基本块的方法
    BasicBlock* getBasicBlockByIndex(size_t index) const {
        return basic_blocks[index].get();
    }

    size_t getBasicBlockCount() const { return basic_blocks.size(); }

    // 迭代器，便于遍历基本块
    using iterator = std::vector<std::unique_ptr<BasicBlock>>::iterator;
    using const_iterator =
        std::vector<std::unique_ptr<BasicBlock>>::const_iterator;

    iterator begin() { return basic_blocks.begin(); }
    iterator end() { return basic_blocks.end(); }
    const_iterator begin() const { return basic_blocks.begin(); }
    const_iterator end() const { return basic_blocks.end(); }
    auto rbegin() { return basic_blocks.rbegin(); }
    auto rend() { return basic_blocks.rend(); }

    bool empty() const { return basic_blocks.empty(); }
    const std::string& getName() const { return name; }

    BasicBlock* getEntryBlock() const {
        if (basic_blocks.empty()) return nullptr;
        return basic_blocks.front().get();
    }

    auto getPostOrder() {
        std::vector<BasicBlock*> postOrder;
        std::unordered_set<BasicBlock*> visited;

        std::function<void(BasicBlock*)> dfs = [&](BasicBlock* bb) {
            if (visited.count(bb)) return;
            visited.insert(bb);
            for (BasicBlock* succ : bb->getSuccessors()) {
                if (succ) {
                    dfs(succ);
                }
            }
            postOrder.push_back(bb);
        };
        auto eb = getEntryBlock();
        if (eb) {
            dfs(eb);
        }
        return postOrder;
    }

    // 管理函数的栈帧信息

    // BasicBlock* getBasicBlockByLabel(const std::string& label) const {
    //     for (const auto& block : basic_blocks) {
    //         if (block->getLabel() == label) {
    //             return block.get();
    //         }
    //     }
    //     return nullptr;  // 如果没有找到，返回 nullptr
    // }
    BasicBlock* getBasicBlock(const midend::BasicBlock* bb) const {
        auto it = bbMap_.find(bb);
        if (it != bbMap_.end()) {
            return it->second;
        }
        throw std::runtime_error("BasicBlock " + bb->toString() +
                                 " not found in function: " + getName());
    }
    void mapBasicBlock(const midend::BasicBlock* bb, BasicBlock* riscvBB) {
        bbMap_[bb] = riscvBB;
    }
    StackFrameManager* getStackFrameManager() const {
        return stackFrameManager_.get();
    }

    unsigned getMaxRegNum() const {
        unsigned M = 0;
        for (const auto& bb_ptr : basic_blocks) {
            const auto* bb = bb_ptr.get();
            for (const auto& inst_ptr : *bb) {
                const auto* inst = inst_ptr.get();
                const auto& ops = inst->getOperands();
                for (const auto& op : ops) {
                    if (op->isReg()) {
                        const auto num = op->getRegNum();
                        M = std::max(M, num);
                    } else if (op->isMem()) {
                        const auto num = static_cast<MemoryOperand*>(op.get())
                                             ->getBaseReg()
                                             ->getRegNum();
                        M = std::max(M, num);
                    }
                }
            }
        }
        return M;
    }

    std::string toString() const;

   private:
    std::string name;
    std::vector<std::unique_ptr<BasicBlock>> basic_blocks;
    std::unique_ptr<StackFrameManager> stackFrameManager_;
    std::unordered_map<const midend::BasicBlock*, BasicBlock*> bbMap_;
};

}  // namespace riscv64