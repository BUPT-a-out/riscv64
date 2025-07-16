#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <iostream>

#include "BasicBlock.h"

namespace riscv64 {

class Function {
   public:
    explicit Function(std::string name) : name(std::move(name)) {}

    void addBasicBlock(std::unique_ptr<BasicBlock> block) {
        basic_blocks.push_back(std::move(block));
    }

    // 提供访问基本块的方法
    BasicBlock* getBasicBlock(size_t index) const {
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

    const std::string& getName() const { return name; }

    // 管理函数的栈帧信息
    void calculateStackFrame() {
        // ... 计算需要多大的栈空间，哪些寄存器需要保存等
    }

    BasicBlock* getBasicBlockByLabel(const std::string& label) const {
        for (const auto& block : basic_blocks) {
            if (block->getLabel() == label) {
                return block.get();
            }
        }
        return nullptr;  // 如果没有找到，返回 nullptr
    }

    std::string toString() const;

   private:
    std::string name;
    std::vector<std::unique_ptr<BasicBlock>> basic_blocks;
    // ... 还可以包含栈帧信息 (StackFrameInfo), 保存的寄存器列表等
};

}  // namespace riscv64