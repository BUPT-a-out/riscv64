#pragma once

#include <list>  // 使用链表便于指令的插入和删除
#include <memory>
#include <string>

#include "Instruction.h"

namespace riscv64 {

class Function;  // 前向声明

class BasicBlock {
   public:
    BasicBlock(Function* parentFunc, const std::string& label)
        : label(label), parent(parentFunc) {}

    void addInstruction(std::unique_ptr<Instruction> inst) {
        inst->setParent(this);
        instructions.push_back(std::move(inst));
    }

    // 提供访问指令的方法
    Instruction* getInstruction(size_t index) const {
        auto it = instructions.begin();
        std::advance(it, index);
        return it->get();
    }

    size_t getInstructionCount() const { return instructions.size(); }

    auto* getParent() const { return parent; }

    // 迭代器，便于遍历指令
    using iterator = std::list<std::unique_ptr<Instruction>>::iterator;
    using const_iterator =
        std::list<std::unique_ptr<Instruction>>::const_iterator;

    iterator begin() { return instructions.begin(); }
    iterator end() { return instructions.end(); }
    const_iterator begin() const { return instructions.begin(); }
    const_iterator end() const { return instructions.end(); }

    iterator erase(iterator it) { return instructions.erase(it); }
    iterator insert(const_iterator pos, std::unique_ptr<Instruction> inst) {
        inst->setParent(this);
        return instructions.insert(pos, std::move(inst));
    }

    auto size() const { return instructions.size(); }

    const std::string& getLabel() const { return label; }

    // CFG (控制流图) 信息 - 使用原始指针，因为这些是弱引用关系
    void addSuccessor(BasicBlock* succ) { successors.push_back(succ); }
    void addPredecessor(BasicBlock* pred) { predecessors.push_back(pred); }
    auto getSuccessors() {
        return successors;
    }
    auto getPredecessors() {
        return predecessors;
    }

    std::string toString() const;

   private:
    std::string label;  // 例如 ".LBB0_1"
    std::list<std::unique_ptr<Instruction>>
        instructions;  // 使用 unique_ptr 管理指令

    // 指向其所在的函数 - 弱引用，不拥有所有权
    Function* parent;

    // 控制流图 (CFG) 的边 - 弱引用，不拥有所有权
    std::vector<BasicBlock*> successors;
    std::vector<BasicBlock*> predecessors;
};

}  // namespace riscv64