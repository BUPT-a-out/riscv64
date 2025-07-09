#pragma once

#include <list>  // 使用链表便于指令的插入和删除
#include <string>

#include "Instruction.h"

namespace riscv64 {

class Function;  // 前向声明

class BasicBlock {
   public:
    BasicBlock(Function* parentFunc, const std::string& label)
        : label(label), parent(parentFunc) {}

    void addInstruction(Instruction* inst) { instructions.push_back(inst); }

    // 迭代器，便于遍历指令
    using iterator = std::list<Instruction*>::iterator;
    iterator begin() { return instructions.begin(); }
    iterator end() { return instructions.end(); }

    const std::string& getLabel() const { return label; }

    // CFG (控制流图) 信息
    void addSuccessor(BasicBlock* succ) { successors.push_back(succ); }
    void addPredecessor(BasicBlock* pred) { predecessors.push_back(pred); }

   private:
    std::string label;                     // 例如 ".LBB0_1"
    std::list<Instruction*> instructions;  // 使用 list 效率更高

    // 指向其所在的函数
    Function* parent;

    // 控制流图 (CFG) 的边
    std::vector<BasicBlock*> successors;
    std::vector<BasicBlock*> predecessors;
};

}  // namespace riscv64