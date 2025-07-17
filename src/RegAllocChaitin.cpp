#include "RegAllocChaitin.h"

#include <iostream>
namespace riscv64 {

void RegAllocChaitin::allocateRegisters() {
    // 1. 活跃性分析
    computeLiveness();

    // 2. 构建冲突图
    buildInterferenceGraph();

    // 3. 图着色
    bool success = colorGraph();

    // 4. 如果着色失败，处理溢出
    if (!success) {
        handleSpills();
        // 重新尝试分配
        allocateRegisters();
        return;
    }

    // 5. 重写指令
    rewriteInstructions();

    // 6. 输出结果（调试用）
    printAllocationResult();
}

void RegAllocChaitin::computeLiveness() {
    // 为每个基本块初始化活跃性信息
    for (auto& bb : *function) {
        livenessInfo[bb.get()] = LivenessInfo{};
        computeDefUse(bb.get(), livenessInfo[bb.get()]);
    }

    // 迭代计算活跃性直到收敛
    bool changed = true;
    while (changed) {
        changed = false;

        // 逆序遍历基本块（后向数据流分析）
        for (auto it = function->begin(); it != function->end(); ++it) {
            BasicBlock* bb = it->get();
            LivenessInfo& info = livenessInfo[bb];

            // 计算新的 liveOut
            std::unordered_set<unsigned> newLiveOut;
            for (BasicBlock* succ : bb->getSuccessors()) {
                const auto& succLiveIn = livenessInfo[succ].liveIn;
                newLiveOut.insert(succLiveIn.begin(), succLiveIn.end());
            }

            // 计算新的 liveIn
            std::unordered_set<unsigned> newLiveIn = newLiveOut;
            for (unsigned reg : info.def) {
                newLiveIn.erase(reg);
            }
            for (unsigned reg : info.use) {
                newLiveIn.insert(reg);
            }

            // 检查是否有变化
            if (newLiveIn != info.liveIn || newLiveOut != info.liveOut) {
                changed = true;
                info.liveIn = std::move(newLiveIn);
                info.liveOut = std::move(newLiveOut);
            }
        }
    }
}

void RegAllocChaitin::computeDefUse(BasicBlock* bb, LivenessInfo& info) {
    // 遍历基本块中的每条指令
    for (const auto& inst : *bb) {
        // 获取指令使用的寄存器
        auto usedRegs = getUsedRegs(inst.get());
        for (unsigned reg : usedRegs) {
            // 如果不在def集合中，则添加到use集合
            if (info.def.find(reg) == info.def.end()) {
                info.use.insert(reg);
            }
        }

        // 获取指令定义的寄存器
        auto definedRegs = getDefinedRegs(inst.get());
        for (unsigned reg : definedRegs) {
            info.def.insert(reg);
        }
    }
}

void RegAllocChaitin::buildInterferenceGraph() {
    // 为每个虚拟寄存器创建节点
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            auto usedRegs = getUsedRegs(inst.get());
            auto definedRegs = getDefinedRegs(inst.get());

            for (unsigned reg : usedRegs) {
                if (interferenceGraph.find(reg) == interferenceGraph.end()) {
                    interferenceGraph[reg] =
                        std::make_unique<InterferenceNode>(reg);
                    interferenceGraph[reg]->isPrecolored = isPhysicalReg(reg);
                }
            }

            for (unsigned reg : definedRegs) {
                if (interferenceGraph.find(reg) == interferenceGraph.end()) {
                    interferenceGraph[reg] =
                        std::make_unique<InterferenceNode>(reg);
                    interferenceGraph[reg]->isPrecolored = isPhysicalReg(reg);
                }
            }
        }
    }

    // 构建冲突边
    for (auto& bb : *function) {
        const LivenessInfo& info = livenessInfo[bb.get()];

        // 在每个程序点，活跃的变量之间都有冲突
        std::unordered_set<unsigned> live = info.liveOut;

        // 逆序遍历指令
        for (auto it = bb->begin(); it != bb->end(); ++it) {
            Instruction* inst = it->get();

            auto definedRegs = getDefinedRegs(inst);
            for (unsigned defReg : definedRegs) {
                // 定义的寄存器与当前活跃的所有寄存器冲突
                for (unsigned liveReg : live) {
                    if (defReg != liveReg) {
                        addInterference(defReg, liveReg);
                    }
                }

                // 移除定义的寄存器
                live.erase(defReg);
            }

            // 添加使用的寄存器
            auto usedRegs = getUsedRegs(inst);
            for (unsigned useReg : usedRegs) {
                live.insert(useReg);
            }
        }
    }
}

void RegAllocChaitin::addInterference(unsigned reg1, unsigned reg2) {
    if (interferenceGraph.find(reg1) != interferenceGraph.end() &&
        interferenceGraph.find(reg2) != interferenceGraph.end()) {
        interferenceGraph[reg1]->neighbors.insert(reg2);
        interferenceGraph[reg2]->neighbors.insert(reg1);
    }
}

bool RegAllocChaitin::colorGraph() {
    auto order = getSimplificationOrder();
    return attemptColoring(order);
}

std::vector<unsigned> RegAllocChaitin::getSimplificationOrder() {
    std::vector<unsigned> order;
    std::unordered_set<unsigned> removed;
    std::stack<unsigned> stack;

    // 简化阶段：移除度数小于K的节点
    bool changed = true;
    while (changed) {
        changed = false;

        for (auto& [regNum, node] : interferenceGraph) {
            if (removed.find(regNum) != removed.end() || node->isPrecolored) {
                continue;
            }

            // 计算当前度数（排除已移除的节点）
            int currentDegree = 0;
            for (unsigned neighbor : node->neighbors) {
                if (removed.find(neighbor) == removed.end()) {
                    currentDegree++;
                }
            }

            // 如果度数小于可用颜色数，则移除
            if (currentDegree < static_cast<int>(availableRegs.size())) {
                stack.push(regNum);
                removed.insert(regNum);
                changed = true;
            }
        }
    }

    // 如果还有未移除的节点，选择溢出候选
    for (auto& [regNum, node] : interferenceGraph) {
        if (removed.find(regNum) == removed.end() && !node->isPrecolored) {
            spilledRegs.insert(regNum);
        }
    }

    // 按栈顺序返回
    while (!stack.empty()) {
        order.push_back(stack.top());
        stack.pop();
    }

    return order;
}

bool RegAllocChaitin::attemptColoring(const std::vector<unsigned>& order) {
    // 预着色物理寄存器
    for (auto& [regNum, node] : interferenceGraph) {
        if (node->isPrecolored) {
            node->color = regNum;
        }
    }

    // 按顺序为每个虚拟寄存器着色
    for (unsigned regNum : order) {
        if (spilledRegs.find(regNum) != spilledRegs.end()) {
            continue;  // 跳过溢出的寄存器
        }

        auto& node = interferenceGraph[regNum];
        std::unordered_set<int> usedColors;

        // 收集邻居使用的颜色
        for (unsigned neighbor : node->neighbors) {
            if (interferenceGraph[neighbor]->color != -1) {
                usedColors.insert(interferenceGraph[neighbor]->color);
            }
        }

        // 选择第一个可用的颜色
        int selectedColor = -1;
        for (unsigned color : availableRegs) {
            if (usedColors.find(color) == usedColors.end()) {
                selectedColor = color;
                break;
            }
        }

        if (selectedColor == -1) {
            // 着色失败，标记为溢出
            spilledRegs.insert(regNum);
            return false;
        }

        node->color = selectedColor;
        virtualToPhysical[regNum] = selectedColor;
    }

    return spilledRegs.empty();
}

void RegAllocChaitin::handleSpills() {
    auto spillCandidates = selectSpillCandidates();

    for (unsigned reg : spillCandidates) {
        insertSpillCode(reg);
    }

    // 清空状态重新开始
    interferenceGraph.clear();
    virtualToPhysical.clear();
    spilledRegs.clear();
}

std::vector<unsigned> RegAllocChaitin::selectSpillCandidates() {
    std::vector<unsigned> candidates(spilledRegs.begin(), spilledRegs.end());

    // 简单的启发式：选择度数最高的寄存器
    std::sort(candidates.begin(), candidates.end(),
              [this](unsigned a, unsigned b) {
                  return interferenceGraph[a]->neighbors.size() >
                         interferenceGraph[b]->neighbors.size();
              });

    return candidates;
}

void RegAllocChaitin::insertSpillCode(unsigned reg) {
    // 为溢出的寄存器在栈上分配空间
    // 这里简化处理，实际需要与栈帧管理器配合

    for (auto& bb : *function) {
        for (auto it = bb->begin(); it != bb->end(); ++it) {
            Instruction* inst = it->get();

            auto usedRegs = getUsedRegs(inst);
            auto definedRegs = getDefinedRegs(inst);

            // 如果指令使用了溢出的寄存器，在指令前插入加载
            if (std::find(usedRegs.begin(), usedRegs.end(), reg) !=
                usedRegs.end()) {
                // 创建临时寄存器
                unsigned tempReg = 1000 + reg;  // 简单的临时寄存器分配

                // 插入加载指令: ld tempReg, offset(sp)
                auto loadInst = std::make_unique<Instruction>(LD);
                loadInst->addOperand(
                    std::make_unique<RegisterOperand>(tempReg));
                loadInst->addOperand(std::make_unique<MemoryOperand>(
                    std::make_unique<RegisterOperand>(2),  // sp
                    std::make_unique<ImmediateOperand>(reg *
                                                       8)  // 简单的偏移计算
                    ));

                it = bb->insert(it, std::move(loadInst));
                ++it;

                // 替换指令中的寄存器使用
                // 这里需要修改指令的操作数，将reg替换为tempReg
            }

            // 如果指令定义了溢出的寄存器，在指令后插入存储
            if (std::find(definedRegs.begin(), definedRegs.end(), reg) !=
                definedRegs.end()) {
                unsigned tempReg = 1000 + reg;

                // 插入存储指令: sd tempReg, offset(sp)
                auto storeInst = std::make_unique<Instruction>(SD);
                storeInst->addOperand(
                    std::make_unique<RegisterOperand>(tempReg));
                storeInst->addOperand(std::make_unique<MemoryOperand>(
                    std::make_unique<RegisterOperand>(2),  // sp
                    std::make_unique<ImmediateOperand>(reg * 8)));

                ++it;
                it = bb->insert(it, std::move(storeInst));
            }
        }
    }
}

void RegAllocChaitin::rewriteInstructions() {
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            rewriteInstruction(inst.get());
        }
    }
}

void RegAllocChaitin::rewriteInstruction(Instruction* inst) {
    // 遍历指令的所有操作数
    const auto& operands = inst->getOperands();
    for (auto& operand : operands) {
        if (operand->isReg()) {
            RegisterOperand* regOp =
                static_cast<RegisterOperand*>(operand.get());
            if (regOp->isVirtual()) {
                unsigned virtualReg = regOp->getRegNum();
                if (virtualToPhysical.find(virtualReg) !=
                    virtualToPhysical.end()) {
                    // 将虚拟寄存器替换为物理寄存器
                    regOp->setPhysicalReg(virtualToPhysical[virtualReg]);
                }
            }
        }
    }
}

// 辅助函数实现
bool RegAllocChaitin::isPhysicalReg(unsigned reg) const {
    return reg < 32;  // RISC-V有32个物理寄存器
}

unsigned RegAllocChaitin::getPhysicalReg(unsigned virtualReg) const {
    auto it = virtualToPhysical.find(virtualReg);
    if (it != virtualToPhysical.end()) {
        return it->second;
    }
    return virtualReg;  // 如果没有映射，返回原寄存器
}

std::vector<unsigned> RegAllocChaitin::getUsedRegs(
    const Instruction* inst) const {
    std::vector<unsigned> usedRegs;

    // 根据指令类型分析使用的寄存器
    // 这里简化处理，实际需要根据具体的指令格式来分析
    const auto& operands = inst->getOperands();

    // 通常第一个操作数是目标寄存器（定义），其余是源寄存器（使用）
    for (size_t i = 1; i < operands.size(); ++i) {
        if (operands[i]->isReg()) {
            usedRegs.push_back(operands[i]->getRegNum());
        }
    }

    return usedRegs;
}

std::vector<unsigned> RegAllocChaitin::getDefinedRegs(
    const Instruction* inst) const {
    std::vector<unsigned> definedRegs;

    // 通常第一个操作数是目标寄存器
    if (!inst->getOperands().empty() && inst->getOperands()[0]->isReg()) {
        definedRegs.push_back(inst->getOperands()[0]->getRegNum());
    }

    return definedRegs;
}

void RegAllocChaitin::printInterferenceGraph() const {
    std::cout << "Interference Graph:\n";
    for (const auto& [regNum, node] : interferenceGraph) {
        std::cout << "Register " << regNum << " conflicts with: ";
        for (unsigned neighbor : node->neighbors) {
            std::cout << neighbor << " ";
        }
        std::cout << "\n";
    }
}

void RegAllocChaitin::printAllocationResult() const {
    std::cout << "Register Allocation Result:\n";
    for (const auto& [virtualReg, physicalReg] : virtualToPhysical) {
        std::cout << "Virtual register " << virtualReg
                  << " -> Physical register " << physicalReg << " ("
                  << ABI::getABINameFromRegNum(physicalReg) << ")\n";
    }

    if (!spilledRegs.empty()) {
        std::cout << "Spilled registers: ";
        for (unsigned reg : spilledRegs) {
            std::cout << reg << " ";
        }
        std::cout << "\n";
    }
}

}  // namespace riscv64
