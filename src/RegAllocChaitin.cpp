#include "RegAllocChaitin.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <stack>

#include "StackFrameManager.h"
namespace riscv64 {

/// Entry
void RegAllocChaitin::allocateRegisters() {
    initializeABIConstraints();

    computeLiveness();

    buildInterferenceGraph();

    // performCoalescing();

    bool success = colorGraph();

    // 如果着色失败，处理溢出
    if (!success) {
        handleSpills();
        // 重新尝试分配
        allocateRegisters();
        return;
    }

    removeCoalescedCopies();

    rewriteInstructions();

    printAllocationResult();
    printCoalesceResult();
    stackManager.printStackLayout();
    // TODO: post opt
}

/// degree cache
void RegAllocChaitin::initializeDegreeCache() {
    degreeCache.clear();
    for (const auto& [regNum, node] : interferenceGraph) {
        if (!node->isPrecolored) {
            degreeCache[regNum] = node->neighbors.size();
        }
    }
}

int RegAllocChaitin::getCachedDegree(unsigned reg) {
    auto it = degreeCache.find(reg);
    if (it != degreeCache.end()) {
        return it->second;
    }

    // 如果缓存中没有，重新计算并缓存
    if (interferenceGraph.find(reg) != interferenceGraph.end()) {
        int degree = interferenceGraph[reg]->neighbors.size();
        degreeCache[reg] = degree;
        return degree;
    }

    return 0;
}

void RegAllocChaitin::updateDegreeAfterRemoval(unsigned removedReg) {
    if (interferenceGraph.find(removedReg) == interferenceGraph.end()) {
        return;
    }

    // 更新所有邻居的度数
    for (unsigned neighbor : interferenceGraph[removedReg]->neighbors) {
        if (degreeCache.find(neighbor) != degreeCache.end()) {
            degreeCache[neighbor]--;
        }
    }

    // 移除被删除节点的度数缓存
    degreeCache.erase(removedReg);
}

void RegAllocChaitin::updateDegreeAfterCoalesce(unsigned merged,
                                                unsigned eliminated) {
    if (interferenceGraph.find(eliminated) == interferenceGraph.end() ||
        interferenceGraph.find(merged) == interferenceGraph.end()) {
        return;
    }

    auto& eliminatedNode = interferenceGraph[eliminated];
    auto& mergedNode = interferenceGraph[merged];

    // 计算合并后的度数变化
    std::unordered_set<unsigned> newNeighbors;
    std::unordered_set<unsigned> commonNeighbors;

    // 找出eliminated的邻居中，merged原本没有的
    for (unsigned neighbor : eliminatedNode->neighbors) {
        if (neighbor != merged) {
            if (mergedNode->neighbors.find(neighbor) ==
                mergedNode->neighbors.end()) {
                newNeighbors.insert(neighbor);
            } else {
                commonNeighbors.insert(neighbor);
            }
        }
    }

    // 更新merged节点的度数
    if (degreeCache.find(merged) != degreeCache.end()) {
        degreeCache[merged] += newNeighbors.size();
    }

    // 更新所有相关邻居的度数
    // 1. eliminated的邻居需要减1（因为eliminated被移除）
    // 2. 如果是新邻居，还需要加1（因为现在与merged连接）
    for (unsigned neighbor : eliminatedNode->neighbors) {
        if (neighbor != merged &&
            degreeCache.find(neighbor) != degreeCache.end()) {
            degreeCache[neighbor]--;  // 失去与eliminated的连接

            // 如果是新连接的邻居，需要加1
            if (newNeighbors.find(neighbor) != newNeighbors.end()) {
                degreeCache[neighbor]++;
            }
        }
    }

    // 移除eliminated节点的度数缓存
    degreeCache.erase(eliminated);
}

void RegAllocChaitin::invalidateDegreeCache(unsigned reg) {
    degreeCache.erase(reg);
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

        // 特殊处理函数调用指令
        if (inst->isCallInstr()) {
            // 函数调用会隐式修改所有调用者保存寄存器
            std::vector<unsigned> callerSavedRegs = {
                5,  6,  7,                       // t0-t2
                10, 11, 12, 13, 14, 15, 16, 17,  // a0-a7
                28, 29, 30, 31                   // t3-t6
            };

            for (unsigned reg : callerSavedRegs) {
                info.def.insert(reg);  // 调用者保存寄存器被隐式定义
            }
        }
    }
}

void RegAllocChaitin::buildInterferenceGraph() {
    // Precolor
    for (unsigned physReg : availableRegs) {
        interferenceGraph[physReg] =
            std::make_unique<InterferenceNode>(physReg);
        interferenceGraph[physReg]->isPrecolored = true;
        interferenceGraph[physReg]->color = physReg;
        interferenceGraph[physReg]->coalesceParent = physReg;
    }

    // 为每个虚拟寄存器创建节点
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            auto usedRegs = getUsedRegs(inst.get());
            auto definedRegs = getDefinedRegs(inst.get());

            for (unsigned reg : usedRegs) {
                if (interferenceGraph.find(reg) == interferenceGraph.end()) {
                    interferenceGraph[reg] =
                        std::make_unique<InterferenceNode>(reg);
                    interferenceGraph[reg]->isPrecolored = false;
                    interferenceGraph[reg]->coalesceParent = reg;
                }
            }

            for (unsigned reg : definedRegs) {
                if (interferenceGraph.find(reg) == interferenceGraph.end()) {
                    interferenceGraph[reg] =
                        std::make_unique<InterferenceNode>(reg);
                    interferenceGraph[reg]->isPrecolored = false;
                    interferenceGraph[reg]->coalesceParent = reg;
                }
            }
        }
    }

    // 构建冲突边：逆序遍历
    for (auto& bb : *function) {
        const LivenessInfo& info = livenessInfo[bb.get()];
        std::unordered_set<unsigned> live = info.liveOut;

        // 逆序遍历指令
        for (auto it = bb->rbegin(); it != bb->rend(); ++it) {
            Instruction* inst = it->get();

            // 特殊处理函数调用指令
            if (inst->isCallInstr()) {
                std::vector<unsigned> callerSavedRegs = {
                    5,  6,  7,                       // t0-t2
                    10, 11, 12, 13, 14, 15, 16, 17,  // a0-a7
                    28, 29, 30, 31                   // t3-t6
                };

                // 调用者保存寄存器与所有在调用后仍活跃的虚拟寄存器冲突
                for (unsigned callerReg : callerSavedRegs) {
                    for (unsigned liveReg : live) {
                        addInterference(liveReg, callerReg);
                    }
                }

                // 移除被调用指令重新定义的寄存器
                for (unsigned callerReg : callerSavedRegs) {
                    live.erase(callerReg);
                }
            }

            // 处理普通指令的定义和使用
            auto definedRegs = getDefinedRegs(inst);
            for (unsigned defReg : definedRegs) {
                // 与当前活跃的虚拟寄存器建立冲突
                for (unsigned liveReg : live) {
                    if (defReg != liveReg && !isPhysicalReg(liveReg)) {
                        addInterference(defReg, liveReg);
                    }
                }
                live.erase(defReg);
            }

            auto usedRegs = getUsedRegs(inst);
            for (unsigned useReg : usedRegs) {
                live.insert(useReg);
            }
        }
    }
}

// 添加物理寄存器约束的方法
void RegAllocChaitin::addPhysicalConstraint(unsigned virtualReg,
                                            unsigned physicalReg) {
    if (physicalConstraints.find(virtualReg) == physicalConstraints.end()) {
        physicalConstraints[virtualReg] = std::unordered_set<unsigned>();
    }
    physicalConstraints[virtualReg].insert(physicalReg);
}

void RegAllocChaitin::addInterference(unsigned reg1, unsigned reg2) {
    // 至少有一个是虚拟寄存器，且都在冲突图中
    if (isPhysicalReg(reg1) && isPhysicalReg(reg2)) return;

    if (interferenceGraph.find(reg1) != interferenceGraph.end() &&
        interferenceGraph.find(reg2) != interferenceGraph.end()) {
        interferenceGraph[reg1]->neighbors.insert(reg2);
        interferenceGraph[reg2]->neighbors.insert(reg1);
    }
}

/// Coloring
bool RegAllocChaitin::colorGraph() {
    auto order = getSimplificationOrder();
    return attemptColoring(order);
}

std::vector<unsigned> RegAllocChaitin::getSimplificationOrder() {
    std::vector<unsigned> order;
    std::unordered_set<unsigned> removed;
    std::stack<unsigned> stack;

    // 初始化度数缓存
    initializeDegreeCache();

    // 工作列表：维护当前度数小于K的节点
    std::queue<unsigned> workList;

    // 初始化工作列表
    for (auto& [regNum, node] : interferenceGraph) {
        if (!node->isPrecolored) {
            int currentDegree = getCachedDegree(regNum);
            if (currentDegree < static_cast<int>(availableRegs.size())) {
                workList.push(regNum);
            }
        }
    }

    // 简化阶段：处理工作列表中的节点
    while (!workList.empty()) {
        unsigned regNum = workList.front();
        workList.pop();

        // 检查节点是否已经被处理
        if (removed.find(regNum) != removed.end()) {
            continue;
        }

        // 再次检查度数（可能在之前的处理中已经改变）
        int currentDegree = getCachedDegree(regNum);
        if (currentDegree >= static_cast<int>(availableRegs.size())) {
            continue;
        }

        // 移除节点
        stack.push(regNum);
        removed.insert(regNum);

        // 更新邻居的度数，并检查是否有新的节点可以加入工作列表
        if (interferenceGraph.find(regNum) != interferenceGraph.end()) {
            for (unsigned neighbor : interferenceGraph[regNum]->neighbors) {
                if (removed.find(neighbor) == removed.end() &&
                    !interferenceGraph[neighbor]->isPrecolored) {
                    // 更新邻居的度数
                    if (degreeCache.find(neighbor) != degreeCache.end()) {
                        degreeCache[neighbor]--;

                        // 如果邻居的度数现在小于K，加入工作列表
                        if (degreeCache[neighbor] <
                            static_cast<int>(availableRegs.size())) {
                            workList.push(neighbor);
                        }
                    }
                }
            }
        }

        // 从度数缓存中移除当前节点
        degreeCache.erase(regNum);
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

// TODO: 没有保护callee saved寄存器s0-s11
bool RegAllocChaitin::attemptColoring(const std::vector<unsigned>& order) {
    for (unsigned regNum : order) {
        if (spilledRegs.find(regNum) != spilledRegs.end()) {
            continue;
        }

        auto& node = interferenceGraph[regNum];
        std::unordered_set<int> usedColors;

        // 收集邻居使用的颜色
        for (unsigned neighbor : node->neighbors) {
            if (interferenceGraph[neighbor]->color != -1) {
                usedColors.insert(interferenceGraph[neighbor]->color);
            }
        }

        int selectedColor = -1;

        // 首先检查强约束
        if (strongConstraints.find(regNum) != strongConstraints.end()) {
            unsigned requiredColor = strongConstraints[regNum];
            if (usedColors.find(requiredColor) == usedColors.end()) {
                selectedColor = requiredColor;
            } else {
                // 强约束冲突，必须溢出
                spilledRegs.insert(regNum);
                return false;
            }
        } else {
            // 正常着色流程，但要避开保留寄存器
            auto preferredRegs = getABIPreferredRegs(regNum);
            for (unsigned color : preferredRegs) {
                if (usedColors.find(color) == usedColors.end() &&
                    reservedPhysicalRegs.find(color) ==
                        reservedPhysicalRegs.end() &&
                    std::find(availableRegs.begin(), availableRegs.end(),
                              color) != availableRegs.end()) {
                    selectedColor = color;
                    break;
                }
            }

            if (selectedColor == -1) {
                for (unsigned color : availableRegs) {
                    if (usedColors.find(color) == usedColors.end() &&
                        reservedPhysicalRegs.find(color) ==
                            reservedPhysicalRegs.end() &&
                        !isReservedReg(color)) {
                        selectedColor = color;
                        break;
                    }
                }
            }
        }

        if (selectedColor == -1) {
            spilledRegs.insert(regNum);
            return false;
        }

        node->color = selectedColor;
        virtualToPhysical[regNum] = selectedColor;
    }

    return spilledRegs.empty();
}

/// Spill
void RegAllocChaitin::handleSpills() {
    auto spillCandidates = selectSpillCandidates();

    for (unsigned reg : spillCandidates) {
        insertSpillCode(reg);
    }

    // 清空状态重新开始
    interferenceGraph.clear();
    virtualToPhysical.clear();
    spilledRegs.clear();

    coalesceCandidates.clear();
    coalesceMap.clear();
    coalescedRegs.clear();
    livenessInfo.clear();

    physicalConstraints.clear();

    clearDegreeCache();  // 清理度数缓存
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
    stackManager.allocateSpillSlot(reg);
    stackManager.computeStackFrame();
    int spillOffset = stackManager.getSpillSlotOffset(reg);

    for (auto& bb : *function) {
        for (auto it = bb->begin(); it != bb->end(); ++it) {
            Instruction* inst = it->get();

            // 检查指令是否使用了溢出的寄存器
            auto usedRegs = getUsedRegs(inst);
            if (std::find(usedRegs.begin(), usedRegs.end(), reg) !=
                usedRegs.end()) {
                // 创建独立的临时寄存器用于load
                unsigned tempReg = createTempReg();

                auto loadInst = std::make_unique<Instruction>(LD);
                loadInst->addOperand(
                    std::make_unique<RegisterOperand>(tempReg, true));
                loadInst->addOperand(std::make_unique<MemoryOperand>(
                    std::make_unique<RegisterOperand>(2, false),
                    std::make_unique<ImmediateOperand>(spillOffset)));

                it = bb->insert(it, std::move(loadInst));
                ++it;

                // 更新原指令中的寄存器引用
                updateRegisterInInstruction(inst, reg, tempReg);
            }

            // 检查指令是否定义了溢出的寄存器
            auto definedRegs = getDefinedRegs(inst);
            if (std::find(definedRegs.begin(), definedRegs.end(), reg) !=
                definedRegs.end()) {
                // 创建独立的临时寄存器用于store
                unsigned tempReg = createTempReg();

                // 更新原指令中的寄存器引用
                updateRegisterInInstruction(inst, reg, tempReg);

                auto storeInst = std::make_unique<Instruction>(SD);
                storeInst->addOperand(
                    std::make_unique<RegisterOperand>(tempReg, true));
                storeInst->addOperand(std::make_unique<MemoryOperand>(
                    std::make_unique<RegisterOperand>(2, false),
                    std::make_unique<ImmediateOperand>(spillOffset)));

                ++it;
                it = bb->insert(it, std::move(storeInst));
            }
        }
    }
}

// 更新指令中的寄存器引用
void RegAllocChaitin::updateRegisterInInstruction(Instruction* inst,
                                                  unsigned oldReg,
                                                  unsigned newReg) {
    const auto& operands = inst->getOperands();
    for (const auto& operand : operands) {
        if (operand->isReg()) {
            RegisterOperand* regOp =
                static_cast<RegisterOperand*>(operand.get());
            if (regOp->getRegNum() == oldReg) {
                regOp->setRegNum(newReg);
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
    const auto& operands = inst->getOperands();
    for (const auto& operand : operands) {
        rewriteOperand(operand.get());
    }
}

void RegAllocChaitin::rewriteOperand(MachineOperand* operand) {
    if (operand->isReg()) {
        RegisterOperand* regOp = static_cast<RegisterOperand*>(operand);
        if (regOp->isVirtual()) {
            unsigned virtualReg = regOp->getRegNum();
            unsigned finalReg = getFinalCoalescedReg(virtualReg);

            if (virtualToPhysical.find(finalReg) != virtualToPhysical.end()) {
                regOp->setPhysicalReg(virtualToPhysical[finalReg]);
            } else if (isPhysicalReg(finalReg)) {
                regOp->setPhysicalReg(finalReg);
            }
        }
    } else if (operand->isMem()) {
        MemoryOperand* memOp = static_cast<MemoryOperand*>(operand);
        // 递归处理内存操作数中的基址寄存器和偏移量
        if (memOp->getBaseReg()) {
            rewriteOperand(memOp->getBaseReg());
        }
        if (memOp->getOffset()) {
            rewriteOperand(memOp->getOffset());
        }
    }
    // 可以继续添加其他类型操作数的处理
}

/// Coalesce
// 获取最终合并后的寄存器
unsigned RegAllocChaitin::getFinalCoalescedReg(unsigned reg) {
    unsigned current = reg;
    std::unordered_set<unsigned> visited;  // 防止循环

    while (coalesceMap.find(current) != coalesceMap.end() &&
           visited.find(current) == visited.end()) {
        visited.insert(current);
        current = coalesceMap[current];
    }

    return current;
}

// 执行寄存器合并的主要方法
void RegAllocChaitin::performCoalescing() {
    // 识别合并候选
    identifyCoalesceCandidates();

    // 按优先级排序
    std::sort(coalesceCandidates.begin(), coalesceCandidates.end(),
              [](const CoalesceInfo& a, const CoalesceInfo& b) {
                  return a.priority > b.priority;
              });

    // 尝试合并
    for (const auto& candidate : coalesceCandidates) {
        if (candidate.canCoalesce &&
            coalescedRegs.find(candidate.src) == coalescedRegs.end() &&
            coalescedRegs.find(candidate.dst) == coalescedRegs.end()) {
            if (canCoalesce(candidate.src, candidate.dst)) {
                coalesceRegisters(candidate.src, candidate.dst);
            }
        }
    }
}

// 识别复制指令中的合并候选
void RegAllocChaitin::identifyCoalesceCandidates() {
    coalesceCandidates.clear();

    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            if (inst->isCopyInstr()) {
                const auto& operands = inst->getOperands();
                if (operands.size() >= 2 && operands[0]->isReg() &&
                    operands[1]->isReg()) {
                    unsigned dst = operands[0]->getRegNum();
                    unsigned src = operands[1]->getRegNum();

                    // 计算合并优先级（基于指令频次、循环嵌套等）
                    int priority = calculateCoalescePriority(src, dst, bb.get(),
                                                             inst.get());

                    CoalesceInfo info;
                    info.src = src;
                    info.dst = dst;
                    info.canCoalesce = true;
                    info.priority = priority;

                    coalesceCandidates.push_back(info);
                }
            }
        }
    }
}

int RegAllocChaitin::calculateCoalescePriority(unsigned src, unsigned dst,
                                               BasicBlock* bb,
                                               Instruction* inst) {
    (void)inst;  // Suppress unused parameter warning
    int priority = 0;

    // 1. 基本块执行频率权重 (0-100)
    int bbFrequency = getBasicBlockFrequency(bb);
    priority += bbFrequency * 10;

    // 2. 寄存器使用频率权重 (0-30)
    int srcUsageCount = getRegisterUsageCount(src);
    int dstUsageCount = getRegisterUsageCount(dst);
    priority += (srcUsageCount + dstUsageCount) * 2;

    // 3. 冲突图度数权重 (度数越小优先级越高) (0-20)
    int srcDegree = getRegisterDegree(src);
    int dstDegree = getRegisterDegree(dst);
    int avgDegree = (srcDegree + dstDegree) / 2;
    priority += std::max(0, 20 - avgDegree);

    // 4. 生命周期重叠度权重 (重叠越少优先级越高) (0-15)
    int lifeOverlap = calculateLifetimeOverlap(src, dst);
    priority += std::max(0, 15 - lifeOverlap);

    // 5. 寄存器压力权重 (0-10)
    int regPressure = getRegisterPressure(bb);
    if (regPressure > static_cast<int>(availableRegs.size() * 0.8)) {
        priority += 10;  // 高寄存器压力时更倾向于合并
    }

    // 6. 物理寄存器偏好权重 (0-25)
    priority += calculatePhysicalRegPreference(src, dst);

    // 7. 复制指令消除收益权重 (0-5)
    priority += 5;  // 消除一条复制指令的基本收益

    // 8. ABI约束权重 (0-40)
    priority += calculateABIPriority(src, dst);

    return priority;
}

int RegAllocChaitin::calculateABIPriority(unsigned src, unsigned dst) const {
    int priority = 0;

    // 如果目标是物理寄存器，根据ABI类型给予不同权重
    if (isPhysicalReg(dst)) {
        if (isArgumentReg(dst)) {
            priority += 20;  // 参数寄存器优先级高
        } else if (isCalleeSaved(dst)) {
            priority += 15;  // 被调用者保存寄存器次之
        } else if (isCallerSaved(dst)) {
            priority += 10;  // 调用者保存寄存器再次之
        }
    } else if (isPhysicalReg(src)) {
        if (isArgumentReg(src)) {
            priority += 18;
        } else if (isCalleeSaved(src)) {
            priority += 13;
        } else if (isCallerSaved(src)) {
            priority += 8;
        }
    }

    // 如果两个寄存器都跨越函数调用，且目标是被调用者保存寄存器，增加权重
    if (crossesFunctionCall(src, dst)) {
        if (isPhysicalReg(dst) && isCalleeSaved(dst)) {
            priority += 25;
        } else if (isPhysicalReg(src) && isCalleeSaved(src)) {
            priority += 20;
        }
    }

    // 如果不跨越函数调用，且目标是调用者保存寄存器，增加权重
    if (!crossesFunctionCall(src, dst)) {
        if (isPhysicalReg(dst) && isCallerSaved(dst)) {
            priority += 15;
        } else if (isPhysicalReg(src) && isCallerSaved(src)) {
            priority += 12;
        }
    }

    return priority;
}

/// 辅助函数

int RegAllocChaitin::getBasicBlockFrequency(BasicBlock* bb) {
    // 静态估算基本块频率
    int frequency = 10;  // 基础频率

    // 1. 如果是函数入口块，频率较高
    if (bb == function->begin()->get()) {
        frequency += 50;
    }

    // 2. 根据前驱块数量调整（更多前驱意味着可能被更频繁执行）
    int predecessorCount = bb->getPredecessors().size();
    if (predecessorCount > 1) {
        frequency += predecessorCount * 10;  // 汇聚点通常执行频率更高
    }

    // 3. 根据后继块数量调整
    int successorCount = bb->getSuccessors().size();
    if (successorCount == 1) {
        frequency += 20;  // 直线代码块
    } else if (successorCount > 1) {
        frequency += 15;  // 分支块
    }

    // 4. 根据基本块大小调整（更大的块可能在热路径上）
    int blockSize = bb->size();
    if (blockSize > 10) {
        frequency += 10;
    }

    // 5. 检查是否包含函数调用（调用通常在较冷的路径上）
    for (const auto& inst : *bb) {
        if (inst->isCallInstr()) {
            frequency -= 15;
            break;
        }
    }

    // 6. 检查是否包含循环相关指令
    for (const auto& inst : *bb) {
        if (inst->isBranch() && inst->isBackEdge()) {
            frequency += 30;  // 循环回边
            break;
        }
    }

    return std::max(frequency, 1);  // 确保至少为1
}

int RegAllocChaitin::getRegisterUsageCount(unsigned reg) {
    int count = 0;
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            auto usedRegs = getUsedRegs(inst.get());
            auto definedRegs = getDefinedRegs(inst.get());
            if (std::find(usedRegs.begin(), usedRegs.end(), reg) !=
                    usedRegs.end() ||
                std::find(definedRegs.begin(), definedRegs.end(), reg) !=
                    definedRegs.end()) {
                count++;
            }
        }
    }
    return count;
}

int RegAllocChaitin::getRegisterDegree(unsigned reg) {
    return getCachedDegree(reg);
}

int RegAllocChaitin::calculateLifetimeOverlap(unsigned src, unsigned dst) {
    // 统计两个寄存器共同活跃的程序点数量
    int overlap = 0;
    for (auto& bb : *function) {
        const auto& liveIn = livenessInfo[bb.get()].liveIn;
        const auto& liveOut = livenessInfo[bb.get()].liveOut;
        if ((liveIn.find(src) != liveIn.end() ||
             liveOut.find(src) != liveOut.end()) &&
            (liveIn.find(dst) != liveIn.end() ||
             liveOut.find(dst) != liveOut.end())) {
            overlap++;
        }
    }
    return overlap;
}

int RegAllocChaitin::getRegisterPressure(BasicBlock* bb) {
    // 计算基本块内活跃变量的峰值数量
    int maxLive = 0;
    std::unordered_set<unsigned> live = livenessInfo[bb].liveOut;

    for (auto it = bb->begin(); it != bb->end(); ++it) {
        auto definedRegs = getDefinedRegs(it->get());
        for (unsigned reg : definedRegs) {
            live.erase(reg);
        }

        auto usedRegs = getUsedRegs(it->get());
        for (unsigned reg : usedRegs) {
            live.insert(reg);
        }

        maxLive = std::max(maxLive, static_cast<int>(live.size()));
    }
    return maxLive;
}

int RegAllocChaitin::calculatePhysicalRegPreference(unsigned src,
                                                    unsigned dst) {
    int score = 0;
    // 如果目标寄存器是物理寄存器，优先合并
    if (isPhysicalReg(dst)) {
        score += 15;
    }
    // 如果源寄存器是物理寄存器，次优先
    else if (isPhysicalReg(src)) {
        score += 10;
    }
    return score;
}

// 检查两个寄存器是否可以合并
bool RegAllocChaitin::canCoalesce(unsigned src, unsigned dst) {
    // 1. 检查是否已经被合并
    if (coalescedRegs.find(src) != coalescedRegs.end() ||
        coalescedRegs.find(dst) != coalescedRegs.end()) {
        return false;
    }

    // 获取最终的合并目标
    unsigned finalSrc = getFinalCoalescedReg(src);
    unsigned finalDst = getFinalCoalescedReg(dst);

    // 如果最终目标相同，则不需要合并
    if (finalSrc == finalDst) {
        return false;
    }

    // 2. 检查ABI约束
    if (!canCoalesceWithABI(src, dst)) {
        return false;
    }

    // 3. 检查是否存在冲突
    if (interferenceGraph.find(src) != interferenceGraph.end() &&
        interferenceGraph.find(dst) != interferenceGraph.end()) {
        auto& srcNode = interferenceGraph[src];
        auto& dstNode = interferenceGraph[dst];

        // 如果两个寄存器直接冲突，不能合并
        if (srcNode->neighbors.find(dst) != srcNode->neighbors.end()) {
            return false;
        }

        // Briggs准则：合并后的节点度数要小于K（可用寄存器数量）
        std::unordered_set<unsigned> combinedNeighbors = srcNode->neighbors;
        for (unsigned neighbor : dstNode->neighbors) {
            combinedNeighbors.insert(neighbor);
        }
        if (combinedNeighbors.size() >= availableRegs.size()) {
            return false;
        }

        // George准则：对于每个高度数的邻居，它要么已经与目标寄存器冲突，要么度数小于K
        for (unsigned neighbor : srcNode->neighbors) {
            if (interferenceGraph[neighbor]->neighbors.size() >=
                availableRegs.size()) {
                if (dstNode->neighbors.find(neighbor) ==
                    dstNode->neighbors.end()) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool RegAllocChaitin::canCoalesceWithABI(unsigned src, unsigned dst) const {
    // 不能合并保留寄存器
    if (isReservedReg(src) || isReservedReg(dst)) {
        return false;
    }

    // 如果其中一个是物理寄存器，检查ABI约束
    if (isPhysicalReg(src) || isPhysicalReg(dst)) {
        // 调用者保存和被调用者保存寄存器不能合并
        if (isPhysicalReg(src) && isPhysicalReg(dst)) {
            if (isCallerSaved(src) && isCalleeSaved(dst)) return false;
            if (isCalleeSaved(src) && isCallerSaved(dst)) return false;
        }

        // 检查函数调用边界
        if (crossesFunctionCall(src, dst)) {
            unsigned physReg = isPhysicalReg(src) ? src : dst;
            if (isCallerSaved(physReg)) {
                return false;  // 调用者保存寄存器不能跨函数调用合并
            }
        }
    }

    return true;
}

bool RegAllocChaitin::crossesFunctionCall(unsigned src, unsigned dst) const {
    // 检查两个寄存器的生命周期是否跨越函数调用
    for (auto& bb : *function) {
        const auto& liveIn = livenessInfo.at(bb.get()).liveIn;
        const auto& liveOut = livenessInfo.at(bb.get()).liveOut;

        bool srcLive = (liveIn.find(src) != liveIn.end()) ||
                       (liveOut.find(src) != liveOut.end());
        bool dstLive = (liveIn.find(dst) != liveIn.end()) ||
                       (liveOut.find(dst) != liveOut.end());

        if (srcLive && dstLive) {
            // 检查基本块是否包含函数调用
            for (const auto& inst : *bb) {
                if (inst->isCallInstr()) {
                    return true;
                }
            }
        }
    }
    return false;
}

// 执行寄存器合并
void RegAllocChaitin::coalesceRegisters(unsigned src, unsigned dst) {
    // 使用Union-Find结构管理合并
    unsigned srcRoot = findCoalesceRoot(src);
    unsigned dstRoot = findCoalesceRoot(dst);

    if (srcRoot != dstRoot) {
        // 合并到dst
        unionCoalesce(srcRoot, dstRoot);
        coalesceMap[src] = dst;
        coalescedRegs.insert(src);

        // 增量更新度数缓存
        updateDegreeAfterCoalesce(dst, src);

        // 更新冲突图
        updateInterferenceAfterCoalesce(dst, src);

        std::cout << "Coalesced register " << src << " into " << dst
                  << std::endl;
    }
}

// Union-Find 查找根节点
unsigned RegAllocChaitin::findCoalesceRoot(unsigned reg) {
    if (interferenceGraph.find(reg) == interferenceGraph.end()) {
        return reg;
    }

    auto& node = interferenceGraph[reg];
    if (node->coalesceParent != reg) {
        node->coalesceParent = findCoalesceRoot(node->coalesceParent);
    }
    return node->coalesceParent;
}

// Union-Find 合并操作
void RegAllocChaitin::unionCoalesce(unsigned reg1, unsigned reg2) {
    unsigned root1 = findCoalesceRoot(reg1);
    unsigned root2 = findCoalesceRoot(reg2);

    if (root1 != root2) {
        // 简单合并，可以根据rank优化
        if (interferenceGraph.find(root1) != interferenceGraph.end()) {
            interferenceGraph[root1]->coalesceParent = root2;
        }
    }
}

// 合并后更新冲突图
void RegAllocChaitin::updateInterferenceAfterCoalesce(unsigned merged,
                                                      unsigned eliminated) {
    if (interferenceGraph.find(eliminated) == interferenceGraph.end() ||
        interferenceGraph.find(merged) == interferenceGraph.end()) {
        return;
    }

    auto& eliminatedNode = interferenceGraph[eliminated];
    auto& mergedNode = interferenceGraph[merged];

    // 将eliminated的所有邻居转移到merged
    for (unsigned neighbor : eliminatedNode->neighbors) {
        if (neighbor != merged &&
            interferenceGraph.find(neighbor) != interferenceGraph.end()) {
            // 添加到merged的邻居
            mergedNode->neighbors.insert(neighbor);
            // 更新邻居的冲突信息
            interferenceGraph[neighbor]->neighbors.erase(eliminated);
            interferenceGraph[neighbor]->neighbors.insert(merged);
        }
    }

    // 移除eliminated节点
    interferenceGraph.erase(eliminated);
}

// 移除已合并的复制指令
void RegAllocChaitin::removeCoalescedCopies() {
    for (auto& bb : *function) {
        for (auto it = bb->begin(); it != bb->end();) {
            auto& inst = *it;
            if (inst->isCopyInstr()) {
                const auto& operands = inst->getOperands();
                if (operands.size() >= 2 && operands[0]->isReg() &&
                    operands[1]->isReg()) {
                    unsigned dst = operands[0]->getRegNum();
                    unsigned src = operands[1]->getRegNum();

                    // 检查是否已经合并
                    if (coalesceMap.find(src) != coalesceMap.end() &&
                        coalesceMap[src] == dst) {
                        // 移除这条复制指令
                        it = bb->erase(it);
                        continue;
                    }
                }
            }
            ++it;
        }
    }
}

// 辅助函数实现
bool RegAllocChaitin::isPhysicalReg(unsigned reg) const {
    return reg < 32;  // RISC-V有32个物理寄存器
}

bool RegAllocChaitin::isCallerSaved(unsigned reg) const {
    // t0-t6: x5-x7, x28-x31
    // a0-a7: x10-x17
    return (reg >= 5 && reg <= 7) || (reg >= 10 && reg <= 17) ||
           (reg >= 28 && reg <= 31);
}

bool RegAllocChaitin::isCalleeSaved(unsigned reg) const {
    // s0-s11: x8-x9, x18-x27
    return (reg >= 8 && reg <= 9) || (reg >= 18 && reg <= 27);
}

bool RegAllocChaitin::isArgumentReg(unsigned reg) const {
    // a0-a7: x10-x17
    return reg >= 10 && reg <= 17;
}

bool RegAllocChaitin::isReturnReg(unsigned reg) const {
    // a0-a1: x10-x11
    return reg == 10 || reg == 11;
}

bool RegAllocChaitin::isReservedReg(unsigned reg) const {
    // x0 (zero), x1 (ra), x2 (sp), x3 (gp), x4 (tp)
    return reg <= 4;
}

std::vector<unsigned> RegAllocChaitin::getABIPreferredRegs(
    unsigned virtualReg) const {
    std::vector<unsigned> preferredRegs;

    // 如果虚拟寄存器有特定的ABI约束，优先考虑相应的物理寄存器
    if (physicalConstraints.find(virtualReg) != physicalConstraints.end()) {
        for (unsigned physReg : physicalConstraints.at(virtualReg)) {
            if (!isReservedReg(physReg)) {
                preferredRegs.push_back(physReg);
            }
        }
    }

    // 根据使用模式推荐寄存器
    if (isUsedAsArgument(virtualReg)) {
        // 优先使用参数寄存器
        for (unsigned reg = 10; reg <= 17; ++reg) {  // a0-a7
            if (std::find(preferredRegs.begin(), preferredRegs.end(), reg) ==
                preferredRegs.end()) {
                preferredRegs.push_back(reg);
            }
        }
    }

    if (isUsedAcrossCalls(virtualReg)) {
        // 优先使用被调用者保存寄存器
        for (unsigned reg = 8; reg <= 9; ++reg) {  // s0-s1
            if (std::find(preferredRegs.begin(), preferredRegs.end(), reg) ==
                preferredRegs.end()) {
                preferredRegs.push_back(reg);
            }
        }
        for (unsigned reg = 18; reg <= 27; ++reg) {  // s2-s11
            if (std::find(preferredRegs.begin(), preferredRegs.end(), reg) ==
                preferredRegs.end()) {
                preferredRegs.push_back(reg);
            }
        }
    } else {
        // 优先使用调用者保存寄存器
        for (unsigned reg = 5; reg <= 7; ++reg) {  // t0-t2
            if (std::find(preferredRegs.begin(), preferredRegs.end(), reg) ==
                preferredRegs.end()) {
                preferredRegs.push_back(reg);
            }
        }
        for (unsigned reg = 28; reg <= 31; ++reg) {  // t3-t6
            if (std::find(preferredRegs.begin(), preferredRegs.end(), reg) ==
                preferredRegs.end()) {
                preferredRegs.push_back(reg);
            }
        }
    }

    return preferredRegs;
}

bool RegAllocChaitin::isUsedAsArgument(unsigned virtualReg) const {
    // 检查虚拟寄存器是否在函数调用中用作参数
    for (auto& bb : *function) {
        for (const auto& inst : *bb) {
            if (inst->isCallInstr()) {
                auto usedRegs = getUsedRegs(inst.get());
                if (std::find(usedRegs.begin(), usedRegs.end(), virtualReg) !=
                    usedRegs.end()) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool RegAllocChaitin::isUsedAcrossCalls(unsigned virtualReg) const {
    // 检查虚拟寄存器的生命周期是否跨越函数调用
    for (auto& bb : *function) {
        const auto& liveIn = livenessInfo.at(bb.get()).liveIn;
        const auto& liveOut = livenessInfo.at(bb.get()).liveOut;

        bool regLive = (liveIn.find(virtualReg) != liveIn.end()) ||
                       (liveOut.find(virtualReg) != liveOut.end());

        if (regLive) {
            for (const auto& inst : *bb) {
                if (inst->isCallInstr()) {
                    return true;
                }
            }
        }
    }
    return false;
}

// TODO: once is enough
void RegAllocChaitin::initializeABIConstraints() {
    // 设置可用寄存器列表（排除保留寄存器）
    availableRegs.clear();

    // 添加调用者保存寄存器
    for (unsigned reg = 5; reg <= 7; ++reg) {  // t0-t2
        availableRegs.push_back(reg);
    }

    for (unsigned reg = 10; reg <= 17; ++reg) {  // a0-a7
        availableRegs.push_back(reg);
    }
    for (unsigned reg = 28; reg <= 31; ++reg) {  // t3-t6
        availableRegs.push_back(reg);
    }

    // 添加被调用者保存寄存器
    for (unsigned reg = 8; reg <= 9; ++reg) {  // s0-s1
        availableRegs.push_back(reg);
    }
    for (unsigned reg = 18; reg <= 27; ++reg) {  // s2-s11
        availableRegs.push_back(reg);
    }

    // 根据函数特征设置特定约束
    setFunctionSpecificConstraints();
}

void RegAllocChaitin::setFunctionSpecificConstraints() {
    // 分析函数签名和调用模式，设置ABI相关约束

    // 1. 分析函数参数约束
    setParameterConstraints();

    // 2. 分析函数返回值约束
    setReturnValueConstraints();

    // 3. 分析函数调用约束
    setCallSiteConstraints();

    // 4. 分析特殊用途寄存器约束
    setSpecialRegisterConstraints();
}

void RegAllocChaitin::setParameterConstraints() {
    if (function->empty()) return;
    BasicBlock* entryBlock = function->begin()->get();
    if (!entryBlock) return;

    // 首先标记哪些参数寄存器实际被使用
    std::set<unsigned> usedParamRegs;

    // 扫描整个函数，识别哪些参数寄存器被使用
    for (auto& bb : *function) {
        for (const auto& inst : *bb) {
            auto usedRegs = getUsedRegs(inst.get());
            for (unsigned reg : usedRegs) {
                if (reg >= 10 && reg <= 17) {  // a0-a7
                    usedParamRegs.insert(reg);
                }
            }
        }
    }

    // 为实际使用的参数寄存器创建专门的虚拟寄存器映射
    std::map<unsigned, unsigned> paramToVirtual;

    int paramIndex = 0;
    for (const auto& inst : *entryBlock) {
        if (inst->isCopyInstr() || inst->isParameterMove()) {
            const auto& operands = inst->getOperands();
            if (operands.size() >= 2 && operands[0]->isReg() &&
                operands[1]->isReg()) {
                unsigned dstReg = operands[0]->getRegNum();
                unsigned srcReg = operands[1]->getRegNum();

                if (srcReg >= 10 && srcReg <= 17 && !isPhysicalReg(dstReg)) {
                    // 强制约束：参数虚拟寄存器必须分配到对应的参数物理寄存器
                    addStrongPhysicalConstraint(dstReg, srcReg);
                    paramToVirtual[srcReg] = dstReg;
                }
            }
        }
        if (++paramIndex > 8) break;  // 只处理前8个参数
    }

    // 确保未被虚拟寄存器接管的参数寄存器不被分配给其他虚拟寄存器
    for (unsigned paramReg = 10; paramReg <= 17; ++paramReg) {
        if (usedParamRegs.count(paramReg) && !paramToVirtual.count(paramReg)) {
            // 这个参数寄存器被直接使用，需要保护
            addReservedPhysicalReg(paramReg);
        }
    }
}

void RegAllocChaitin::setReturnValueConstraints() {
    // 查找函数中的返回指令
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            if (inst->isReturnInstr()) {
                // 分析返回指令前的值准备
                auto it = std::find_if(
                    bb->begin(), bb->end(),
                    [&inst](const auto& i) { return i.get() == inst.get(); });

                if (it != bb->begin()) {
                    // 检查返回值准备指令
                    auto prevIt = std::prev(it);
                    Instruction* prevInst = prevIt->get();

                    // 查找向a0, a1赋值的指令
                    auto definedRegs = getDefinedRegs(prevInst);
                    auto usedRegs = getUsedRegs(prevInst);

                    // 如果指令定义了a0或a1
                    for (unsigned reg : definedRegs) {
                        if (reg == 10 || reg == 11) {  // a0 或 a1
                            // 查找产生返回值的虚拟寄存器
                            for (unsigned srcReg : usedRegs) {
                                if (!isPhysicalReg(srcReg)) {
                                    addPhysicalConstraint(srcReg, reg);

                                    std::cout << "Added return value "
                                                 "constraint: virtual reg "
                                              << srcReg << " -> physical reg "
                                              << reg << " (a" << (reg - 10)
                                              << ")" << std::endl;
                                }
                            }
                        }
                    }

                    // 检查复制到返回寄存器的指令
                    if (prevInst->isCopyInstr()) {
                        const auto& operands = prevInst->getOperands();
                        if (operands.size() >= 2 && operands[0]->isReg() &&
                            operands[1]->isReg()) {
                            unsigned dstReg = operands[0]->getRegNum();
                            unsigned srcReg = operands[1]->getRegNum();

                            // 如果目标是a0或a1，源是虚拟寄存器
                            if ((dstReg == 10 || dstReg == 11) &&
                                !isPhysicalReg(srcReg)) {
                                addPhysicalConstraint(srcReg, dstReg);
                            }
                        }
                    }
                }
            }
        }
    }
}

void RegAllocChaitin::setCallSiteConstraints() {
    // 分析函数调用点的约束
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            if (inst->isCallInstr()) {
                // 分析调用前的参数准备
                setPreCallConstraints(bb.get(), inst.get());

                // 分析调用后的返回值处理
                setPostCallConstraints(bb.get(), inst.get());
            }
        }
    }
}

void RegAllocChaitin::setPreCallConstraints(BasicBlock* bb,
                                            Instruction* callInst) {
    // 查找调用指令在基本块中的位置
    auto callIt = std::find_if(
        bb->begin(), bb->end(),
        [callInst](const auto& inst) { return inst.get() == callInst; });

    if (callIt == bb->end()) return;

    // 向前扫描，查找参数准备指令
    int paramCount = 0;
    for (auto it = std::make_reverse_iterator(callIt);
         it != bb->rend() && paramCount < 8; ++it) {
        Instruction* inst = it->get();
        auto definedRegs = getDefinedRegs(inst);

        // 查找向参数寄存器a0-a7赋值的指令
        for (unsigned reg : definedRegs) {
            if (reg >= 10 && reg <= 17) {  // a0-a7
                auto usedRegs = getUsedRegs(inst);
                for (unsigned srcReg : usedRegs) {
                    if (!isPhysicalReg(srcReg)) {
                        addPhysicalConstraint(srcReg, reg);

                        std::cout
                            << "Added call argument constraint: virtual reg "
                            << srcReg << " -> physical reg " << reg << " (a"
                            << (reg - 10) << ")" << std::endl;
                    }
                }
                paramCount++;
            }
        }
    }
}

void RegAllocChaitin::setPostCallConstraints(BasicBlock* bb,
                                             Instruction* callInst) {
    // 查找调用指令在基本块中的位置
    auto callIt = std::find_if(
        bb->begin(), bb->end(),
        [callInst](const auto& inst) { return inst.get() == callInst; });

    if (callIt == bb->end()) return;

    // 获取调用前活跃的虚拟寄存器
    std::unordered_set<unsigned> liveBeforeCall;

    // 向前扫描到调用点，收集活跃的虚拟寄存器
    for (auto it = bb->begin(); it != callIt; ++it) {
        auto usedRegs = getUsedRegs(it->get());
        auto definedRegs = getDefinedRegs(it->get());

        for (unsigned reg : usedRegs) {
            if (reg == 10 || reg == 11) {  // a0 或 a1
                // 如果指令将返回值复制到虚拟寄存器
                for (unsigned dstReg : definedRegs) {
                    if (!isPhysicalReg(dstReg)) {
                        addPhysicalConstraint(dstReg, reg);

                        std::cout
                            << "Added call return constraint: virtual reg "
                            << dstReg << " -> physical reg " << reg << " (a"
                            << (reg - 10) << ")" << std::endl;
                    }
                }
            }
        }
    }
}

void RegAllocChaitin::setSpecialRegisterConstraints() {
    // 设置特殊用途寄存器的约束

    // 1. 栈指针约束 (sp = x2)
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            auto usedRegs = getUsedRegs(inst.get());
            auto definedRegs = getDefinedRegs(inst.get());

            // 禁止虚拟寄存器使用保留寄存器
            for (unsigned reg : usedRegs) {
                if (!isPhysicalReg(reg)) {
                    // 确保虚拟寄存器不会分配到保留寄存器
                    for (unsigned reservedReg = 0; reservedReg <= 4;
                         ++reservedReg) {
                        addPhysicalConstraint(reg, reservedReg);
                    }
                }
            }

            for (unsigned reg : definedRegs) {
                if (!isPhysicalReg(reg)) {
                    // 确保虚拟寄存器不会分配到保留寄存器
                    for (unsigned reservedReg = 0; reservedReg <= 4;
                         ++reservedReg) {
                        addPhysicalConstraint(reg, reservedReg);
                    }
                }
            }
        }
    }

    // 2. 帧指针约束 (如果使用帧指针)
    if (usesFramePointer()) {
        // s0/fp约束处理
        setFramePointerConstraints();
    }

    // 3. 返回地址寄存器约束 (ra = x1)
    setReturnAddressConstraints();
}

bool RegAllocChaitin::usesFramePointer() const {
    // 检查函数是否使用帧指针
    // 简单启发式：如果函数有复杂的栈操作或大量局部变量
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            // 查找设置帧指针的指令模式
            if (inst->isFrameSetup() ||
                (inst->isCopyInstr() && inst->involvesStackPointer())) {
                return true;
            }
        }
    }
    return false;
}

void RegAllocChaitin::setFramePointerConstraints() {
    // 为使用帧指针的虚拟寄存器添加s0约束
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            if (inst->isFrameSetup() || inst->isFramePointerRelated()) {
                auto definedRegs = getDefinedRegs(inst.get());
                for (unsigned reg : definedRegs) {
                    if (!isPhysicalReg(reg)) {
                        addPhysicalConstraint(reg, 8);  // s0/fp = x8
                    }
                }
            }
        }
    }
}

void RegAllocChaitin::setReturnAddressConstraints() {
    // 处理返回地址寄存器的约束
    for (auto& bb : *function) {
        for (auto& inst : *bb) {
            if (inst->isCallInstr() || inst->isReturnInstr()) {
                // 调用和返回指令会使用ra寄存器
                auto usedRegs = getUsedRegs(inst.get());
                auto definedRegs = getDefinedRegs(inst.get());

                // 确保相关虚拟寄存器不会冲突ra
                for (unsigned reg : usedRegs) {
                    if (!isPhysicalReg(reg)) {
                        // 避免与ra冲突
                        addPhysicalConstraint(reg, 1);  // ra = x1
                    }
                }
            }
        }
    }
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

/// Print
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
        if (!isPhysicalReg(virtualReg)) {
            std::cout << "Virtual register " << virtualReg
                      << " -> Physical register " << physicalReg << " ("
                      << ABI::getABINameFromRegNum(physicalReg) << ")\n";
        }
    }

    if (!spilledRegs.empty()) {
        std::cout << "Spilled registers: ";
        for (unsigned reg : spilledRegs) {
            std::cout << reg << " ";
        }
        std::cout << "\n";
    }
}

// 打印合并结果
void RegAllocChaitin::printCoalesceResult() const {
    if (!coalesceMap.empty()) {
        std::cout << "Register Coalescing Result:\n";
        for (const auto& [src, dst] : coalesceMap) {
            std::cout << "Register " << src << " coalesced into " << dst
                      << std::endl;
        }
        std::cout << std::endl;
    }
}

}  // namespace riscv64
