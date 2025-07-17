#include "RegAllocChaitin.h"

#include <algorithm>
#include <iostream>
#include <queue>
#include <stack>

#include "StackFrameManager.h"
namespace riscv64 {

/// Entry
void RegAllocChaitin::allocateRegisters() {
    computeLiveness();

    buildInterferenceGraph();

    performCoalescing();

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

void RegAllocChaitin::updateDegreeAfterCoalesce(unsigned merged, unsigned eliminated) {
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
            if (mergedNode->neighbors.find(neighbor) == mergedNode->neighbors.end()) {
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
        if (neighbor != merged && degreeCache.find(neighbor) != degreeCache.end()) {
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
                        if (degreeCache[neighbor] < static_cast<int>(availableRegs.size())) {
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


bool RegAllocChaitin::attemptColoring(const std::vector<unsigned>& order) {
    // 预着色物理寄存器
    for (auto& [regNum, node] : interferenceGraph) {
        if (node->isPrecolored) {
            node->color = regNum;
            virtualToPhysical[regNum] = regNum; // 确保物理寄存器有映射
        }
    }

    // 按顺序为每个虚拟寄存器着色
    for (unsigned regNum : order) {
        if (spilledRegs.find(regNum) != spilledRegs.end()) {
            continue;  // 跳过溢出的寄存器
        }

        // 跳过已经被合并的寄存器
        if (coalescedRegs.find(regNum) != coalescedRegs.end()) {
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

    // 为被合并的寄存器建立映射关系
    for (const auto& [src, dst] : coalesceMap) {
        unsigned finalDst = getFinalCoalescedReg(dst);
        if (virtualToPhysical.find(finalDst) != virtualToPhysical.end()) {
            virtualToPhysical[src] = virtualToPhysical[finalDst];
        }
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
    // 获取栈帧管理器
    StackFrameManager stackManager(function);

    // 为溢出寄存器分配栈槽
    stackManager.allocateSpillSlot(reg);
    int spillOffset = stackManager.getSpillSlotOffset(reg);

    for (auto& bb : *function) {
        for (auto it = bb->begin(); it != bb->end(); ++it) {
            Instruction* inst = it->get();

            // 检查指令是否使用了溢出的寄存器
            auto usedRegs = getUsedRegs(inst);
            if (std::find(usedRegs.begin(), usedRegs.end(), reg) !=
                usedRegs.end()) {
                // 在指令前插入加载
                auto loadInst = std::make_unique<Instruction>(LD);
                loadInst->addOperand(
                    std::make_unique<RegisterOperand>(reg, false));
                loadInst->addOperand(std::make_unique<MemoryOperand>(
                    std::make_unique<RegisterOperand>(2, false),  // sp
                    std::make_unique<ImmediateOperand>(spillOffset)));
                it = bb->insert(it, std::move(loadInst));
                ++it;
            }

            // 检查指令是否定义了溢出的寄存器
            auto definedRegs = getDefinedRegs(inst);
            if (std::find(definedRegs.begin(), definedRegs.end(), reg) !=
                definedRegs.end()) {
                // 在指令后插入存储
                auto storeInst = std::make_unique<Instruction>(SD);
                storeInst->addOperand(
                    std::make_unique<RegisterOperand>(reg, false));
                storeInst->addOperand(std::make_unique<MemoryOperand>(
                    std::make_unique<RegisterOperand>(2, false),  // sp
                    std::make_unique<ImmediateOperand>(spillOffset)));
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
    const auto& operands = inst->getOperands();
    for (const auto& operand : operands) {
        if (operand->isReg()) {
            RegisterOperand* regOp = static_cast<RegisterOperand*>(operand.get());
            if (regOp->isVirtual()) {
                unsigned virtualReg = regOp->getRegNum();
                
                // 首先检查是否被合并，如果被合并则使用合并后的寄存器
                unsigned finalReg = getFinalCoalescedReg(virtualReg);
                
                // 然后查找物理寄存器映射
                if (virtualToPhysical.find(finalReg) != virtualToPhysical.end()) {
                    regOp->setPhysicalReg(virtualToPhysical[finalReg]);
                } else {
                    // 如果找不到映射，可能是被合并到了物理寄存器
                    if (isPhysicalReg(finalReg)) {
                        regOp->setPhysicalReg(finalReg);
                    }
                }
            }
        }
    }
}

// 获取最终合并后的寄存器
unsigned RegAllocChaitin::getFinalCoalescedReg(unsigned reg) {
    unsigned current = reg;
    std::unordered_set<unsigned> visited; // 防止循环
    
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

    // 2. 循环嵌套深度权重 (0-50)
    // int loopDepth = getLoopNestingDepth(bb);
    // priority += loopDepth * 50;

    // 3. 寄存器使用频率权重 (0-30)
    int srcUsageCount = getRegisterUsageCount(src);
    int dstUsageCount = getRegisterUsageCount(dst);
    priority += (srcUsageCount + dstUsageCount) * 2;

    // 4. 冲突图度数权重 (度数越小优先级越高) (0-20)
    int srcDegree = getRegisterDegree(src);
    int dstDegree = getRegisterDegree(dst);
    int avgDegree = (srcDegree + dstDegree) / 2;
    priority += std::max(0, 20 - avgDegree);

    // 5. 生命周期重叠度权重 (重叠越少优先级越高) (0-15)
    int lifeOverlap = calculateLifetimeOverlap(src, dst);
    priority += std::max(0, 15 - lifeOverlap);

    // 6. 寄存器压力权重 (0-10)
    int regPressure = getRegisterPressure(bb);
    if (regPressure > static_cast<int>(availableRegs.size() * 0.8)) {
        priority += 10;  // 高寄存器压力时更倾向于合并
    }

    // 7. 物理寄存器偏好权重 (0-25)
    priority += calculatePhysicalRegPreference(src, dst);

    // 8. 复制指令消除收益权重 (0-5)
    priority += 5;  // 消除一条复制指令的基本收益

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

    // 2. 检查是否存在冲突
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
