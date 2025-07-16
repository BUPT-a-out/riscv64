#include "RegAllocGreedy.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_set>

#include "ABI.h"

namespace riscv64 {

RegAllocGreedy::RegAllocGreedy(Module& module)
    : module_(module),
      allocQueue_([](LiveInterval* a, LiveInterval* b) {
          return a->weight < b->weight;
      }),
      nextSpillSlot_(0),
      numSpills_(0),
      numReloads_(0),
      instructionNumber_(0) {}

bool RegAllocGreedy::run() {
    initialize();

    // 检查是否有虚拟寄存器需要分配
    if (liveIntervals_.empty()) {
        return false;
    }

    calculateCSRCosts();

    // 将所有活跃区间加入分配队列
    for (auto& [virtReg, interval] : liveIntervals_) {
        allocQueue_.push(interval.get());
        regStages_[virtReg] = RegAllocStage::RS_New;
    }

    // 主分配循环
    while (!allocQueue_.empty()) {
        LiveInterval* virtReg = allocQueue_.top();
        allocQueue_.pop();
        selectOrSplit(virtReg);
    }

    tryHintRecoloring();
    postOptimization();
    reportStatistics();

    return true;
}

void RegAllocGreedy::initialize() {
    initializePhysicalRegs();
    calculateLiveIntervals();
}

void RegAllocGreedy::initializePhysicalRegs() {
    // RISC-V有32个整数寄存器 (x0-x31)
    // x0始终为0，不可分配
    // x1(ra), x2(sp), x3(gp), x4(tp) 有特殊用途
    // 可分配的寄存器：x5-x31

    std::vector<unsigned> allocatableRegs = {
        5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

    for (unsigned regNum : allocatableRegs) {
        physicalRegs_.emplace_back(regNum);
    }
}

void RegAllocGreedy::calculateLiveIntervals() {
    // 为每个函数计算活跃区间
    for (auto& function : module_) {
        instructionNumber_ = 0;
        std::unordered_map<RegisterOperand*, int> firstDef;
        std::unordered_map<RegisterOperand*, int> lastUse;

        // 第一遍：收集所有虚拟寄存器的定义和使用
        for (auto& bb : *function) {
            for (auto& inst : *bb) {
                instructionNumber_++;
                instructionMap_[inst.get()] = instructionNumber_;

                // 检查所有操作数
                for (size_t i = 0; i < inst->getOprandCount(); ++i) {
                    MachineOperand* operand = inst->getOperand(i);
                    if (operand->getType() == OperandType::Register) {
                        RegisterOperand* regOp =
                            static_cast<RegisterOperand*>(operand);

                        if (regOp->isVirtual()) {
                            // 如果是第一次遇到这个虚拟寄存器
                            if (liveIntervals_.find(regOp) ==
                                liveIntervals_.end()) {
                                liveIntervals_[regOp] =
                                    std::make_unique<LiveInterval>(regOp);
                                firstDef[regOp] = instructionNumber_;
                            }

                            // 更新最后使用点
                            lastUse[regOp] = instructionNumber_;

                            // 记录定义和使用关系
                            if (i == 0) {
                                defMap_[regOp].push_back(inst.get());
                            } else {
                                useMap_[regOp].push_back(inst.get());
                            }
                        }
                    }
                }
            }
        }

        // 第二遍：构建活跃区间
        for (auto& [virtReg, interval] : liveIntervals_) {
            if (firstDef.count(virtReg) && lastUse.count(virtReg)) {
                int start = firstDef[virtReg];
                int end = lastUse[virtReg];
                interval->ranges.push_back({start, end});
                interval->weight = end - start + 1;  // 权重按区间长度计算
            }
        }
    }
}

void RegAllocGreedy::calculateCSRCosts() {
    // 计算调用者保存寄存器的使用成本
    // 被调用者保存寄存器在函数入口和出口需要保存/恢复
    for (auto& function : module_) {
        bool hasCall = false;

        // 检查函数是否包含调用指令
        for (auto& bb : *function) {
            for (auto& inst : *bb) {
                if (inst->isCallInstr()) {
                    hasCall = true;
                    break;
                }
            }
            if (hasCall) break;
        }

        // 如果函数包含调用，CSR成本较高
        csrCost_ = hasCall ? 10.0f : 1.0f;
    }
}

void RegAllocGreedy::selectOrSplit(LiveInterval* virtReg) {
    // Stage 1: 尝试直接分配
    if (auto physReg = tryDirectAssign(virtReg)) {
        assignRegister(virtReg, *physReg);
        return;
    }

    // Stage 2: 尝试驱逐
    if (regStages_[virtReg->virtReg] != RegAllocStage::RS_Split) {
        if (auto physReg = tryEviction(virtReg)) {
            assignRegister(virtReg, *physReg);
            return;
        }
    }

    // Stage 3: 尝试分割
    if (regStages_[virtReg->virtReg] == RegAllocStage::RS_New) {
        regStages_[virtReg->virtReg] = RegAllocStage::RS_Split;
        allocQueue_.push(virtReg);  // 推迟处理
        return;
    } else if (trySplit(virtReg)) {
        return;
    }

    // Stage 4: 最后机会重着色
    if (virtReg->spillable) {
        if (auto physReg = tryLastChanceRecoloring(virtReg)) {
            assignRegister(virtReg, *physReg);
            return;
        }
    }

    // Final: 溢出
    spillInterval(virtReg);
}

std::optional<unsigned> RegAllocGreedy::tryDirectAssign(LiveInterval* virtReg) {
    auto allocOrder = getAllocationOrder(virtReg);

    for (unsigned physReg : allocOrder) {
        auto& reg = physicalRegs_[physReg - 5];  // 调整索引
        if (!reg.allocated) {
            return physReg;
        }
    }

    return std::nullopt;
}

std::optional<unsigned> RegAllocGreedy::tryEviction(LiveInterval* virtReg) {
    auto allocOrder = getAllocationOrder(virtReg);

    for (unsigned physReg : allocOrder) {
        auto interfering = getInterferingIntervals(virtReg, physReg);

        if (interfering.empty()) {
            continue;
        }

        // 计算驱逐成本
        int evictCost = 0;
        bool canEvictAll = true;

        for (LiveInterval* interval : interfering) {
            if (!canEvictInterval(interval)) {
                canEvictAll = false;
                break;
            }
            evictCost += calculateEvictionCost(interval);
        }

        if (canEvictAll && evictCost < virtReg->weight) {
            // 执行驱逐
            for (LiveInterval* interval : interfering) {
                freeRegister(physReg);
                regStages_[interval->virtReg] = RegAllocStage::RS_New;
                allocQueue_.push(interval);
            }
            return physReg;
        }
    }

    return std::nullopt;
}

bool RegAllocGreedy::trySplit(LiveInterval* virtReg) {
    // 尝试不同的分割策略
    if (tryRegionSplit(virtReg)) return true;
    if (tryBlockSplit(virtReg)) return true;
    if (tryInstructionSplit(virtReg)) return true;

    return false;
}

std::optional<unsigned> RegAllocGreedy::tryLastChanceRecoloring(
    LiveInterval* virtReg) {
    auto allocOrder = getAllocationOrder(virtReg);

    for (unsigned physReg : allocOrder) {
        auto interfering = getInterferingIntervals(virtReg, physReg);

        bool canRecolorAll = true;
        for (LiveInterval* interval : interfering) {
            if (!canRecolorInterval(interval, physReg)) {
                canRecolorAll = false;
                break;
            }
        }

        if (canRecolorAll) {
            // 重着色所有冲突的区间
            for (LiveInterval* interval : interfering) {
                unsigned newReg = findBestPhysicalReg(interval);
                recolorInterval(interval, newReg);
            }
            return physReg;
        }
    }

    return std::nullopt;
}

void RegAllocGreedy::spillInterval(LiveInterval* virtReg) {
    spilledRegs_.insert(virtReg->virtReg);
    spillSlots_[virtReg->virtReg] = nextSpillSlot_++;
    generateSpillCode(virtReg);
    numSpills_++;
}

// 分割策略实现
bool RegAllocGreedy::tryRegionSplit(LiveInterval* virtReg) {
    // 在函数边界分割 - 简化实现
    if (virtReg->ranges.size() > 1) {
        // 如果有多个范围，在中间分割
        size_t mid = virtReg->ranges.size() / 2;

        // 创建新的活跃区间
        auto newInterval = std::make_unique<LiveInterval>(virtReg->virtReg);
        newInterval->ranges.assign(virtReg->ranges.begin() + mid,
                                   virtReg->ranges.end());
        newInterval->weight = 0;
        for (const auto& range : newInterval->ranges) {
            newInterval->weight += range.second - range.first + 1;
        }

        // 更新原区间
        virtReg->ranges.resize(mid);
        virtReg->weight = 0;
        for (const auto& range : virtReg->ranges) {
            virtReg->weight += range.second - range.first + 1;
        }

        // 添加新区间到队列
        liveIntervals_[virtReg->virtReg] = std::move(newInterval);
        allocQueue_.push(liveIntervals_[virtReg->virtReg].get());
        regStages_[virtReg->virtReg] = RegAllocStage::RS_New;

        return true;
    }
    return false;
}

bool RegAllocGreedy::tryBlockSplit(LiveInterval* virtReg) {
    // 在基本块边界分割
    if (virtReg->ranges.size() == 1 &&
        virtReg->ranges[0].second - virtReg->ranges[0].first > 20) {
        // 如果区间太长，分割成两部分
        auto& range = virtReg->ranges[0];
        int mid = (range.first + range.second) / 2;

        // 创建新区间
        auto newInterval = std::make_unique<LiveInterval>(virtReg->virtReg);
        newInterval->ranges.push_back({mid + 1, range.second});
        newInterval->weight = range.second - mid;

        // 更新原区间
        range.second = mid;
        virtReg->weight = mid - range.first + 1;

        // 添加新区间到队列
        liveIntervals_[virtReg->virtReg] = std::move(newInterval);
        allocQueue_.push(liveIntervals_[virtReg->virtReg].get());
        regStages_[virtReg->virtReg] = RegAllocStage::RS_New;

        return true;
    }
    return false;
}

bool RegAllocGreedy::tryInstructionSplit(LiveInterval* virtReg) {
    // 在指令边界分割 - 围绕使用点分割
    if (!virtReg->ranges.empty()) {
        auto& range = virtReg->ranges[0];
        if (range.second - range.first > 5) {
            // 在使用点周围分割
            int splitPoint = range.first + (range.second - range.first) / 3;

            // 创建新区间
            auto newInterval = std::make_unique<LiveInterval>(virtReg->virtReg);
            newInterval->ranges.push_back({splitPoint, range.second});
            newInterval->weight = range.second - splitPoint + 1;

            // 更新原区间
            range.second = splitPoint - 1;
            virtReg->weight = splitPoint - range.first;

            // 添加新区间到队列
            liveIntervals_[virtReg->virtReg] = std::move(newInterval);
            allocQueue_.push(liveIntervals_[virtReg->virtReg].get());
            regStages_[virtReg->virtReg] = RegAllocStage::RS_New;

            return true;
        }
    }
    return false;
}

// 辅助函数实现
std::vector<LiveInterval*> RegAllocGreedy::getInterferingIntervals(
    LiveInterval* virtReg, unsigned physReg) {
    std::vector<LiveInterval*> interfering;

    auto& reg = physicalRegs_[physReg - 5];
    if (reg.allocated && reg.currentInterval) {
        if (virtReg->interferesWith(*reg.currentInterval)) {
            interfering.push_back(reg.currentInterval);
        }
    }

    return interfering;
}

bool RegAllocGreedy::canEvictInterval(LiveInterval* interval) {
    return interval->spillable &&
           regStages_[interval->virtReg] != RegAllocStage::RS_Spill;
}

int RegAllocGreedy::calculateEvictionCost(LiveInterval* interval) {
    // 基于权重和阶段计算驱逐成本
    int baseCost = interval->weight;

    // 已经被分割的区间驱逐成本更高
    if (regStages_[interval->virtReg] == RegAllocStage::RS_Split) {
        baseCost *= 2;
    }

    return baseCost;
}

bool RegAllocGreedy::canRecolorInterval(LiveInterval* interval,
                                        unsigned newReg) {
    // 检查是否可以将区间重新着色到新寄存器
    if (newReg < 5 || newReg > 31) return false;

    auto& reg = physicalRegs_[newReg - 5];
    if (reg.allocated && reg.currentInterval) {
        return !interval->interferesWith(*reg.currentInterval);
    }

    return !reg.allocated;
}

void RegAllocGreedy::assignRegister(LiveInterval* virtReg, unsigned physReg) {
    auto& reg = physicalRegs_[physReg - 5];
    reg.allocated = true;
    reg.currentInterval = virtReg;

    // 记录分配映射
    virtualToPhysical_[virtReg->virtReg] = physReg;

    // 更新所有使用该虚拟寄存器的指令
    auto uses = getUsesAndDefs(virtReg->virtReg);
    for (Instruction* inst : uses) {
        updateInstruction(inst, virtReg->virtReg, physReg);
    }
}

void RegAllocGreedy::freeRegister(unsigned physReg) {
    auto& reg = physicalRegs_[physReg - 5];
    if (reg.currentInterval) {
        virtualToPhysical_.erase(reg.currentInterval->virtReg);
    }
    reg.allocated = false;
    reg.currentInterval = nullptr;
}

void RegAllocGreedy::generateSpillCode(LiveInterval* virtReg) {
    // 生成溢出和重新加载代码
    auto uses = getUsesAndDefs(virtReg->virtReg);

    for (Instruction* inst : uses) {
        // 在定义处插入溢出代码
        if (isDefinition(inst, virtReg->virtReg)) {
            insertSpillAtDef(virtReg->virtReg, inst);
        }

        // 在使用处插入重新加载代码
        if (isUse(inst, virtReg->virtReg)) {
            insertReloadAtUse(virtReg->virtReg, inst);
        }
    }
}

void RegAllocGreedy::insertSpillAtDef(RegisterOperand* virtReg,
                                      Instruction* defInst) {
    // 1. Get the basic block containing the definition instruction
    BasicBlock* bb = defInst->getParent();
    if (!bb) {
        throw std::runtime_error(
            "Definition instruction has no parent basic block");
    }

    // 2. Find the position to insert after the definition
    auto insertPos =
        std::find_if(bb->begin(), bb->end(),
                     [defInst](const std::unique_ptr<Instruction>& inst) {
                         return inst.get() == defInst;
                     });

    if (insertPos == bb->end()) {
        throw std::runtime_error(
            "Definition instruction not found in its basic block");
    }
    ++insertPos;  // Insert after the definition

    // 3. Get spill slot information
    auto slotIt = spillSlots_.find(virtReg);
    if (slotIt == spillSlots_.end()) {
        throw std::runtime_error(
            "No spill slot allocated for virtual register");
    }
    int offset = slotIt->second * 8;  // 8 bytes per slot

    // 4. Create operands for the store instruction
    std::vector<std::unique_ptr<MachineOperand>> operands;

    // 4a. Source register operand
    unsigned physReg = 0;
    if (virtualToPhysical_.count(virtReg)) {
        physReg = virtualToPhysical_[virtReg];
        operands.push_back(std::make_unique<RegisterOperand>(physReg));
    } else {
        // No physical register assigned yet - need to use a temporary
        // Try to find an available temporary register (t0-t6)
        for (unsigned tempReg : {5, 6, 7, 28, 29, 30, 31}) {
            if (!physicalRegs_[tempReg - 5].allocated) {
                // Create move instruction first
                std::vector<std::unique_ptr<MachineOperand>> moveOperands;
                moveOperands.push_back(
                    std::make_unique<RegisterOperand>(tempReg));
                moveOperands.push_back(
                    std::make_unique<RegisterOperand>(*virtReg));

                auto moveInst = std::make_unique<Instruction>(
                    Opcode::MV, std::move(moveOperands));
                moveInst->setParent(bb);
                bb->insert(insertPos, std::move(moveInst));

                // Use this temp register for the spill
                physReg = tempReg;
                operands.push_back(std::make_unique<RegisterOperand>(physReg));
                break;
            }
        }

        if (physReg == 0) {
            throw std::runtime_error(
                "No available temporary registers for spilling");
        }
    }

    // 4b. Offset immediate operand
    operands.push_back(std::make_unique<ImmediateOperand>(offset));

    // 4c. Base register operand (stack pointer)
    operands.push_back(std::make_unique<RegisterOperand>(2));  // x2 = SP

    // 5. Create and insert the store instruction
    auto spillInst =
        std::make_unique<Instruction>(Opcode::SD, std::move(operands));
    spillInst->setParent(bb);
    bb->insert(insertPos, std::move(spillInst));

    // 6. Update statistics
    numSpills_++;
}

void RegAllocGreedy::insertReloadAtUse(RegisterOperand* virtReg,
                                       Instruction* useInst) {
    // 1. Get the basic block containing the use instruction
    BasicBlock* bb = useInst->getParent();
    if (!bb) {
        throw std::runtime_error("Use instruction has no parent basic block");
    }

    // 2. Find the position to insert before the use
    auto insertPos =
        std::find_if(bb->begin(), bb->end(),
                     [useInst](const std::unique_ptr<Instruction>& inst) {
                         return inst.get() == useInst;
                     });

    if (insertPos == bb->end()) {
        throw std::runtime_error(
            "Use instruction not found in its basic block");
    }

    // 3. Get spill slot information
    auto slotIt = spillSlots_.find(virtReg);
    if (slotIt == spillSlots_.end()) {
        throw std::runtime_error(
            "No spill slot allocated for virtual register");
    }
    int offset = slotIt->second * 8;  // 8 bytes per slot

    // 4. Create operands for the load instruction
    std::vector<std::unique_ptr<MachineOperand>> operands;

    // 4a. Destination register handling
    unsigned physReg = 0;
    if (virtualToPhysical_.count(virtReg)) {
        // Already assigned a physical register
        physReg = virtualToPhysical_[virtReg];
        operands.push_back(std::make_unique<RegisterOperand>(physReg));
    } else {
        // Need to use a temporary register
        // Try to find an available temporary register (t0-t6)
        for (unsigned tempReg : {5, 6, 7, 28, 29, 30, 31}) {
            if (!physicalRegs_[tempReg - 5].allocated) {
                physReg = tempReg;
                operands.push_back(std::make_unique<RegisterOperand>(physReg));
                break;
            }
        }

        if (physReg == 0) {
            throw std::runtime_error(
                "No available temporary registers for reloading");
        }
    }

    // 4b. Offset immediate operand
    operands.push_back(std::make_unique<ImmediateOperand>(offset));

    // 4c. Base register operand (stack pointer)
    operands.push_back(std::make_unique<RegisterOperand>(2));  // x2 = SP

    // 5. Create and insert the load instruction
    auto reloadInst =
        std::make_unique<Instruction>(Opcode::LD, std::move(operands));
    reloadInst->setParent(bb);
    insertPos = bb->insert(insertPos, std::move(reloadInst));

    // 6. If we used a temporary register, insert a move to the virtual register
    if (physReg != 0 && !virtualToPhysical_.count(virtReg)) {
        std::vector<std::unique_ptr<MachineOperand>> moveOperands;
        moveOperands.push_back(std::make_unique<RegisterOperand>(*virtReg));
        moveOperands.push_back(std::make_unique<RegisterOperand>(physReg));

        auto moveInst =
            std::make_unique<Instruction>(Opcode::MV, std::move(moveOperands));
        moveInst->setParent(bb);
        bb->insert(++insertPos, std::move(moveInst));
    }

    // 7. Update statistics
    numReloads_++;

    if (virtualToPhysical_.count(virtReg)) {
        updateInstruction(useInst, virtReg, physReg);
    }
}

bool RegAllocGreedy::tryHintRecoloring() {
    // 尝试基于提示的重着色
    // 寻找可以合并的拷贝指令
    bool changed = false;

    for (auto& function : module_) {
        for (auto& bb : *function) {
            for (auto& inst : *bb) {
                // 检查是否是拷贝指令
                if (inst->isCopyInstr()) {
                    // 尝试将源和目标分配到同一寄存器
                    if (tryCoalesceCopy(inst.get())) {
                        changed = true;
                    }
                }
            }
        }
    }

    return changed;
}

bool RegAllocGreedy::tryCoalesceCopy(Instruction* copyInst) {
    // 1. Get source and destination operands
    if (copyInst->getOprandCount() < 2) return false;

    MachineOperand* destOp = copyInst->getOperand(0);
    MachineOperand* srcOp = copyInst->getOperand(1);

    // 2. Verify both are register operands
    if (!destOp->isReg() || !srcOp->isReg()) return false;

    RegisterOperand* destReg = static_cast<RegisterOperand*>(destOp);
    RegisterOperand* srcReg = static_cast<RegisterOperand*>(srcOp);

    // 3. Check if both are virtual registers (or at least one is virtual)
    if (!destReg->isVirtual() && !srcReg->isVirtual()) {
        // Both physical - nothing to coalesce
        return false;
    }

    // 4. Get live intervals for both registers
    LiveInterval* destInterval = nullptr;
    LiveInterval* srcInterval = nullptr;

    if (destReg->isVirtual()) {
        auto it = liveIntervals_.find(destReg);
        if (it != liveIntervals_.end()) {
            destInterval = it->second.get();
        }
    }

    if (srcReg->isVirtual()) {
        auto it = liveIntervals_.find(srcReg);
        if (it != liveIntervals_.end()) {
            srcInterval = it->second.get();
        }
    }

    // 5. Check if we can coalesce these registers
    bool canCoalesce = false;
    unsigned coalescedReg = 0;

    if (destReg->isVirtual() && srcReg->isVirtual()) {
        // Case 1: Both virtual registers - check if they can share a register
        if (!destInterval || !srcInterval) return false;

        // Check if they don't interfere
        if (!destInterval->interferesWith(*srcInterval)) {
            // They can be coalesced - pick one to keep
            if (virtualToPhysical_.count(destReg)) {
                coalescedReg = virtualToPhysical_[destReg];
                canCoalesce = canRecolorInterval(srcInterval, coalescedReg);
            } else if (virtualToPhysical_.count(srcReg)) {
                coalescedReg = virtualToPhysical_[srcReg];
                canCoalesce = canRecolorInterval(destInterval, coalescedReg);
            } else {
                // Neither assigned yet - can coalesce when we assign
                canCoalesce = true;
            }
        }
    } else if (destReg->isVirtual() && !srcReg->isVirtual()) {
        // Case 2: Virtual dest, physical src
        if (!destInterval) return false;

        unsigned srcPhysReg = srcReg->getRegNum();
        canCoalesce = canRecolorInterval(destInterval, srcPhysReg);
        if (canCoalesce) coalescedReg = srcPhysReg;
    } else if (!destReg->isVirtual() && srcReg->isVirtual()) {
        // Case 3: Physical dest, virtual src
        if (!srcInterval) return false;

        unsigned destPhysReg = destReg->getRegNum();
        canCoalesce = canRecolorInterval(srcInterval, destPhysReg);
        if (canCoalesce) coalescedReg = destPhysReg;
    }

    // 6. If we can coalesce, perform the coalescing
    if (canCoalesce) {
        if (destReg->isVirtual() && srcReg->isVirtual()) {
            // Both virtual - merge into one
            if (virtualToPhysical_.count(destReg)) {
                // Dest already assigned - assign src to same reg
                unsigned destPhysReg = virtualToPhysical_[destReg];
                assignRegister(srcInterval, destPhysReg);
            } else if (virtualToPhysical_.count(srcReg)) {
                // Src already assigned - assign dest to same reg
                unsigned srcPhysReg = virtualToPhysical_[srcReg];
                assignRegister(destInterval, srcPhysReg);
            } else {
                // Neither assigned yet - will get same reg when assigned
                // We can just merge the live intervals
                destInterval->ranges.insert(destInterval->ranges.end(),
                                            srcInterval->ranges.begin(),
                                            srcInterval->ranges.end());
                // Update weight
                destInterval->weight += srcInterval->weight;

                // Remove src interval
                liveIntervals_.erase(srcReg);
            }
        } else if (destReg->isVirtual()) {
            // Virtual dest - assign to src physical reg
            assignRegister(destInterval, coalescedReg);
        } else if (srcReg->isVirtual()) {
            // Virtual src - assign to dest physical reg
            assignRegister(srcInterval, coalescedReg);
        }

        // Update the instruction operands to use the same register
        if (destReg->isVirtual()) {
            destReg->setPhysicalReg(coalescedReg);
        }
        if (srcReg->isVirtual()) {
            srcReg->setPhysicalReg(coalescedReg);
        }

        return true;
    }

    return false;
}

bool RegAllocGreedy::recolorInterval(LiveInterval* interval, unsigned newReg) {
    // 重新着色区间到新寄存器
    if (canRecolorInterval(interval, newReg)) {
        // 释放旧寄存器
        if (virtualToPhysical_.count(interval->virtReg)) {
            unsigned oldReg = virtualToPhysical_[interval->virtReg];
            freeRegister(oldReg);
        }

        // 分配新寄存器
        assignRegister(interval, newReg);
        return true;
    }
    return false;
}

void RegAllocGreedy::postOptimization() { coalesceRegisters(); }

void RegAllocGreedy::coalesceRegisters() {
    // 合并寄存器 - 处理拷贝指令
    for (auto& function : module_) {
        for (auto& bb : *function) {
            auto it = bb->begin();
            while (it != bb->end()) {
                auto& inst = *it;

                if (inst->isCopyInstr()) {
                    // 检查是否可以消除拷贝
                    if (canEliminateCopy(inst.get())) {
                        it = bb->erase(it);
                        continue;
                    }
                }
                ++it;
            }
        }
    }
}

bool RegAllocGreedy::canEliminateCopy(Instruction* copyInst) {
    // 获取拷贝指令的源和目标操作数
    if (copyInst->getOprandCount() < 2) return false;

    auto dest = copyInst->getOperand(0);
    auto src = copyInst->getOperand(1);

    // 必须是寄存器操作数
    if (!dest->isReg() || !src->isReg()) return false;

    // 获取物理寄存器编号（虚拟寄存器已分配物理寄存器）
    unsigned destReg = static_cast<RegisterOperand*>(dest)->getRegNum();
    unsigned srcReg = static_cast<RegisterOperand*>(src)->getRegNum();

    // 如果物理寄存器相同，可以消除拷贝
    return destReg == srcReg;
}

std::vector<unsigned> RegAllocGreedy::getAllocationOrder(
    LiveInterval* virtReg) const {
    (void)virtReg;  // Suppress unused parameter warning
    return RegAllocUtils::getRegisterAllocationOrder();
}

unsigned RegAllocGreedy::findBestPhysicalReg(LiveInterval* virtReg) const {
    auto allocOrder = getAllocationOrder(virtReg);

    // 寻找第一个可用的寄存器
    for (unsigned reg : allocOrder) {
        if (!physicalRegs_[reg - 5].allocated) {
            return reg;
        }
    }

    // 如果都被占用，返回第一个
    return allocOrder.empty() ? 5 : allocOrder[0];
}

void RegAllocGreedy::reportStatistics() {
    std::cout << "Register Allocation Statistics:\n";
    std::cout << "Spills: " << numSpills_ << "\n";
    std::cout << "Reloads: " << numReloads_ << "\n";
    std::cout << "Physical registers used: " << physicalRegs_.size() << "\n";
    std::cout << "Virtual registers allocated: " << virtualToPhysical_.size()
              << "\n";
    std::cout << "Spilled registers: " << spilledRegs_.size() << "\n";
}

void RegAllocGreedy::verifyAllocation() {
    bool allocationValid = true;
    std::unordered_map<unsigned, std::vector<LiveInterval*>> physRegAssignments;

    // 1. Verify physical register assignments
    for (const auto& [virtReg, physReg] : virtualToPhysical_) {
        // Check physical register is valid
        if (physReg < 5 || physReg > 31) {
            std::cerr << "ERROR: Invalid physical register " << physReg
                      << " assigned to virtual register\n";
            allocationValid = false;
            continue;
        }

        // Check virtual register exists in live intervals
        if (liveIntervals_.find(virtReg) == liveIntervals_.end()) {
            std::cerr
                << "ERROR: Virtual register not found in live intervals\n";
            allocationValid = false;
            continue;
        }

        // Group intervals by physical register
        physRegAssignments[physReg].push_back(liveIntervals_[virtReg].get());
    }

    // 2. Verify no overlapping intervals share the same physical register
    for (const auto& [physReg, intervals] : physRegAssignments) {
        // Check all intervals assigned to this physical register
        for (size_t i = 0; i < intervals.size(); ++i) {
            for (size_t j = i + 1; j < intervals.size(); ++j) {
                if (intervals[i]->interferesWith(*intervals[j])) {
                    std::cerr
                        << "ERROR: Interference detected between intervals "
                        << "assigned to physical register " << physReg << ":\n";
                    std::cerr << "  Interval 1: ";
                    // intervals[i]->print(std::cerr);
                    std::cerr << "\n  Interval 2: ";
                    // intervals[j]->print(std::cerr);
                    std::cerr << "\n";
                    allocationValid = false;
                }
            }
        }
    }

    // 3. Verify all definitions and uses are properly assigned
    for (const auto& [virtReg, interval] : liveIntervals_) {
        // Skip spilled registers
        if (spilledRegs_.count(virtReg)) {
            continue;
        }

        // Check physical register was assigned
        if (!virtualToPhysical_.count(virtReg)) {
            std::cerr << "ERROR: Virtual register not assigned to any physical "
                         "register\n";
            allocationValid = false;
            continue;
        }

        // Check all definitions and uses reference the assigned physical
        // register
        for (Instruction* inst : getUsesAndDefs(virtReg)) {
            bool found = false;
            for (size_t i = 0; i < inst->getOprandCount(); ++i) {
                MachineOperand* operand = inst->getOperand(i);
                if (operand->getType() == OperandType::Register) {
                    RegisterOperand* regOp =
                        static_cast<RegisterOperand*>(operand);
                    if (regOp == virtReg) {
                        if (regOp->getRegNum() != virtualToPhysical_[virtReg]) {
                            std::cerr << "ERROR: Instruction operand has wrong "
                                         "register assignment\n";
                            allocationValid = false;
                        }
                        found = true;
                        break;
                    }
                }
            }

            if (!found) {
                std::cerr << "ERROR: Virtual register not found in instruction "
                             "where it should be used\n";
                allocationValid = false;
            }
        }
    }

    // 4. Verify spill slots are properly assigned
    for (const auto& [virtReg, slot] : spillSlots_) {
        if (slot < 0 || slot >= nextSpillSlot_) {
            std::cerr << "ERROR: Invalid spill slot " << slot
                      << " for virtual register\n";
            allocationValid = false;
        }
    }

    // 5. Verify all physical registers are properly allocated
    for (const auto& physReg : physicalRegs_) {
        if (physReg.allocated && !physReg.currentInterval) {
            std::cerr << "ERROR: Physical register " << physReg.regNum
                      << " marked allocated but has no interval\n";
            allocationValid = false;
        }

        if (!physReg.allocated && physReg.currentInterval) {
            std::cerr << "ERROR: Physical register " << physReg.regNum
                      << " has interval but not marked allocated\n";
            allocationValid = false;
        }
    }

    if (!allocationValid) {
        std::cerr << "Register allocation verification failed!\n";
        assert(false && "Register allocation verification failed");
    } else {
        std::cout << "Register allocation verified successfully\n";
    }
}

void RegAllocGreedy::updateInstruction(Instruction* inst,
                                       RegisterOperand* oldReg,
                                       unsigned newPhysReg) {
    // 更新指令中的寄存器操作数
    for (size_t i = 0; i < inst->getOprandCount(); ++i) {
        MachineOperand* operand = inst->getOperand(i);
        if (operand->getType() == OperandType::Register) {
            RegisterOperand* regOp = static_cast<RegisterOperand*>(operand);
            if (regOp == oldReg) {
                // 创建新的物理寄存器操作数替换虚拟寄存器
                regOp->setPhysicalReg(newPhysReg);
            }
        }
    }
}

int RegAllocGreedy::getInstructionNumber(Instruction* inst) const {
    auto it = instructionMap_.find(inst);
    return it != instructionMap_.end() ? it->second : 0;
}

std::vector<Instruction*> RegAllocGreedy::getUsesAndDefs(
    RegisterOperand* virtReg) const {
    std::vector<Instruction*> result;

    // 添加定义
    auto defIt = defMap_.find(virtReg);
    if (defIt != defMap_.end()) {
        result.insert(result.end(), defIt->second.begin(), defIt->second.end());
    }

    // 添加使用
    auto useIt = useMap_.find(virtReg);
    if (useIt != useMap_.end()) {
        result.insert(result.end(), useIt->second.begin(), useIt->second.end());
    }

    return result;
}

bool RegAllocGreedy::isDefinition(Instruction* inst,
                                  RegisterOperand* virtReg) const {
    auto it = defMap_.find(virtReg);
    if (it != defMap_.end()) {
        return std::find(it->second.begin(), it->second.end(), inst) !=
               it->second.end();
    }
    return false;
}

bool RegAllocGreedy::isUse(Instruction* inst, RegisterOperand* virtReg) const {
    auto it = useMap_.find(virtReg);
    if (it != useMap_.end()) {
        return std::find(it->second.begin(), it->second.end(), inst) !=
               it->second.end();
    }
    return false;
}

// LiveInterval 方法实现
bool LiveInterval::interferesWith(const LiveInterval& other) const {
    for (const auto& range1 : ranges) {
        for (const auto& range2 : other.ranges) {
            if (!(range1.second < range2.first ||
                  range2.second < range1.first)) {
                return true;
            }
        }
    }
    return false;
}

bool LiveInterval::contains(int point) const {
    for (const auto& range : ranges) {
        if (point >= range.first && point <= range.second) {
            return true;
        }
    }
    return false;
}

// RegAllocUtils 实现
namespace RegAllocUtils {
std::vector<unsigned> getRegisterAllocationOrder() {
    // 基于RISC-V调用约定的寄存器分配顺序
    // 优先使用临时寄存器，然后是保存的寄存器
    return {
        5,  6,  7,  28, 29, 30, 31,      // t0-t2, t3-t6 (临时寄存器)
        10, 11, 12, 13, 14, 15, 16, 17,  // a0-a7 (参数寄存器)
        8,  9,  18, 19, 20, 21, 22, 23, 24, 25, 26, 27  // s0-s11 (保存寄存器)
    };
}

bool isCallerSaved(unsigned physReg) {
    // t0-t6, a0-a7 是调用者保存
    return (physReg >= 5 && physReg <= 7) || (physReg >= 10 && physReg <= 17) ||
           (physReg >= 28 && physReg <= 31);
}

bool isCalleeSaved(unsigned physReg) {
    // s0-s11 是被调用者保存
    return (physReg >= 8 && physReg <= 9) || (physReg >= 18 && physReg <= 27);
}

// 所有寄存器都是整数寄存器
// TODO: float
RegClass getRegisterClass(unsigned physReg) {
    (void)physReg;  // Suppress unused parameter warning
    return RegClass::Integer;
}

bool interferes(const LiveInterval& a, const LiveInterval& b) {
    return a.interferesWith(b);
}
}  // namespace RegAllocUtils

}  // namespace riscv64