#include "RAGreedy/RegAllocGreedy.h"

#include <algorithm>
#include <climits>
#include <iomanip>

// TODO: scheisse
namespace riscv64 {

void RegAllocGreedy::run(void) {
    init();
    seedLiveRegs();
    allocateRegisters();
}

void RegAllocGreedy::init() {
    // 初始化各个组件
    VRM = new VirtRegMap(*function);
    Matrix = new LiveRegMatrix();

    // 初始化Matrix
    Matrix->init(*function, *LIS, *VRM);

    // 清空工作队列和数据结构
    while (!Queue.empty()) Queue.pop();
    unassignedRegs.clear();
}

void RegAllocGreedy::seedLiveRegs() {
    // 收集所有虚拟寄存器的活跃区间并加入队列
    auto LiveIntervals = LIS->getAllLiveIntervals();
    for (auto LI : LiveIntervals) {
        enqueue(LI);
    }
}

void RegAllocGreedy::allocateRegisters() {
    // 主要的寄存器分配循环
    while (!Queue.empty()) {
        LiveInterval *LI = dequeue();
        if (!LI) continue;

        // 获取分配顺序
        std::vector<unsigned> AllocationOrder = getAllocationOrder(*LI);
        std::vector<unsigned> NewVRegs;

        selectOrSplit(LI, AllocationOrder);
    }
}

void RegAllocGreedy::selectOrSplit(LiveInterval *LI,
                                   std::vector<unsigned> allocationOrder) {
    std::vector<unsigned> NewVRegs;
    std::set<unsigned> FixedRegs;

    // 尝试直接分配
    unsigned PhysReg = tryAssign(*LI, allocationOrder, NewVRegs, FixedRegs);

    if (PhysReg != NO_PHYS_REG) {
        std::cout << "Assigning " << LI->reg().getRegNum() << " to " << PhysReg
                  << std::endl;
        Matrix->assign(*LI, RegisterOperand(PhysReg));
        VRM->assignVirt2Phys(LI->reg().getRegNum(), PhysReg);
        return;
    }

    // TODO: not right
    // 尝试驱逐其他虚拟寄存器
    // PhysReg = tryEvict(*LI, allocationOrder, NewVRegs, UINT_MAX, FixedRegs);
    // if (PhysReg != NO_PHYS_REG) {
    //     Matrix->assign(*LI, RegisterOperand(PhysReg));
    //     VRM->assignVirt2Phys(LI->reg().getRegNum(), PhysReg);
    //     return;
    // }

    // 尝试区域分割
    PhysReg = tryRegionSplit(*LI, allocationOrder, NewVRegs);
    if (PhysReg != NO_PHYS_REG) {
        for (unsigned NewVReg : NewVRegs) {
            if (LIS->hasInterval(RegisterOperand(NewVReg))) {
                enqueue(&LIS->getInterval(RegisterOperand(NewVReg)));
            }
        }
        return;
    }

    // 尝试基本块分割
    PhysReg = tryBlockSplit(*LI, allocationOrder, NewVRegs);
    if (PhysReg != NO_PHYS_REG) {
        for (unsigned NewVReg : NewVRegs) {
            if (LIS->hasInterval(RegisterOperand(NewVReg))) {
                enqueue(&LIS->getInterval(RegisterOperand(NewVReg)));
            }
        }
        return;
    }

    // 尝试局部分割
    PhysReg = tryLocalSplit(*LI, allocationOrder, NewVRegs);
    if (PhysReg != NO_PHYS_REG) {
        for (unsigned NewVReg : NewVRegs) {
            if (LIS->hasInterval(RegisterOperand(NewVReg))) {
                enqueue(&LIS->getInterval(RegisterOperand(NewVReg)));
            }
        }
        return;
    }

    // 最后尝试重着色
    std::unordered_set<unsigned> FixedRegisters;
    std::vector<std::pair<const LiveInterval *, unsigned>> RecolorStack;
    PhysReg = tryLastChanceRecoloring(*LI, allocationOrder, NewVRegs,
                                      FixedRegisters, RecolorStack, 0);
    if (PhysReg != NO_PHYS_REG) {
        Matrix->assign(*LI, RegisterOperand(PhysReg));
        VRM->assignVirt2Phys(LI->reg().getRegNum(), PhysReg);
        return;
    }

    // 如果所有方法都失败，则溢出到内存
    VRM->assignVirt2StackSlot(LI->reg().getRegNum(), stackSlotNum_);
    stackSlotNum_ ++;
}

void RegAllocGreedy::enqueue(LiveInterval *LI) {
    if (!LI || LI->empty()) return;

    // 计算优先级（基于权重和大小）
    unsigned Priority =
        static_cast<unsigned>(LI->weight() * 1000) + LI->getSize();

    Queue.push(std::make_pair(Priority, LI));
}

LiveInterval *RegAllocGreedy::dequeue() {
    if (Queue.empty()) return nullptr;

    auto Top = Queue.top();
    Queue.pop();
    return Top.second;
}

// TODO：more constraint
std::vector<unsigned> RegAllocGreedy::getAllocationOrder(
    const LiveInterval &VirtReg) {
    std::vector<unsigned> preferredRegs;

    // 从LiveInterval获取虚拟寄存器号
    unsigned virtualReg = VirtReg.reg().getRegNum();

    // 如果虚拟寄存器有特定的ABI约束，优先考虑相应的物理寄存器
    // if (physicalConstraints.find(virtualReg) != physicalConstraints.end()) {
    //     const auto& constraints = physicalConstraints.at(virtualReg);
    //     for (unsigned physReg : constraints) {
    //         if (!ABI::isReservedReg(physReg, assigningFloat)) {
    //             preferredRegs.push_back(physReg);
    //         }
    //     }
    // }

    // 根据使用模式推荐寄存器
    if (isUsedAsArgument(virtualReg)) {
        // 优先使用参数寄存器
        if (assigningFloat) {
            for (unsigned reg = 42; reg <= 49; ++reg) {  // fa0-fa7 -> 42-49
                if (std::find(preferredRegs.begin(), preferredRegs.end(),
                              reg) == preferredRegs.end()) {
                    preferredRegs.push_back(reg);
                }
            }
        } else {
            for (unsigned reg = 10; reg <= 17; ++reg) {  // a0-a7
                if (std::find(preferredRegs.begin(), preferredRegs.end(),
                              reg) == preferredRegs.end()) {
                    preferredRegs.push_back(reg);
                }
            }
        }
    }

    if (isUsedAcrossCalls(virtualReg)) {
        // 优先使用被调用者保存寄存器
        if (assigningFloat) {
            // fs0-fs1 -> 40-41, fs2-fs11 -> 50-59
            for (unsigned reg = 40; reg <= 41; ++reg) {
                if (std::find(preferredRegs.begin(), preferredRegs.end(),
                              reg) == preferredRegs.end()) {
                    preferredRegs.push_back(reg);
                }
            }
            for (unsigned reg = 50; reg <= 59; ++reg) {  // fs2-fs11
                if (std::find(preferredRegs.begin(), preferredRegs.end(),
                              reg) == preferredRegs.end()) {
                    preferredRegs.push_back(reg);
                }
            }
        } else {
            // s1
            if (std::find(preferredRegs.begin(), preferredRegs.end(), 9) ==
                preferredRegs.end()) {
                preferredRegs.push_back(9);
            }
            for (unsigned reg = 18; reg <= 27; ++reg) {  // s2-s11
                if (std::find(preferredRegs.begin(), preferredRegs.end(),
                              reg) == preferredRegs.end()) {
                    preferredRegs.push_back(reg);
                }
            }
        }
    } else {
        // 优先使用调用者保存寄存器
        if (assigningFloat) {
            // ft0-ft7 -> 32-39, ft8-ft11 -> 60-63
            // TODO：糟糕的预留
            // for (unsigned reg = 32; reg <= 34; ++reg) {  // ft0-ft2
            //     if (std::find(preferredRegs.begin(), preferredRegs.end(),
            //                   reg) == preferredRegs.end()) {
            //         preferredRegs.push_back(reg);
            //     }
            // }
            for (unsigned reg = 35; reg <= 39; ++reg) {  // ft3-ft7
                if (std::find(preferredRegs.begin(), preferredRegs.end(),
                              reg) == preferredRegs.end()) {
                    preferredRegs.push_back(reg);
                }
            }
            // TODO: 确保正确
            // for (unsigned reg = 42; reg <= 49; ++reg) {  // fa0-fa7 -> 42-49
            //     if (std::find(preferredRegs.begin(), preferredRegs.end(),
            //                   reg) == preferredRegs.end()) {
            //         preferredRegs.push_back(reg);
            //     }
            // }
            for (unsigned reg = 60; reg <= 63; ++reg) {  // ft8-ft11
                if (std::find(preferredRegs.begin(), preferredRegs.end(),
                              reg) == preferredRegs.end()) {
                    preferredRegs.push_back(reg);
                }
            }
            for (unsigned reg = 50; reg <= 59; ++reg) {  // fs2-fs11
                if (std::find(preferredRegs.begin(), preferredRegs.end(),
                              reg) == preferredRegs.end()) {
                    preferredRegs.push_back(reg);
                }
            }
        } else {
            // TODO: 悲剧性的预留
            // for (unsigned reg = 5; reg <= 7; ++reg) {  // t0-t2
            //     if (std::find(preferredRegs.begin(), preferredRegs.end(),
            //                   reg) == preferredRegs.end()) {
            //         preferredRegs.push_back(reg);
            //     }
            // }
            // TODO：确保正确
            // for (unsigned reg = 10; reg <= 17; ++reg) {  // a0-a7
            //     if (std::find(preferredRegs.begin(), preferredRegs.end(),
            //                   reg) == preferredRegs.end()) {
            //         preferredRegs.push_back(reg);
            //     }
            // }
            for (unsigned reg = 28; reg <= 31; ++reg) {  // t3-t6
                if (std::find(preferredRegs.begin(), preferredRegs.end(),
                              reg) == preferredRegs.end()) {
                    preferredRegs.push_back(reg);
                }
            }
            if (std::find(preferredRegs.begin(), preferredRegs.end(), 9) ==
                preferredRegs.end()) {
                preferredRegs.push_back(9);
            }
            for (unsigned reg = 18; reg <= 27; ++reg) {  // s2-s11
                if (std::find(preferredRegs.begin(), preferredRegs.end(),
                              reg) == preferredRegs.end()) {
                    preferredRegs.push_back(reg);
                }
            }
        }
    }

    return preferredRegs;
}

bool RegAllocGreedy::isUsedAsArgument(unsigned virtualReg) const {
    // 检查虚拟寄存器是否在函数调用中用作参数
    for (auto &bb : *function) {
        for (const auto &inst : *bb) {
            if (inst->isCallInstr()) {
                auto usedRegs = inst->getUsedIntegerRegs();
                if (std::find(usedRegs.begin(), usedRegs.end(), virtualReg) !=
                    usedRegs.end()) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool RegAllocGreedy::isUsedAcrossCalls(unsigned virtualReg) const {
    // 获取该虚拟寄存器的活跃区间
    RegisterOperand regOp(virtualReg);
    if (!LIS->hasInterval(regOp)) {
        return false;
    }

    const LiveInterval &LI = LIS->getInterval(regOp);

    // 检查活跃区间是否跨越任何函数调用
    for (auto &bb : *function) {
        for (const auto &inst : *bb) {
            if (inst->isCallInstr()) {
                SlotIndex callSlot = LIS->getInstructionIndex(*inst);

                // 检查调用指令是否在活跃区间内
                // 如果是，说明这个虚拟寄存器需要在调用前后保持值
                if (LI.liveAt(callSlot)) {
                    // 进一步检查：寄存器是否在调用前定义，在调用后使用
                    bool definedBefore = false;
                    bool usedAfter = false;

                    // 检查调用前是否有定义
                    SlotIndex beforeCall = callSlot.getPrevSlot();
                    if (LI.liveAt(beforeCall)) {
                        definedBefore = true;
                    }

                    // 检查调用后是否有使用
                    SlotIndex afterCall = callSlot.getNextSlot();
                    if (LI.liveAt(afterCall)) {
                        usedAfter = true;
                    }

                    if (definedBefore && usedAfter) {
                        return true;
                    }
                }
            }
        }
    }

    return false;
}

bool RegAllocGreedy::checkInterference(const LiveInterval &VirtReg,
                                       unsigned PhysReg) {
    RegisterOperand PhysRegOp(PhysReg);
    return Matrix->checkInterference(VirtReg, PhysRegOp) !=
           LiveRegMatrix::IK_Free;
}

unsigned RegAllocGreedy::calculateSpillCost(const LiveInterval &VirtReg) {
    // TODO: make it
    return static_cast<unsigned>(VirtReg.weight() * 100);
}

bool RegAllocGreedy::canSplit(const LiveInterval &VirtReg) {
    // 检查是否可以分割（至少有2个段）
    return VirtReg.segments().size() > 1;
}

std::vector<LiveInterval *> RegAllocGreedy::splitInterval(
    const LiveInterval &VirtReg) {
    std::vector<LiveInterval *> SplitIntervals;

    // 简化的分割实现：在每个段的中点分割
    // TODO: 这不对
    // const auto &Segments = VirtReg.segments();
    // for (const auto &Segment : Segments) {
    //     SlotIndex MidPoint = SlotIndex((Segment.start.getIndex() +
    //     Segment.end.getIndex()) / 2);

    //     // 创建新的活跃区间（实际实现会更复杂）
    //     RegisterOperand NewReg(/* 新的虚拟寄存器号 */);
    //     LiveInterval *NewLI = &LIS->createEmptyInterval(NewReg);
    //     SplitIntervals.push_back(NewLI);
    // }

    return SplitIntervals;
}

std::vector<LiveInterval *> RegAllocGreedy::getEvictionCandidates(
    unsigned PhysReg) {
    std::vector<LiveInterval *> Candidates;

    RegisterOperand PhysRegOp(PhysReg);
    RegisterOperand VirtReg = Matrix->getOneVReg(PhysReg);

    if (VirtReg.getRegNum() != NO_PHYS_REG && LIS->hasInterval(VirtReg)) {
        Candidates.push_back(&LIS->getInterval(VirtReg));
    }

    return Candidates;
}

bool RegAllocGreedy::shouldEvict(const LiveInterval &Evictee,
                                 const LiveInterval &Evictor) {
    // 简化的驱逐决策：比较权重
    // TODO: make it better
    return Evictor.weight() > Evictee.weight() * 1.1;  // 需要显著更高的权重
}

unsigned RegAllocGreedy::tryAssign(const LiveInterval &VirtReg,
                                   std::vector<unsigned> allocationOrder,
                                   std::vector<unsigned> &newVRegs,
                                   std::set<unsigned> fixedRegs) {
    for (unsigned PhysReg : allocationOrder) {
        if (fixedRegs.count(PhysReg)) continue;

        if (!checkInterference(VirtReg, PhysReg)) {
            return PhysReg;
        }
    }
    return NO_PHYS_REG;
}

unsigned RegAllocGreedy::tryEvict(const LiveInterval &VirtReg,
                                  std::vector<unsigned> allocationOrder,
                                  std::vector<unsigned> &newVRegs,
                                  unsigned costPerUseLimit,
                                  std::set<unsigned> fixedRegs) {
    for (unsigned PhysReg : allocationOrder) {
        if (fixedRegs.count(PhysReg)) continue;

        auto Candidates = getEvictionCandidates(PhysReg);
        bool CanEvictAll = true;

        for (auto *Candidate : Candidates) {
            if (!shouldEvict(*Candidate, VirtReg)) {
                CanEvictAll = false;
                break;
            }
        }

        if (CanEvictAll) {
            // 执行驱逐
            for (auto *Candidate : Candidates) {
                Matrix->unassign(*Candidate);
                enqueue(Candidate);
            }
            return PhysReg;
        }
    }
    return NO_PHYS_REG;
}

unsigned RegAllocGreedy::tryRegionSplit(const LiveInterval &VirtReg,
                                        std::vector<unsigned> allocationOrder,
                                        std::vector<unsigned> &newVRegs) {
    if (!canSplit(VirtReg)) return NO_PHYS_REG;

    // TODO: this is wrong
    // 简化的区域分割实现
    auto SplitIntervals = splitInterval(VirtReg);
    for (auto *SplitLI : SplitIntervals) {
        newVRegs.push_back(SplitLI->reg().getRegNum());
    }

    return newVRegs.empty() ? NO_PHYS_REG : 1;  // 返回非零值表示成功
}

unsigned RegAllocGreedy::tryBlockSplit(const LiveInterval &VirtReg,
                                       std::vector<unsigned> allocationOrder,
                                       std::vector<unsigned> &newVRegs) {
    if (!canSplit(VirtReg)) return NO_PHYS_REG;

    // TODO: this is wrong
    // 基本块级别的分割（简化实现）
    return tryRegionSplit(VirtReg, allocationOrder, newVRegs);
}

unsigned RegAllocGreedy::tryLocalSplit(const LiveInterval &VirtReg,
                                       std::vector<unsigned> allocationOrder,
                                       std::vector<unsigned> &newVRegs) {
    if (!canSplit(VirtReg)) return NO_PHYS_REG;

    // TODO: this is wrong
    // 局部分割（简化实现）
    return tryRegionSplit(VirtReg, allocationOrder, newVRegs);
}

unsigned RegAllocGreedy::tryLastChanceRecoloring(
    const LiveInterval &VirtReg, std::vector<unsigned> allocationOrder,
    std::vector<unsigned> &NewVRegs,
    std::unordered_set<unsigned> &FixedRegisters,
    std::vector<std::pair<const LiveInterval *, unsigned>> &RecolorStack,
    unsigned Depth) {
    // 防止递归过深
    if (Depth > 5) return NO_PHYS_REG;

    for (unsigned PhysReg : allocationOrder) {
        if (FixedRegisters.count(PhysReg)) continue;

        auto Candidates = getEvictionCandidates(PhysReg);
        if (Candidates.empty()) {
            return PhysReg;
        }

        // 尝试为被驱逐的寄存器重着色
        bool Success = true;
        for (auto *Candidate : Candidates) {
            std::vector<unsigned> CandidateOrder =
                getAllocationOrder(*Candidate);
            std::vector<unsigned> CandidateNewVRegs;

            unsigned NewPhysReg = tryLastChanceRecoloring(
                *Candidate, CandidateOrder, CandidateNewVRegs, FixedRegisters,
                RecolorStack, Depth + 1);

            if (NewPhysReg == NO_PHYS_REG) {
                Success = false;
                break;
            }

            RecolorStack.push_back(std::make_pair(Candidate, NewPhysReg));
        }

        if (Success) {
            // 应用重着色
            for (auto &Entry : RecolorStack) {
                Matrix->unassign(*Entry.first);
                Matrix->assign(*Entry.first, RegisterOperand(Entry.second));
                VRM->assignVirt2Phys(Entry.first->reg().getRegNum(),
                                     Entry.second);
            }
            return PhysReg;
        }

        // 回滚
        RecolorStack.clear();
    }

    return NO_PHYS_REG;
}

void RegAllocGreedy::print(std::ostream &OS) const {
    OS << "RegAllocGreedy for Function: " << function->getName() << "\n";
    OS << "====================================================\n";

    // 打印分配队列状态
    OS << "Allocation Queue Status:\n";
    OS << "  Queue size: " << Queue.size() << "\n";
    OS << "  Unassigned registers: " << unassignedRegs.size() << "\n";
    OS << "\n";

    // 打印虚拟寄存器到物理寄存器的映射
    OS << "Virtual to Physical Register Mapping:\n";
    bool hasMapping = false;

    // 遍历函数中的所有虚拟寄存器
    for (auto &BB : *function) {
        for (auto &Inst : *BB) {
            for (auto &Operand : Inst->getOperands()) {
                if (Operand->isReg()) {
                    RegisterOperand *RegOp =
                        static_cast<RegisterOperand *>(Operand.get());
                    unsigned VirtReg = RegOp->getRegNum();

                    if (VRM && VRM->hasPhys(VirtReg)) {
                        unsigned PhysReg = VRM->getPhys(VirtReg);
                        OS << "  v" << VirtReg << " -> "
                           << ABI::getABINameFromRegNum(PhysReg) << "\n";
                        hasMapping = true;
                    }
                }
            }
        }
    }

    if (!hasMapping) {
        OS << "  (no mappings)\n";
    }
    OS << "\n";

    // 打印溢出的虚拟寄存器
    OS << "Spilled Virtual Registers:\n";
    bool hasSpills = false;

    for (auto &BB : *function) {
        for (auto &Inst : *BB) {
            for (auto &Operand : Inst->getOperands()) {
                if (Operand->isReg()) {
                    RegisterOperand *RegOp =
                        static_cast<RegisterOperand *>(Operand.get());
                    unsigned VirtReg = RegOp->getRegNum();

                    if (VRM && VRM->hasStackSlot(VirtReg)) {
                        StackSlot Slot = VRM->getStackSlot(VirtReg);
                        OS << "  v" << VirtReg << " -> stack slot " << Slot
                           << "\n";
                        hasSpills = true;
                    }
                }
            }
        }
    }

    if (!hasSpills) {
        OS << "  (no spills)\n";
    }
    OS << "\n";

    // 打印活跃区间信息
    OS << "Live Intervals Summary:\n";
    if (LIS) {
        unsigned totalIntervals = 0;
        float totalWeight = 0.0f;

        for (auto &BB : *function) {
            for (auto &Inst : *BB) {
                for (auto &Operand : Inst->getOperands()) {
                    if (Operand->isReg()) {
                        RegisterOperand *RegOp =
                            static_cast<RegisterOperand *>(Operand.get());

                        if (LIS->hasInterval(*RegOp)) {
                            const LiveInterval &LI = LIS->getInterval(*RegOp);
                            totalIntervals++;
                            totalWeight += LI.weight();
                        }
                    }
                }
            }
        }

        OS << "  Total live intervals: " << totalIntervals << "\n";
        OS << "  Average weight: "
           << (totalIntervals > 0 ? totalWeight / totalIntervals : 0.0f)
           << "\n";
    } else {
        OS << "  (LiveIntervals not available)\n";
    }
    OS << "\n";

    // 打印详细的活跃区间信息（可选）
    OS << "Detailed Live Intervals:\n";
    if (LIS) {
        std::vector<std::pair<unsigned, const LiveInterval *>> intervals;

        for (auto &BB : *function) {
            for (auto &Inst : *BB) {
                for (auto &Operand : Inst->getOperands()) {
                    if (Operand->isReg()) {
                        RegisterOperand *RegOp =
                            static_cast<RegisterOperand *>(Operand.get());
                        if (LIS->hasInterval(*RegOp)) {
                            const LiveInterval &LI = LIS->getInterval(*RegOp);
                            intervals.push_back(
                                std::make_pair(RegOp->getRegNum(), &LI));
                        }
                    }
                }
            }
        }

        // 按虚拟寄存器号排序
        std::sort(intervals.begin(), intervals.end());

        // 移除重复项
        intervals.erase(std::unique(intervals.begin(), intervals.end()),
                        intervals.end());

        if (intervals.empty()) {
            OS << "  (no intervals)\n";
        } else {
            for (const auto &pair : intervals) {
                unsigned VirtReg = pair.first;
                const LiveInterval *LI = pair.second;

                OS << "  v" << VirtReg << ": weight=" << std::fixed
                   << std::setprecision(2) << LI->weight()
                   << ", segments=" << LI->segments().size();

                if (VRM) {
                    if (VRM->hasPhys(VirtReg)) {
                        OS << ", assigned to "
                           << ABI::getABINameFromRegNum(VRM->getPhys(VirtReg));
                    } else if (VRM->hasStackSlot(VirtReg)) {
                        OS << ", spilled to slot "
                           << VRM->getStackSlot(VirtReg);
                    } else {
                        OS << ", unassigned";
                    }
                }
                OS << "\n";

                // 打印前几个段的详细信息
                const auto &segments = LI->segments();
                size_t maxSegments = std::min(segments.size(), size_t(3));
                for (size_t i = 0; i < maxSegments; ++i) {
                    const auto &seg = segments[i];
                    OS << "    [" << seg.start << ", " << seg.end << ")\n";
                }
                if (segments.size() > 3) {
                    OS << "    ... and " << (segments.size() - 3)
                       << " more segments\n";
                }
            }
        }
    }

    OS << "\n====================================================\n";
}
}  // namespace riscv64
