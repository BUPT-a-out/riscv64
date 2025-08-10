#include "RAGreedy/LiveIntervals.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <set>

#include "Instructions/Function.h"

namespace riscv64 {

// VNInfo的dump方法实现
void VNInfo::dump() const {
    std::cout << "VN" << id << ":";
    if (isUnused()) {
        std::cout << "unused";
    } else if (isPHIDef()) {
        std::cout << "PHI@" << def;
    } else {
        std::cout << "@" << def;
    }
}

// LiveInterval::Segment的dump方法实现
void LiveInterval::Segment::dump() const {
    std::cout << "[" << start << "," << end << ":";
    if (valno) {
        std::cout << "VN" << valno->id;
    } else {
        std::cout << "null";
    }
    std::cout << ")";
}

// LiveInterval::SubRange的方法实现
bool LiveInterval::SubRange::liveAt(SlotIndex Pos) const {
    auto it = std::upper_bound(
        segments.begin(), segments.end(), Pos,
        [](SlotIndex pos, const Segment &seg) { return pos < seg.end; });
    return it != segments.end() && it->start <= Pos;
}

void LiveInterval::SubRange::print(std::ostream &OS) const {
    OS << "SubRange: ";
    for (const auto &seg : segments) {
        OS << "[" << seg.start << "," << seg.end << ")";
    }
}

void LiveInterval::SubRange::dump() const {
    print(std::cout);
    std::cout << std::endl;
}

// LiveInterval的主要方法实现
unsigned LiveInterval::getSize() const {
    unsigned Size = 0;
    for (const auto &segment : segments_) {
        Size += segment.end.getIndex() - segment.start.getIndex();
    }
    return Size;
}

LiveInterval::Segments::iterator LiveInterval::find(SlotIndex Pos) {
    return std::upper_bound(
        segments_.begin(), segments_.end(), Pos,
        [](SlotIndex pos, const Segment &seg) { return pos < seg.end; });
}

LiveInterval::Segments::const_iterator LiveInterval::find(SlotIndex Pos) const {
    return std::upper_bound(
        segments_.begin(), segments_.end(), Pos,
        [](SlotIndex pos, const Segment &seg) { return pos < seg.end; });
}

bool LiveInterval::overlaps(const LiveInterval &Other) const {
    auto I1 = segments_.begin(), E1 = segments_.end();
    auto I2 = Other.segments_.begin(), E2 = Other.segments_.end();

    while (I1 != E1 && I2 != E2) {
        if (I1->start < I2->end && I2->start < I1->end) {
            return true;
        }
        if (I1->end <= I2->start) {
            ++I1;
        } else {
            ++I2;
        }
    }
    return false;
}

bool LiveInterval::overlaps(SlotIndex Start, SlotIndex End) const {
    assert(Start < End && "Invalid range");
    auto I = find(Start);
    return I != segments_.end() && I->start < End;
}

VNInfo *LiveInterval::getVNInfoAt(SlotIndex Pos) const {
    auto I = find(Pos);
    return (I != segments_.end() && I->start <= Pos) ? I->valno : nullptr;
}

VNInfo *LiveInterval::getVNInfoBefore(SlotIndex Pos) const {
    auto I = find(Pos);
    if (I != segments_.begin()) {
        --I;
        if (I->end <= Pos) {
            return I->valno;
        }
    }
    return nullptr;
}

void LiveInterval::addSegment(Segment S) {
    auto I = findInsertPos(S);
    addSegment(I, S);
}

LiveInterval::Segments::iterator LiveInterval::addSegment(Segments::iterator I,
                                                          Segment S) {
    assert(S.start < S.end && "Cannot add empty segment");

    // 找到插入位置并合并重叠的段
    I = segments_.insert(I, S);
    mergeSegments();
    return I;
}

void LiveInterval::removeSegment(SlotIndex Start, SlotIndex End,
                                 bool RemoveDeadValNo) {
    removeSegment(Segment(Start, End, nullptr), RemoveDeadValNo);
}

void LiveInterval::removeSegment(Segment S, bool RemoveDeadValNo) {
    auto I = find(S.start);
    while (I != segments_.end() && I->start < S.end) {
        if (I->end <= S.start) {
            ++I;
            continue;
        }

        if (I->start >= S.start && I->end <= S.end) {
            // 完全包含，删除整个段
            I = segments_.erase(I);
        } else if (I->start < S.start && I->end > S.end) {
            // 分割段
            Segment NewSeg(S.end, I->end, I->valno);
            I->end = S.start;
            ++I;
            I = segments_.insert(I, NewSeg);
            ++I;
        } else if (I->start < S.start) {
            // 截断右边
            I->end = S.start;
            ++I;
        } else {
            // 截断左边
            I->start = S.end;
            ++I;
        }
    }

    if (RemoveDeadValNo) {
        // 移除未使用的值编号
        RenumberValues();
    }
}

void LiveInterval::clear() {
    segments_.clear();
    valnos_.clear();
    clearSubRanges();
}

VNInfo *LiveInterval::createDeadDef(SlotIndex Def) {
    auto VNI = std::make_unique<VNInfo>(valnos_.size(), Def);
    VNI->markUnused();
    VNInfo *ptr = VNI.get();
    valnos_.push_back(std::move(VNI));
    return ptr;
}

void LiveInterval::RenumberValues() {
    // 收集所有使用的值编号
    std::set<VNInfo *> UsedVNs;
    for (const auto &seg : segments_) {
        if (seg.valno) {
            UsedVNs.insert(seg.valno);
        }
    }

    // 重新编号
    unsigned NewID = 0;
    for (auto &vn : valnos_) {
        if (UsedVNs.count(vn.get())) {
            vn->id = NewID++;
        }
    }
}

bool LiveInterval::verify() const {
    if (segments_.empty()) {
        return valnos_.empty();
    }

    // 检查段是否按顺序排列且不重叠
    for (size_t i = 1; i < segments_.size(); ++i) {
        if (segments_[i - 1].end > segments_[i].start) {
            return false;
        }
    }

    // 检查所有段的值编号是否有效
    for (const auto &seg : segments_) {
        if (!seg.valno || seg.valno->id >= valnos_.size()) {
            return false;
        }
    }

    return true;
}

LiveInterval::SubRange *LiveInterval::createSubRange() {
    auto SR = new SubRange();
    SR->Next = SubRanges_;
    SubRanges_ = SR;
    return SR;
}

void LiveInterval::removeSubRange() {
    if (SubRanges_) {
        auto ToDelete = SubRanges_;
        SubRanges_ = SubRanges_->Next;
        delete ToDelete;
    }
}

void LiveInterval::removeEmptySubRanges() {
    SubRange **Current = &SubRanges_;
    while (*Current) {
        if ((*Current)->empty()) {
            SubRange *ToDelete = *Current;
            *Current = (*Current)->Next;
            delete ToDelete;
        } else {
            Current = &((*Current)->Next);
        }
    }
}

LiveInterval::SubRange *LiveInterval::getSubRange() { return SubRanges_; }

const LiveInterval::SubRange *LiveInterval::getSubRange() const {
    return SubRanges_;
}

bool LiveInterval::operator<(const LiveInterval &Other) const {
    return reg_.getRegNum() < Other.reg_.getRegNum();
}

bool LiveInterval::operator==(const LiveInterval &Other) const {
    return reg_.getRegNum() == Other.reg_.getRegNum() &&
           segments_ == Other.segments_;
}

void LiveInterval::print(std::ostream &OS) const {
    OS << "LiveInterval for " << reg_.getRegNum() << " weight=" << weight_
       << ": ";
    for (const auto &seg : segments_) {
        OS << "[" << seg.start << "," << seg.end << ":VN" << seg.valno->id
        << (seg.valno->isPHIDef() ? "phi" : "")
           << ") ";
    }
}

void LiveInterval::dump() const {
    print(std::cout);
    std::cout << std::endl;
}

// 私有辅助方法实现
void LiveInterval::mergeSegments() {
    if (segments_.size() <= 1) return;

    std::sort(segments_.begin(), segments_.end());

    auto WriteIt = segments_.begin();
    for (auto ReadIt = segments_.begin() + 1; ReadIt != segments_.end();
         ++ReadIt) {
        if (WriteIt->end >= ReadIt->start && WriteIt->valno == ReadIt->valno) {
            // 合并段
            WriteIt->end = std::max(WriteIt->end, ReadIt->end);
        } else {
            ++WriteIt;
            if (WriteIt != ReadIt) {
                *WriteIt = *ReadIt;
            }
        }
    }

    segments_.erase(WriteIt + 1, segments_.end());
}

void LiveInterval::normalizeSegments() { mergeSegments(); }

LiveInterval::Segments::iterator LiveInterval::findInsertPos(Segment S) {
    return std::lower_bound(segments_.begin(), segments_.end(), S);
}

void LiveInterval::clearSubRanges() {
    while (SubRanges_) {
        SubRange *ToDelete = SubRanges_;
        SubRanges_ = SubRanges_->Next;
        delete ToDelete;
    }
}

void LiveInterval::join(LiveInterval &Other) {
    // 简单实现：将另一个区间的所有段添加到当前区间
    for (const auto &seg : Other.segments_) {
        addSegment(seg);
    }

    // 合并值编号
    for (auto &vn : Other.valnos_) {
        valnos_.push_back(std::move(vn));
    }

    RenumberValues();
    Other.clear();
}

/// LiveIntervals

void LiveIntervals::analyze(Function &fn) {
    if (!Indexes) {
        throw std::runtime_error("LIS should have non-empty SI");
    }

    // 逆转后序遍历得到逆后序
    auto postOrderBlocks = fn.getPostOrder();
    std::reverse(postOrderBlocks.begin(), postOrderBlocks.end());

    // 3. 为每个虚拟寄存器创建活跃区间
    std::set<unsigned> AllRegs;
    for (BasicBlock *BB : postOrderBlocks) {
        for (auto &I : *BB) {
            // TODO: float
            auto usedRegs = I->getUsedIntegerRegs();
            auto definedRegs = I->getDefinedIntegerRegs();

            AllRegs.insert(usedRegs.begin(), usedRegs.end());
            AllRegs.insert(definedRegs.begin(), definedRegs.end());
        }
    }

    // 4. 计算每个寄存器的活跃区间
    for (unsigned regNum : AllRegs) {
        if (regNum >= 64) {
            RegisterOperand reg(regNum);

            LiveInterval &LI = createAndComputeVirtRegInterval(reg);

            VirtRegIntervals[reg] = &LI;
        }
    }
}

void LiveIntervals::clear() {
    // 清理所有活跃区间
    for (auto &pair : VirtRegIntervals) {
        delete pair.second;
    }
    VirtRegIntervals.clear();
}

LiveInterval &LiveIntervals::getInterval(RegisterOperand Reg) {
    auto it = VirtRegIntervals.find(Reg);
    if (it != VirtRegIntervals.end()) {
        return *it->second;
    }

    // 如果不存在，创建并计算
    return createAndComputeVirtRegInterval(Reg);
}

const LiveInterval &LiveIntervals::getInterval(RegisterOperand Reg) const {
    auto it = VirtRegIntervals.find(Reg);
    if (it != VirtRegIntervals.end()) {
        return *it->second;
    }

    // 抛出异常或返回默认值
    throw std::runtime_error("LiveInterval not found for register");
}

bool LiveIntervals::hasInterval(RegisterOperand Reg) const {
    return VirtRegIntervals.find(Reg) != VirtRegIntervals.end();
}

LiveInterval &LiveIntervals::createEmptyInterval(RegisterOperand Reg) {
    // 使用默认权重创建空区间
    auto *LI = new LiveInterval(Reg, 0.0f);
    VirtRegIntervals[Reg] = LI;
    return *LI;
}

// TODO: float
// LiveInterval &LiveIntervals::createAndComputeVirtRegInterval(RegisterOperand Reg) {
//     LiveInterval &LI = createEmptyInterval(Reg);
    
//     // 初始化每个基本块的live-in、live-out、def、use信息
//     std::map<BasicBlock*, bool> LiveIn, LiveOut, Def, Use;
    
//     // 1. 计算每个基本块内的def和use
//     for (auto &BB : *function) {
//         bool hasDef = false;
//         bool hasUse = false;
        
//         // 遍历基本块内的指令，注意use要在def之前才算
//         for (auto &I : *BB) {
//             // 检查使用（只有在定义之前的使用才算本基本块的use）
//             if (!hasDef) {
//                 auto usedRegs = I->getUsedIntegerRegs();
//                 if (std::find(usedRegs.begin(), usedRegs.end(), Reg.getRegNum()) != usedRegs.end()) {
//                     hasUse = true;
//                 }
//             }
            
//             // 检查定义
//             auto definedRegs = I->getDefinedIntegerRegs();
//             if (std::find(definedRegs.begin(), definedRegs.end(), Reg.getRegNum()) != definedRegs.end()) {
//                 hasDef = true;
//             }
//         }
        
//         Def[BB.get()] = hasDef;
//         Use[BB.get()] = hasUse;
//         LiveIn[BB.get()] = false;
//         LiveOut[BB.get()] = false;
//     }
    
//     // 2. 迭代计算live-in和live-out直到收敛
//     bool changed = true;
//     while (changed) {
//         changed = false;
        
//         // 使用逆转后序遍历以获得更快的收敛
//         auto postOrder = function->getPostOrder();
//         std::reverse(postOrder.begin(), postOrder.end());
        
//         for (BasicBlock *BB : postOrder) {
//             // LiveOut[BB] = ∪ LiveIn[successor] for all successors
//             bool newLiveOut = false;
//             for (BasicBlock *Succ : BB->getSuccessors()) {
//                 if (LiveIn[Succ]) {
//                     newLiveOut = true;
//                     break;
//                 }
//             }
            
//             // LiveIn[BB] = Use[BB] ∪ (LiveOut[BB] - Def[BB])
//             bool newLiveIn = Use[BB] || (newLiveOut && !Def[BB]);
            
//             if (LiveOut[BB] != newLiveOut || LiveIn[BB] != newLiveIn) {
//                 changed = true;
//                 LiveOut[BB] = newLiveOut;
//                 LiveIn[BB] = newLiveIn;
//             }
//         }
//     }
    
//     // 3. 根据活跃性信息构建活跃区间
//     std::map<SlotIndex, VNInfo*> DefToVNInfo; // 记录定义点到VNInfo的映射
    
//     // 再次遍历所有基本块，构建活跃区间段
//     auto postOrder = function->getPostOrder();
//     std::reverse(postOrder.begin(), postOrder.end());
    
//     for (BasicBlock *BB : postOrder) {
//         SlotIndex bbStart = getBBStartIdx(BB);
//         SlotIndex bbEnd = getBBEndIdx(BB);
        
//         VNInfo *currentVNI = nullptr;
//         SlotIndex segmentStart;
        
//         // 如果寄存器在基本块开始时是活跃的
//         if (LiveIn[BB]) {
//             // 寻找这个活跃值的来源
//             // 从前驱基本块中找到最近的定义
//             for (BasicBlock *Pred : BB->getPredecessors()) {
//                 for (auto it = Pred->rbegin(); it != Pred->rend(); ++it) {
//                     auto definedRegs = (*it)->getDefinedIntegerRegs();
//                     if (std::find(definedRegs.begin(), definedRegs.end(), Reg.getRegNum()) != definedRegs.end()) {
//                         SlotIndex defIdx = getInstructionIndex(**it);
//                         auto vnIt = DefToVNInfo.find(defIdx);
//                         if (vnIt != DefToVNInfo.end()) {
//                             currentVNI = vnIt->second;
//                         }
//                         break;
//                     }
//                 }
//                 if (currentVNI) break;
//             }
            
//             // 如果没找到定义，可能需要创建PHI定义
//             if (!currentVNI) {
//                 currentVNI = LI.createValueAt(bbStart);
//                 DefToVNInfo[bbStart] = currentVNI;
//             }
            
//             segmentStart = bbStart;
//         }
        
//         // 处理基本块内的每条指令
//         for (auto &I : *BB) {
//             SlotIndex instrIdx = getInstructionIndex(*I);
            
//             // 检查是否有定义
//             auto definedRegs = I->getDefinedIntegerRegs();
//             bool hasDef = std::find(definedRegs.begin(), definedRegs.end(), Reg.getRegNum()) != definedRegs.end();
            
//             if (hasDef) {
//                 // 如果之前有活跃段，先结束它
//                 if (currentVNI && segmentStart.isValid()) {
//                     LI.addSegment(LiveInterval::Segment(segmentStart, instrIdx, currentVNI));
//                 }
                
//                 // 创建新的定义
//                 currentVNI = LI.createValueAt(instrIdx);
//                 DefToVNInfo[instrIdx] = currentVNI;
//                 segmentStart = instrIdx;
//             }
            
//             // 检查是否有使用
//             auto usedRegs = I->getUsedIntegerRegs();
//             bool hasUse = std::find(usedRegs.begin(), usedRegs.end(), Reg.getRegNum()) != usedRegs.end();
            
//             // 如果有使用但没有当前活跃的VNI，这是个错误状态
//             if (hasUse && !currentVNI) {
//                 // 可能是从其他地方传入的值，创建一个虚拟的定义
//                 currentVNI = LI.createValueAt(bbStart);
//                 DefToVNInfo[bbStart] = currentVNI;
//                 segmentStart = bbStart;
//             }
//         }
        
//         // 处理基本块结束
//         if (currentVNI && segmentStart.isValid()) {
//             SlotIndex segmentEnd;
            
//             if (LiveOut[BB]) {
//                 // 如果live-out，活跃到基本块结束
//                 segmentEnd = bbEnd;
//             } else {
//                 // 否则找到基本块内最后一个使用点
//                 segmentEnd = segmentStart.getNextSlot(); // 至少活跃一个slot
                
//                 for (auto &I : *BB) {
//                     SlotIndex instrIdx = getInstructionIndex(*I);
//                     if (instrIdx >= segmentStart) {
//                         auto usedRegs = I->getUsedIntegerRegs();
//                         if (std::find(usedRegs.begin(), usedRegs.end(), Reg.getRegNum()) != usedRegs.end()) {
//                             segmentEnd = instrIdx.getNextSlot();
//                         }
//                     }
//                 }
//             }
            
//             LI.addSegment(LiveInterval::Segment(segmentStart, segmentEnd, currentVNI));
//         }
//     }
    
//     return LI;
// }

LiveInterval &LiveIntervals::createAndComputeVirtRegInterval(RegisterOperand Reg) {
    LiveInterval &LI = createEmptyInterval(Reg);

    // 1. 收集所有定义点
    std::map<SlotIndex, VNInfo*> DefToVNInfo;
    for (auto &BB : *function) {
        for (auto &I : *BB) {
            auto definedRegs = I->getDefinedIntegerRegs();
            if (std::find(definedRegs.begin(), definedRegs.end(), Reg.getRegNum()) != definedRegs.end()) {
                SlotIndex defIdx = getInstructionIndex(*I);
                VNInfo *VNI = LI.createValueAt(defIdx);
                DefToVNInfo[defIdx] = VNI;
            }
        }
    }

    // 2. 计算每个基本块的def/use信息
    std::map<BasicBlock*, bool> Def, Use, LiveIn, LiveOut;
    std::map<BasicBlock*, std::vector<SlotIndex>> DefsInBB;

    for (auto &BB : *function) {
        bool hasDef = false;
        bool hasUse = false;
        std::vector<SlotIndex> defs;
        
        for (auto &I : *BB) {
            SlotIndex instrIdx = getInstructionIndex(*I);
            
            // 检查使用（在所有定义之前的使用）
            if (!hasDef) {
                auto usedRegs = I->getUsedIntegerRegs();
                if (std::find(usedRegs.begin(), usedRegs.end(), Reg.getRegNum()) != usedRegs.end()) {
                    hasUse = true;
                }
            }
            
            // 检查定义（可能有多个）
            auto definedRegs = I->getDefinedIntegerRegs();
            if (std::find(definedRegs.begin(), definedRegs.end(), Reg.getRegNum()) != definedRegs.end()) {
                hasDef = true;
                defs.push_back(instrIdx);
            }
        }
        
        Def[BB.get()] = hasDef;
        Use[BB.get()] = hasUse;
        DefsInBB[BB.get()] = defs;
        LiveIn[BB.get()] = false;
        LiveOut[BB.get()] = false;
    }

    // 3. 活跃性分析
    bool changed = true;
    auto postOrder = function->getPostOrder();
    std::reverse(postOrder.begin(), postOrder.end());

    while (changed) {
        changed = false;
        for (BasicBlock *BB : postOrder) {
            bool newLiveOut = false;
            for (BasicBlock *Succ : BB->getSuccessors()) {
                if (LiveIn[Succ]) {
                    newLiveOut = true;
                    break;
                }
            }
            bool newLiveIn = Use[BB] || (newLiveOut && !Def[BB]);

            if (LiveOut[BB] != newLiveOut || LiveIn[BB] != newLiveIn) {
                LiveOut[BB] = newLiveOut;
                LiveIn[BB] = newLiveIn;
                changed = true;
            }
        }
    }

    // 4. 检测PHI节点需求的辅助函数
    auto needsPHI = [&](BasicBlock *BB) -> bool {
        if (BB->getPredecessors().size() <= 1) return false;
        if (!LiveIn[BB]) return false;
        
        // 检查是否有多个前驱都有到达该基本块的定义
        int reachingDefs = 0;
        for (BasicBlock *Pred : BB->getPredecessors()) {
            // 检查前驱是否有定义或者从更前面传来定义
            if (Def[Pred] || LiveIn[Pred]) {
                reachingDefs++;
            }
        }
        return reachingDefs > 1;
    };

    // 5. 改进的找到达定义函数
    std::function<VNInfo*(BasicBlock*)> findReachingDef = [&](BasicBlock *BB) -> VNInfo* {
        // 如果只有一个前驱，直接找那个前驱的定义
        auto preds = BB->getPredecessors();
        if (preds.size() == 1) {
            BasicBlock *Pred = *preds.begin();
            // 先检查前驱块内的定义
            if (Def[Pred] && !DefsInBB[Pred].empty()) {
                SlotIndex lastDef = *DefsInBB[Pred].rbegin();
                return DefToVNInfo[lastDef];
            }
            // 再递归查找更前面的定义
            if (LiveIn[Pred]) {
                return findReachingDef(Pred);
            }
        }
        
        // 多个前驱的情况，随便找一个（PHI情况下会被覆盖）
        for (BasicBlock *Pred : preds) {
            if (Def[Pred] && !DefsInBB[Pred].empty()) {
                SlotIndex lastDef = *DefsInBB[Pred].rbegin();
                return DefToVNInfo[lastDef];
            }
        }
        return nullptr;
    };

    // 6. 构建活跃区间
    for (auto &BB : *function) {
        SlotIndex bbStart = getBBStartIdx(BB.get());
        SlotIndex bbEnd = getBBEndIdx(BB.get());

        VNInfo *activeVNI = nullptr;
        SlotIndex segmentStart;

        // Live-in处理
        if (LiveIn[BB.get()]) {
            if (needsPHI(BB.get())) {
                // 需要PHI定义：在基本块开始创建新的VNInfo
                activeVNI = LI.createValueAt(bbStart);
                DefToVNInfo[bbStart] = activeVNI;
                segmentStart = bbStart;
            } else {
                // 不需要PHI：找到唯一的到达定义
                activeVNI = findReachingDef(BB.get());
                if (!activeVNI) {
                    // 如果还是找不到，创建一个
                    activeVNI = LI.createValueAt(bbStart);
                    DefToVNInfo[bbStart] = activeVNI;
                }
                segmentStart = bbStart;
            }
        }

        // 处理基本块内的每个定义
        for (SlotIndex defPoint : DefsInBB[BB.get()]) {
            // 结束前一个段
            if (activeVNI && segmentStart.isValid()) {
                LI.addSegment(LiveInterval::Segment(segmentStart, defPoint, activeVNI));
            }
            
            // 开始新段
            activeVNI = DefToVNInfo[defPoint];
            segmentStart = defPoint;
        }

        // 处理段的结束
        if (activeVNI && segmentStart.isValid()) {
            SlotIndex segmentEnd;
            if (LiveOut[BB.get()]) {
                segmentEnd = bbEnd;
            } else {
                // 找到最后一次使用
                segmentEnd = segmentStart.getNextSlot();
                for (auto &I : *BB) {
                    SlotIndex instrIdx = getInstructionIndex(*I);
                    if (instrIdx >= segmentStart) {
                        auto usedRegs = I->getUsedIntegerRegs();
                        if (std::find(usedRegs.begin(), usedRegs.end(), Reg.getRegNum()) != usedRegs.end()) {
                            segmentEnd = instrIdx.getNextSlot();
                        }
                    }
                }
            }
            LI.addSegment(LiveInterval::Segment(segmentStart, segmentEnd, activeVNI));
        }
    }

    return LI;
}




LiveInterval &LiveIntervals::getOrCreateEmptyInterval(RegisterOperand Reg) {
    auto it = VirtRegIntervals.find(Reg);
    if (it != VirtRegIntervals.end()) {
        return *it->second;
    }

    return createEmptyInterval(Reg);
}

void LiveIntervals::removeInterval(RegisterOperand Reg) {
    auto it = VirtRegIntervals.find(Reg);
    if (it != VirtRegIntervals.end()) {
        delete it->second;
        VirtRegIntervals.erase(it);
    }
}

LiveInterval::Segment LiveIntervals::addSegmentToEndOfBlock(
    RegisterOperand RegOp, Instruction &startInst) {
    LiveInterval &LI = getOrCreateEmptyInterval(RegOp);
    SlotIndex start = getInstructionIndex(startInst);

    // 获取基本块结束索引
    BasicBlock *BB = startInst.getParent();
    SlotIndex end = getBBEndIdx(BB);

    // 创建值号信息
    VNInfo *VNI = LI.createValueAt(start);
    LiveInterval::Segment segment(start, end, VNI);
    LI.addSegment(segment);
    return segment;
}

bool LiveIntervals::shrinkToUses(LiveInterval *li,
                                 std::vector<Instruction *> *dead) {
    if (!li || li->empty()) {
        return false;
    }

    RegisterOperand Reg = li->reg();
    bool Changed = false;

    // 收集所有实际的使用点和定义点
    std::vector<SlotIndex> UsePoints;
    std::vector<SlotIndex> DefPoints;
    std::vector<Instruction *> DeadInsts;

    // 遍历所有指令查找使用和定义
    // 逆转后序遍历得到逆后序
    auto postOrderBlocks = function->getPostOrder();
    std::reverse(postOrderBlocks.begin(), postOrderBlocks.end());

    // TODO: float
    for (BasicBlock *BB : postOrderBlocks) {
        for (auto &I : *BB) {
            SlotIndex idx = getInstructionIndex(*I);

            // 检查是否定义了这个寄存器
            auto definedRegs = I->getDefinedIntegerRegs();
            bool isDef = std::find(definedRegs.begin(), definedRegs.end(),
                                   Reg.getRegNum()) != definedRegs.end();

            // 检查是否使用了这个寄存器
            auto usedRegs = I->getUsedIntegerRegs();
            bool isUse = std::find(usedRegs.begin(), usedRegs.end(),
                                   Reg.getRegNum()) != usedRegs.end();

            if (isDef) {
                DefPoints.push_back(idx);

                // 检查这个定义是否有实际使用
                if (!isUse) {
                    // 检查这个定义点之后是否有使用
                    bool hasUseAfterDef = false;
                    SlotIndex nextSlot = idx.getNextSlot();

                    // 检查后续是否有使用
                    for (auto segIt = li->find(nextSlot); segIt != li->end();
                         ++segIt) {
                        if (segIt->valno && segIt->valno->def == idx) {
                            // 这个段属于当前定义，检查是否有实际使用
                            // 简化处理：如果段长度大于1个slot，认为有使用
                            if (segIt->end > nextSlot) {
                                hasUseAfterDef = true;
                                break;
                            }
                        }
                    }

                    if (!hasUseAfterDef) {
                        DeadInsts.push_back(I.get());
                    }
                }
            }

            if (isUse) {
                UsePoints.push_back(idx);
            }
        }
    }

    // 创建新的活跃区间
    std::vector<LiveInterval::Segment> NewSegments;

    // 为每个定义点创建段
    for (SlotIndex defIdx : DefPoints) {
        VNInfo *defVNI = li->getVNInfoAt(defIdx);
        if (!defVNI || defVNI->def != defIdx) {
            continue;
        }

        SlotIndex segStart = defIdx;
        SlotIndex segEnd = defIdx.getNextSlot();  // 至少活跃到定义后

        // 找到这个定义的所有使用点
        bool hasUse = false;
        for (SlotIndex useIdx : UsePoints) {
            if (useIdx > defIdx) {
                // 检查这个使用是否属于当前定义
                VNInfo *useVNI = li->getVNInfoAt(useIdx);
                if (useVNI == defVNI) {
                    segEnd = std::max(segEnd, useIdx.getNextSlot());
                    hasUse = true;
                }
            }
        }

        // 如果有使用或者定义本身也是使用，创建段
        if (hasUse || std::find(UsePoints.begin(), UsePoints.end(), defIdx) !=
                          UsePoints.end()) {
            NewSegments.push_back(
                LiveInterval::Segment(segStart, segEnd, defVNI));
        } else {
            // 这个定义没有使用，标记值为未使用
            defVNI->markUnused();
            Changed = true;
        }
    }

    // 处理PHI定义（在基本块边界的定义）
    for (auto &VNIPtr : li->valnos()) {
        if (VNIPtr->isPHIDef() && !VNIPtr->isUnused()) {
            SlotIndex phiIdx = VNIPtr->def;
            SlotIndex segEnd = phiIdx.getNextSlot();

            // 找到PHI的使用点
            bool hasUse = false;
            for (SlotIndex useIdx : UsePoints) {
                if (useIdx >= phiIdx) {
                    VNInfo *useVNI = li->getVNInfoAt(useIdx);
                    if (useVNI == VNIPtr.get()) {
                        segEnd = std::max(segEnd, useIdx.getNextSlot());
                        hasUse = true;
                    }
                }
            }

            if (hasUse) {
                NewSegments.push_back(
                    LiveInterval::Segment(phiIdx, segEnd, VNIPtr.get()));
            } else {
                VNIPtr->markUnused();
                Changed = true;
            }
        }
    }

    // 更新活跃区间
    if (!NewSegments.empty() || Changed) {
        // 清除旧段
        li->segments().clear();

        // 添加新段并排序
        for (const auto &seg : NewSegments) {
            li->addSegment(seg);
        }

        // 处理子范围
        if (li->hasSubRanges()) {
            for (auto *SR = li->subrange_begin(); SR != li->subrange_end();
                 SR = SR->Next) {
                shrinkToUses(*SR, Reg);
            }
        }

        Changed = true;
    }

    // 输出死代码指令
    if (dead) {
        dead->insert(dead->end(), DeadInsts.begin(), DeadInsts.end());
    }

    // 如果整个区间变空，标记为需要移除
    if (li->empty()) {
        Changed = true;
    }

    return Changed;
}

void LiveIntervals::shrinkToUses(LiveInterval::SubRange &SR,
                                 RegisterOperand RegOp) {
    if (SR.empty()) {
        return;
    }

    // 收集实际的使用点和定义点
    std::vector<SlotIndex> UsePoints;
    std::vector<SlotIndex> DefPoints;

    // 遍历所有指令查找使用和定义（使用逆后序）
    // 逆转后序遍历得到逆后序
    auto postOrderBlocks = function->getPostOrder();
    std::reverse(postOrderBlocks.begin(), postOrderBlocks.end());
    // TODO: float
    for (BasicBlock *BB : postOrderBlocks) {
        for (auto &InstPtr : *BB) {
            Instruction &I = *InstPtr;
            SlotIndex idx = getInstructionIndex(I);

            // 检查是否定义了这个寄存器
            auto definedRegs = I.getDefinedIntegerRegs();  // 使用更正后的方法名
            bool isDef = std::find(definedRegs.begin(), definedRegs.end(),
                                   RegOp.getRegNum()) != definedRegs.end();

            // 检查是否使用了这个寄存器
            auto usedRegs = I.getUsedIntegerRegs();  // 使用更正后的方法名
            bool isUse = std::find(usedRegs.begin(), usedRegs.end(),
                                   RegOp.getRegNum()) != usedRegs.end();

            if (isDef) {
                DefPoints.push_back(idx);
            }

            if (isUse) {
                UsePoints.push_back(idx);
            }
        }
    }

    // 创建新的段列表
    std::vector<LiveInterval::Segment> NewSegments;

    // 为每个在子范围中的值定义创建新段
    for (auto &VNI : SR.valnos) {
        if (VNI.isUnused()) {
            continue;
        }

        SlotIndex defIdx = VNI.def;
        SlotIndex segStart = defIdx;
        SlotIndex segEnd = defIdx.getNextSlot();

        // 检查这个定义是否在DefPoints中
        bool isActualDef = std::find(DefPoints.begin(), DefPoints.end(),
                                     defIdx) != DefPoints.end();

        if (isActualDef) {
            // 找到这个定义的所有使用点
            bool hasUse = false;
            for (SlotIndex useIdx : UsePoints) {
                if (useIdx > defIdx) {
                    // 检查这个使用是否在当前值的活跃范围内
                    bool inRange = false;
                    for (const auto &seg : SR.segments) {
                        if (seg.valno == &VNI && seg.contains(useIdx)) {
                            inRange = true;
                            break;
                        }
                    }

                    if (inRange) {
                        segEnd = std::max(segEnd, useIdx.getNextSlot());
                        hasUse = true;
                    }
                }
            }

            // 如果有使用，或者定义点本身也是使用点，创建段
            if (hasUse || std::find(UsePoints.begin(), UsePoints.end(),
                                    defIdx) != UsePoints.end()) {
                NewSegments.push_back(
                    LiveInterval::Segment(segStart, segEnd, &VNI));
            } else {
                // 没有使用，标记为未使用
                VNI.markUnused();
            }
        } else if (VNI.isPHIDef()) {
            // 处理PHI定义
            bool hasUse = false;
            for (SlotIndex useIdx : UsePoints) {
                if (useIdx >= defIdx) {
                    // 检查这个使用是否属于当前PHI值
                    bool inRange = false;
                    for (const auto &seg : SR.segments) {
                        if (seg.valno == &VNI && seg.contains(useIdx)) {
                            inRange = true;
                            break;
                        }
                    }

                    if (inRange) {
                        segEnd = std::max(segEnd, useIdx.getNextSlot());
                        hasUse = true;
                    }
                }
            }

            if (hasUse) {
                NewSegments.push_back(
                    LiveInterval::Segment(segStart, segEnd, &VNI));
            } else {
                VNI.markUnused();
            }
        }
    }

    // 更新子范围的段
    SR.segments.clear();
    SR.segments = std::move(NewSegments);

    // 按开始位置排序
    std::sort(SR.segments.begin(), SR.segments.end());

    // 合并相邻的段（如果它们使用相同的值定义）
    if (SR.segments.size() > 1) {
        auto writeIt = SR.segments.begin();
        for (auto readIt = SR.segments.begin() + 1; readIt != SR.segments.end();
             ++readIt) {
            if (writeIt->end == readIt->start &&
                writeIt->valno == readIt->valno) {
                // 合并段
                writeIt->end = readIt->end;
            } else {
                ++writeIt;
                if (writeIt != readIt) {
                    *writeIt = *readIt;
                }
            }
        }
        SR.segments.erase(writeIt + 1, SR.segments.end());
    }
}

void LiveIntervals::extendToIndices(LiveInterval &LI,
                                    std::vector<SlotIndex> Indices,
                                    std::vector<SlotIndex> Undefs) {
    for (SlotIndex idx : Indices) {
        BasicBlock *BB = getBBFromIndex(idx);
        if (BB) {
            SlotIndex bbStart = getBBStartIdx(BB);
            // 扩展区间到该索引
            VNInfo *VNI = LI.createValueAt(bbStart);
            LiveInterval::Segment segment(bbStart, idx.getNextSlot(), VNI);
            LI.addSegment(segment);
        }
    }

    // 处理未定义的索引
    for (SlotIndex undef : Undefs) {
        // 标记为未定义
    }
}

void LiveIntervals::extendToIndices(LiveInterval &LI,
                                    std::vector<SlotIndex> Indices) {
    std::vector<SlotIndex> empty;
    extendToIndices(LI, Indices, empty);
}

void LiveIntervals::pruneValue(LiveInterval &LI, SlotIndex Kill,
                               std::vector<SlotIndex> *EndPoints) {
    // 修剪活跃区间到Kill点
    LI.removeSegment(Kill, LI.endIndex());

    if (EndPoints) {
        EndPoints->push_back(Kill);
    }
}

bool LiveIntervals::isNotInMIMap(const Instruction &Instr) const {
    return !Indexes || !Indexes->hasIndex(Instr);
}

SlotIndex LiveIntervals::getInstructionIndex(const Instruction &Instr) const {
    if (!Indexes) {
        return SlotIndex();
    }
    return Indexes->getInstructionIndex(Instr);
}

Instruction *LiveIntervals::getInstructionFromIndex(SlotIndex index) const {
    if (!Indexes) {
        return nullptr;
    }
    return Indexes->getInstructionFromIndex(index);
}

SlotIndex LiveIntervals::getBBStartIdx(const BasicBlock *bb) const {
    if (!Indexes) {
        return SlotIndex();
    }
    return Indexes->getBBStartIdx(bb);
}

SlotIndex LiveIntervals::getBBEndIdx(const BasicBlock *bb) const {
    if (!Indexes) {
        return SlotIndex();
    }
    return Indexes->getBBEndIdx(bb);
}

bool LiveIntervals::isLiveInToBB(const LiveInterval &LI,
                                 const BasicBlock *bb) const {
    SlotIndex start = getBBStartIdx(bb);
    return LI.liveAt(start);
}

bool LiveIntervals::isLiveOutOfBB(const LiveInterval &LI,
                                  const BasicBlock *bb) const {
    SlotIndex end = getBBEndIdx(bb);
    return LI.liveAt(end.getPrevSlot());
}

BasicBlock *LiveIntervals::getBBFromIndex(SlotIndex index) const {
    if (!Indexes) {
        return nullptr;
    }
    return Indexes->getBBFromIndex(index);
}

void LiveIntervals::insertBBInMaps(BasicBlock *BB) {
    if (Indexes) {
        Indexes->insertBBInMaps(BB);
    }
}

SlotIndex LiveIntervals::InsertInstructionInMaps(Instruction &I) {
    if (!Indexes) {
        return SlotIndex();
    }
    return Indexes->insertInstrInMaps(I);
}

void LiveIntervals::InsertInstructionRangeInMaps(BasicBlock::iterator B,
                                                 BasicBlock::iterator E) {
    if (Indexes) {
        for (auto it = B; it != E; ++it) {
            Indexes->insertInstrInMaps(**it);
        }
    }
}

void LiveIntervals::RemoveInstructionFromMaps(Instruction &I) {
    if (Indexes) {
        Indexes->removeInstrFromMaps(I);
    }
}

SlotIndex LiveIntervals::ReplaceInstructionInMaps(Instruction &I,
                                                  Instruction &NewI) {
    if (!Indexes) {
        return SlotIndex();
    }
    return Indexes->replaceInstrInMaps(I, NewI);
}

void LiveIntervals::removePhysRegDefAt(RegisterOperand RegOp, SlotIndex Pos) {
    // 检查是否为物理寄存器
    if (RegOp.isVirtual()) {
        return;
    }

    // 检查位置是否有效
    if (!Pos.isValid()) {
        return;
    }

    // 检查该寄存器是否有活跃区间
    if (!hasInterval(RegOp)) {
        return;
    }

    LiveInterval &LI = getInterval(RegOp);

    // 查找在指定位置的值定义
    VNInfo *DefVNI = nullptr;
    for (auto &VNIPtr : LI.valnos()) {
        if (VNIPtr->def == Pos) {
            DefVNI = VNIPtr.get();
            break;
        }
    }

    if (!DefVNI) {
        // 没有找到在该位置的定义，直接返回
        return;
    }

    // 收集需要移除的段
    std::vector<LiveInterval::Segment> SegmentsToRemove;

    // 查找所有使用这个值定义的段
    for (const auto &segment : LI.segments()) {
        if (segment.valno == DefVNI) {
            SegmentsToRemove.push_back(segment);
        }
    }

    // 移除所有相关的段
    for (const auto &segment : SegmentsToRemove) {
        LI.removeSegment(segment, false);  // 暂时不移除死值定义
    }

    // 将值定义标记为未使用
    DefVNI->markUnused();

    // 处理子范围（如果存在）
    if (LI.hasSubRanges()) {
        for (auto *SR = LI.subrange_begin(); SR != LI.subrange_end();
             SR = SR->Next) {
            // 查找在子范围中对应的值定义
            VNInfo *SubDefVNI = nullptr;
            for (auto &VNI : SR->valnos) {
                if (VNI.def == Pos) {
                    SubDefVNI = &VNI;
                    break;
                }
            }

            if (SubDefVNI) {
                // 收集需要移除的子范围段
                std::vector<LiveInterval::Segment> SubSegmentsToRemove;
                for (const auto &segment : SR->segments) {
                    if (segment.valno == SubDefVNI) {
                        SubSegmentsToRemove.push_back(segment);
                    }
                }

                // 移除子范围中的段
                for (const auto &segment : SubSegmentsToRemove) {
                    auto it = std::find(SR->segments.begin(),
                                        SR->segments.end(), segment);
                    if (it != SR->segments.end()) {
                        SR->segments.erase(it);
                    }
                }

                // 标记子范围的值定义为未使用
                SubDefVNI->markUnused();
            }
        }
    }

    // 清理空的子范围
    LI.removeEmptySubRanges();

    // 重新规范化段（确保段按顺序排列且合并相邻段）
    LI.normalizeSegments();

    // 如果整个活跃区间变空了，可以考虑移除它
    if (LI.empty() && !LI.hasSubRanges()) {
        removeInterval(RegOp);
        return;
    }

// 验证区间的一致性（调试模式）
#ifdef DEBUG
    assert(LI.verify() &&
           "Live interval verification failed after removePhysRegDefAt");
#endif
}

void LiveIntervals::removeVRegDefAt(LiveInterval &LI, SlotIndex Pos) {
    // 移除虚拟寄存器在指定位置的定义
    LI.removeSegment(Pos, Pos.getRegSlot());
}

// TODO: this is incomplete.
float LiveIntervals::getSpillWeight(bool isDef, bool isUse,
                                    const Instruction &Instr) {
    float weight = 0.0f;

    if (isDef) weight += 1.0f;
    if (isUse) weight += 1.0f;

    // 可以根据指令类型、循环深度等调整权重
    return weight;
}

// TODO: this is incomplete.
float LiveIntervals::getSpillWeight(bool isDef, bool isUse,
                                    const BasicBlock *BB) {
    float weight = 0.0f;

    if (isDef) weight += 1.0f;
    if (isUse) weight += 1.0f;

    // 可以根据基本块的执行频率调整权重
    return weight;
}

void LiveIntervals::handleMove(Instruction &I, bool UpdateFlags) {
    // 获取指令的当前SlotIndex
    SlotIndex OldIndex = getInstructionIndex(I);
    if (!OldIndex.isValid()) {
        // 如果指令不在映射中，直接返回
        return;
    }

    // 移除指令的旧映射
    RemoveInstructionFromMaps(I);

    // 重新插入指令，获取新的SlotIndex
    SlotIndex NewIndex = InsertInstructionInMaps(I);

    // 收集受影响的寄存器
    std::vector<RegisterOperand> AffectedRegs;

    // 收集指令定义和使用的寄存器
    // 遍历指令的操作数
    for (auto &operand : I.getOperands()) {
        if (operand->isReg()) {
            auto reg = static_cast<RegisterOperand *>(operand.get());
            if (hasInterval(*reg)) {
                AffectedRegs.push_back(*reg);
            }
        }
    }

    // 对每个受影响的寄存器，更新其活跃区间
    for (RegisterOperand reg : AffectedRegs) {
        LiveInterval &LI = getInterval(reg);

        // 查找包含旧索引的段
        auto OldSegIt = LI.find(OldIndex);
        if (OldSegIt != LI.end() && OldSegIt->contains(OldIndex)) {
            // 检查是否需要分割段
            if (OldSegIt->start < OldIndex && OldIndex < OldSegIt->end) {
                // 段被分割，需要处理
                SlotIndex SegStart = OldSegIt->start;
                SlotIndex SegEnd = OldSegIt->end;
                VNInfo *VNI = OldSegIt->valno;

                // 移除旧段
                LI.removeSegment(*OldSegIt);

                // 根据新位置决定如何重建段
                if (NewIndex < SegStart) {
                    // 指令移动到段开始之前
                    LI.addSegment(LiveInterval::Segment(NewIndex, SegEnd, VNI));
                } else if (NewIndex >= SegEnd) {
                    // 指令移动到段结束之后
                    LI.addSegment(LiveInterval::Segment(
                        SegStart, NewIndex.getNextSlot(), VNI));
                } else {
                    // 指令在段内移动
                    LI.addSegment(LiveInterval::Segment(SegStart, SegEnd, VNI));
                }
            }
        }

        // 处理值定义点的移动
        for (auto &VNIPtr : LI.valnos()) {
            if (VNIPtr->def == OldIndex) {
                VNIPtr->def = NewIndex;
            }
        }

        // 处理子范围
        if (LI.hasSubRanges()) {
            for (auto *SR = LI.subrange_begin(); SR != LI.subrange_end();
                 SR = SR->Next) {
                // 更新子范围中的段
                for (auto &segment : SR->segments) {
                    if (segment.contains(OldIndex)) {
                        // 如果段包含旧索引，可能需要调整
                        if (segment.start == OldIndex) {
                            segment.start = NewIndex;
                        }
                        if (segment.end == OldIndex.getNextSlot()) {
                            segment.end = NewIndex.getNextSlot();
                        }
                    }
                }

                // 更新子范围中的值定义
                for (auto &VNI : SR->valnos) {
                    if (VNI.def == OldIndex) {
                        VNI.def = NewIndex;
                    }
                }
            }
        }
    }

    // 如果需要更新标志位
    if (UpdateFlags) {
        // 可以在这里添加额外的标志位更新逻辑
        // 比如更新指令的调试信息、状态标志等
    }

// 验证受影响区间的一致性（调试模式）
#ifdef DEBUG
    for (RegisterOperand reg : AffectedRegs) {
        LiveInterval &LI = getInterval(reg);
        assert(LI.verify() &&
               "Live interval verification failed after handleMove");
    }
#endif
}

void LiveIntervals::repairIntervalsInRange(
    BasicBlock *BB, BasicBlock::iterator Begin, BasicBlock::iterator End,
    const std::vector<RegisterOperand> &OrigRegs) {
    // 修复指定范围内的活跃区间
    for (auto it = Begin; it != End; ++it) {
        // Instruction &I = *it;

        // 重新计算涉及的寄存器的活跃区间
        for (RegisterOperand Reg : OrigRegs) {
            if (hasInterval(Reg)) {
                LiveInterval &LI = getInterval(Reg);
                // 重新计算这部分的活跃区间
            }
        }
    }
}

void LiveIntervals::print(std::ostream &O) const {
    O << "Live Intervals:\n";
    for (const auto &pair : VirtRegIntervals) {
        O << "Register " << pair.first.toString() << ": ";
        pair.second->print(O);
        O << "\n";
    }
}

void LiveIntervals::dump() const { print(std::cerr); }

}  // namespace riscv64
