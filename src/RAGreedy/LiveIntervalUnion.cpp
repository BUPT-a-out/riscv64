#include "RAGreedy/LiveIntervalUnion.h"

#include <algorithm>

namespace riscv64 {

// LiveIntervalUnion 实现

LiveIntervalUnion::SegmentIter LiveIntervalUnion::begin() {
    return Segments.begin();
}

LiveIntervalUnion::SegmentIter LiveIntervalUnion::end() {
    return Segments.end();
}

LiveIntervalUnion::SegmentIter LiveIntervalUnion::find(SlotIndex x) {
    return Segments.lower_bound(x);
}

LiveIntervalUnion::ConstSegmentIter LiveIntervalUnion::begin() const {
    return Segments.begin();
}

LiveIntervalUnion::ConstSegmentIter LiveIntervalUnion::end() const {
    return Segments.end();
}

LiveIntervalUnion::ConstSegmentIter LiveIntervalUnion::find(SlotIndex x) const {
    return Segments.lower_bound(x);
}

bool LiveIntervalUnion::empty() const { return Segments.empty(); }

SlotIndex LiveIntervalUnion::startIndex() const {
    return empty() ? SlotIndex() : Segments.begin()->first;
}

SlotIndex LiveIntervalUnion::endIndex() const {
    return empty() ? SlotIndex() : Segments.rbegin()->first;
}

unsigned LiveIntervalUnion::getTag() const { return Tag; }

bool LiveIntervalUnion::changedSince(unsigned tag) const { return tag != Tag; }

void LiveIntervalUnion::unify(std::shared_ptr<const LiveInterval> VirtReg,
                              const LiveInterval &Interval) {
    if (Interval.empty()) return;
    ++Tag;

    // 使用LiveInterval的segments接口
    for (auto segment: Interval.segments()) {
        Segments[segment.start] = VirtReg;
    }
}

void LiveIntervalUnion::extract(std::shared_ptr<const LiveInterval> VirtReg,
                                const LiveInterval &Interval) {
    if (Interval.empty()) return;
    ++Tag;

    // 使用LiveInterval的segments接口
    for (auto segment: Interval.segments()) {
        auto it = Segments.find(segment.start);
        if (it != Segments.end() && it->second == VirtReg) {
            Segments.erase(it);
        }
    }
}

void LiveIntervalUnion::clear() {
    Segments.clear();
    ++Tag;
}

void LiveIntervalUnion::print(std::ostream &OS) const {
    if (empty()) {
        OS << " empty\n";
        return;
    }
    for (const auto &seg : Segments) {
        OS << " [" << seg.first << "):" << seg.second->reg().getRegNum();
    }
    OS << '\n';
}

std::shared_ptr<const LiveInterval> LiveIntervalUnion::getOneVReg() const {
    if (empty()) return nullptr;
    return Segments.begin()->second;
}

// Query 类实现

LiveIntervalUnion::Query::Query(const LiveInterval &li,
                                const LiveIntervalUnion &liu)
    : LiveUnion(&liu), LI(&li) {}

void LiveIntervalUnion::Query::reset(unsigned NewUserTag,
                                     const LiveInterval &NewLI,
                                     const LiveIntervalUnion &NewLiveUnion) {
    LiveUnion = &NewLiveUnion;
    LI = &NewLI;
    InterferingVRegs.clear();
    CheckedFirstInterference = false;
    SeenAllInterferences = false;
    Tag = NewLiveUnion.getTag();
    UserTag = NewUserTag;
    SegmentIdx = 0;
}

void LiveIntervalUnion::Query::init(unsigned NewUserTag,
                                    const LiveInterval &NewLI,
                                    const LiveIntervalUnion &NewLiveUnion) {
    if (UserTag == NewUserTag && LI == &NewLI && LiveUnion == &NewLiveUnion &&
        !NewLiveUnion.changedSince(Tag)) {
        return;
    }
    reset(NewUserTag, NewLI, NewLiveUnion);
}

bool LiveIntervalUnion::Query::checkInterference() {
    return collectInterferingVRegs(1) > 0;
}

const std::vector<std::shared_ptr<const LiveInterval>> &
LiveIntervalUnion::Query::interferingVRegs(unsigned MaxInterferingRegs) {
    if (!SeenAllInterferences || MaxInterferingRegs < InterferingVRegs.size())
        collectInterferingVRegs(MaxInterferingRegs);
    return InterferingVRegs;
}

unsigned LiveIntervalUnion::Query::collectInterferingVRegs(
    unsigned MaxInterferingRegs) {
    if (SeenAllInterferences || InterferingVRegs.size() >= MaxInterferingRegs)
        return InterferingVRegs.size();

    if (!CheckedFirstInterference) {
        CheckedFirstInterference = true;
        if (LI->empty() || LiveUnion->empty()) {
            SeenAllInterferences = true;
            return 0;
        }
        SegmentIdx = 0;
        LiveUnionI = LiveUnion->Segments.begin();
    }

    // 使用LiveInterval的segments接口进行干扰检测
    auto liSegments = LI->segments();
    for (size_t i = SegmentIdx; i < liSegments.size(); ++i) {
        const auto& segment = liSegments[i];
        
        // 从当前位置开始遍历union中的区间
        while (LiveUnionI != LiveUnion->Segments.end()) {
            const auto& unionSlot = LiveUnionI->first;
            const auto& unionInterval = LiveUnionI->second;
            
            // 如果union区间的起始点在当前segment结束之后，跳出内层循环
            if (unionSlot >= segment.end) {
                break;
            }
            
            // 检查union中的LiveInterval与当前segment是否有重叠
            bool hasOverlap = false;
            for (const auto& unionSeg : unionInterval->segments()) {
                // 完整的区间重叠检测
                if (segment.start < unionSeg.end && unionSeg.start < segment.end) {
                    hasOverlap = true;
                    break;
                }
            }
            
            if (hasOverlap) {
                // 检查是否已经在干扰列表中
                if (!isSeenInterference(unionInterval)) {
                    InterferingVRegs.push_back(unionInterval);
                    if (InterferingVRegs.size() >= MaxInterferingRegs) {
                        // 保存当前状态以便下次继续
                        SegmentIdx = i;
                        return InterferingVRegs.size();
                    }
                }
            }
            
            ++LiveUnionI;
        }
        
        // 重置union迭代器为下一个segment做准备
        LiveUnionI = LiveUnion->Segments.begin();
    }

    // 所有segment都检查完毕
    SeenAllInterferences = true;
    return InterferingVRegs.size();
}


bool LiveIntervalUnion::Query::isSeenInterference(
    std::shared_ptr<const LiveInterval> VirtReg) const {
    return std::find(InterferingVRegs.begin(), InterferingVRegs.end(),
                     VirtReg) != InterferingVRegs.end();
}

// Array 类实现

void LiveIntervalUnion::Array::init(unsigned size) {
    if (size == LIUs.size()) return;
    LIUs.clear();
    LIUs.reserve(size);
    for (unsigned i = 0; i < size; ++i) {
        LIUs.push_back(std::make_unique<LiveIntervalUnion>());
    }
}

unsigned LiveIntervalUnion::Array::size() const { return LIUs.size(); }

void LiveIntervalUnion::Array::clear() { LIUs.clear(); }

LiveIntervalUnion &LiveIntervalUnion::Array::operator[](unsigned idx) {
    assert(idx < LIUs.size() && "idx out of bounds");
    return *LIUs[idx];
}

const LiveIntervalUnion &LiveIntervalUnion::Array::operator[](
    unsigned idx) const {
    assert(idx < LIUs.size() && "idx out of bounds");
    return *LIUs[idx];
}

}  // end namespace riscv64
