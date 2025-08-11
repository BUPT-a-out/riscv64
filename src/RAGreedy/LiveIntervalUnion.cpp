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
    for (auto segment: LI->segments()) {

        for (auto unionIt = LiveUnion->Segments.begin();
             unionIt != LiveUnion->Segments.end(); ++unionIt) {
            // 检查重叠
            if (segment.start < unionIt->first &&
                segment.end > unionIt->first) {
                auto VReg = unionIt->second;
                if (std::find(InterferingVRegs.begin(), InterferingVRegs.end(),
                              VReg) == InterferingVRegs.end()) {
                    InterferingVRegs.push_back(VReg);
                    if (InterferingVRegs.size() >= MaxInterferingRegs) {
                        return InterferingVRegs.size();
                    }
                }
            }
        }
    }

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
