#pragma once

#include <iterator>
#include <ostream>

#include "Instructions/BasicBlock.h"
#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"
#include "SlotIndexes.h"

namespace riscv64 {

class VNInfo {
   public:
    /// The ID number of this value.
    unsigned id;

    /// The index of the defining instruction.
    SlotIndex def;

    /// VNInfo constructor.
    VNInfo(unsigned i, SlotIndex d) : id(i), def(d) {}

    /// VNInfo constructor, copies values from orig, except for the value
    /// number.
    VNInfo(unsigned i, const VNInfo &orig) : id(i), def(orig.def) {}

    /// Copy from the parameter into this VNInfo.
    void copyFrom(VNInfo &src) { def = src.def; }

    /// Returns true if this value is defined by a PHI instruction (or was,
    /// PHI instructions may have been eliminated).
    /// PHI-defs begin at a block boundary, all other defs begin at register or
    /// EC slots.
    bool isPHIDef() const { return def.isBlock(); }

    /// Returns true if this value is unused.
    bool isUnused() const { return !def.isValid(); }

    /// Mark this value as unused.
    void markUnused() { def = SlotIndex(); }

    void dump() const;
};

class LiveInterval {
    friend class LiveIntervals;

   public:
    struct Segment {
        SlotIndex start;  // Start point of the interval (inclusive)
        SlotIndex end;    // End point of the interval (exclusive)
        VNInfo *valno;    // Value number information

        Segment() : valno(nullptr) {}
        Segment(SlotIndex S, SlotIndex E, VNInfo *V)
            : start(S), end(E), valno(V) {
            assert(S < E && "Cannot create empty or backwards segment");
        }

        /// Return true if the index is covered by this segment.
        bool contains(SlotIndex I) const { return start <= I && I < end; }

        /// Return true if the given interval, [S, E), is covered by this
        /// segment.
        bool containsInterval(SlotIndex S, SlotIndex E) const {
            assert(S < E && "Backwards interval?");
            return start <= S && E <= end;
        }

        bool operator<(const Segment &Other) const {
            return std::tie(start, end) < std::tie(Other.start, Other.end);
        }
        bool operator==(const Segment &Other) const {
            return start == Other.start && end == Other.end;
        }

        void dump() const;
    };
    class SubRange {
       public:
        // LaneBitmask LaneMask;   /// Lane mask for this subrange
        SubRange *Next;                 /// Pointer to next subrange in list
        std::vector<Segment> segments;  /// The segments in this subrange
        std::vector<VNInfo> valnos;     /// Value number list

        SubRange() : Next(nullptr) {}

        ~SubRange() {
            // Clean up VNInfos
            // for (VNInfo *VNI : valnos) delete VNI;
        }

        bool empty() const { return segments.empty(); }
        bool liveAt(SlotIndex Pos) const;
        void print(std::ostream &OS) const;
        void dump() const;
    };

   private:
    RegisterOperand
        reg_;       // The register or stack slot this interval represents
    float weight_;  // The weight of this interval for spilling
    using Segments = std::vector<Segment>;
    Segments segments_;  // The segments that make up this interval
    using ValNos = std::vector<std::unique_ptr<VNInfo>>;
    ValNos valnos_;        // Value number list
    SubRange *SubRanges_;  // List of subranges for this interval

   public:
    /// Construct a live interval for the given register.
    explicit LiveInterval(RegisterOperand Reg, float Weight)
        : reg_(Reg), weight_(Weight), SubRanges_(nullptr) {}

    /// Destructor - Clean up subranges and value numbers
    ~LiveInterval() { clear(); }

    // Disable copy constructor and assignment
    LiveInterval(const LiveInterval &) = delete;
    LiveInterval &operator=(const LiveInterval &) = delete;

    // Enable move constructor and assignment
    LiveInterval(LiveInterval &&Other) noexcept
        : reg_(Other.reg_),
          weight_(Other.weight_),
          segments_(std::move(Other.segments_)),
          valnos_(std::move(Other.valnos_)),
          SubRanges_(Other.SubRanges_) {
        Other.SubRanges_ = nullptr;
    }

    LiveInterval &operator=(LiveInterval &&Other) noexcept {
        if (this != &Other) {
            clear();
            reg_ = Other.reg_;
            weight_ = Other.weight_;
            segments_ = std::move(Other.segments_);
            valnos_ = std::move(Other.valnos_);
            SubRanges_ = Other.SubRanges_;
            Other.SubRanges_ = nullptr;
        }
        return *this;
    }

    RegisterOperand reg() const { return reg_; }
    float weight() const { return weight_; }
    void setWeight(float Weight) { weight_ = Weight; }

    bool empty() const { return segments_.empty(); }
    SlotIndex beginIndex() const {
        assert(!empty() && "Call beginIndex() on empty interval");
        return segments_.front().start;
    }
    SlotIndex endIndex() const {
        assert(!empty() && "Call endIndex() on empty interval");
        return segments_.back().end;
    }

    /// Get the size of the interval in machine instructions
    unsigned getSize() const;

    /// Iterator Interface
    auto begin() { return segments_.begin(); }
    auto end() { return segments_.end(); }
    auto begin() const { return segments_.begin(); }
    auto end() const { return segments_.end(); }

    /// Segment Access
    Segments &segments() { return segments_; }
    const Segments &segments() const { return segments_; }

    /// Value Number Access
    ValNos &valnos() { return valnos_; }
    const ValNos &valnos() const { return valnos_; }

    /// Query Operations
    /// find - Return an iterator pointing to the first segment that ends after
    /// or on Pos, or end() if no such segment exists.
    Segments::iterator find(SlotIndex Pos);
    Segments::const_iterator find(SlotIndex Pos) const;

    /// Return true if the index is live at the given point.
    bool liveAt(SlotIndex Pos) const {
        auto I = find(Pos);
        return I != end() && I->start <= Pos;
    }

    /// Return true if this interval overlaps with another interval.
    bool overlaps(const LiveInterval &Other) const;
    bool overlaps(SlotIndex Start, SlotIndex End) const;

    /// Return the VNInfo that is live at Pos, or nullptr if Pos is not live.
    VNInfo *getVNInfoAt(SlotIndex Pos) const;

    /// Return the VNInfo that ends at or before Pos, nullptr if none exists.
    VNInfo *getVNInfoBefore(SlotIndex Pos) const;

    /// getValNumInfo - Returns pointer to the specified val#.
    VNInfo *getValNumInfo(unsigned ValNo) {
        assert(ValNo < valnos_.size());
        return valnos_[ValNo].get();
    }
    const VNInfo *getValNumInfo(unsigned ValNo) const {
        assert(ValNo < valnos_.size());
        return valnos_[ValNo].get();
    }

    /// Modification Operations
    /// addSegment - Add the specified Segment to this interval, merging
    /// segments as appropriate.
    void addSegment(Segment S);

    /// addSegment - Add the specified Segment to this interval. The segment is
    /// inserted at the specified iterator position.
    Segments::iterator addSegment(Segments::iterator I, Segment S);

    /// removeSegment - Remove the specified segment from this interval.
    void removeSegment(SlotIndex Start, SlotIndex End,
                       bool RemoveDeadValNo = false);
    void removeSegment(Segment S, bool RemoveDeadValNo = false);

    /// Remove all segments.
    void clear();

    /// Value Number Management
    /// Create a new value number and return it.
    VNInfo *createValueAt(SlotIndex Def) {
        auto VNI = std::make_unique<VNInfo>(valnos_.size(), Def);
        VNInfo *ptr = VNI.get();
        valnos_.push_back(std::move(VNI));
        return ptr;
    }

    /// Create a dead def value.
    VNInfo *createDeadDef(SlotIndex Def);

    /// Create a copy of the given value.
    VNInfo *createValueCopy(const VNInfo *Original) {
        auto VNI = std::make_unique<VNInfo>(valnos_.size(), Original->def);
        VNInfo *ptr = VNI.get();
        valnos_.push_back(std::move(VNI));
        return ptr;
    }

    /// RenumberValues - Renumber all values in order of appearance.
    void RenumberValues();

    /// Advanced Operations
    /// join - Join with another live interval.
    void join(LiveInterval &Other);

    /// Verify that this interval is valid and consistent.
    bool verify() const;

    /// SubRange Management
    bool hasSubRanges() const { return SubRanges_ != nullptr; }

    /// Create a new subrange for the given lane mask.
    SubRange *createSubRange();

    /// Remove the subrange for the given lane mask.
    void removeSubRange();

    /// Remove empty subranges.
    void removeEmptySubRanges();

    /// Get the subrange for the given lane mask.
    SubRange *getSubRange();
    const SubRange *getSubRange() const;

    /// SubRange iteration
    using subrange_iterator = SubRange *;
    using const_subrange_iterator = const SubRange *;

    subrange_iterator subrange_begin() { return SubRanges_; }
    subrange_iterator subrange_end() { return nullptr; }
    const_subrange_iterator subrange_begin() const { return SubRanges_; }
    const_subrange_iterator subrange_end() const { return nullptr; }

    /// Comparison Operations
    bool operator<(const LiveInterval &Other) const;
    bool operator==(const LiveInterval &Other) const;

    /// Debugging
    void print(std::ostream &OS) const;
    void dump() const;

   private:
    /// Helper functions
    void mergeSegments();
    void normalizeSegments();
    Segments::iterator findInsertPos(Segment S);
    void clearSubRanges();
};

class LiveIntervals {
   private:
    Function *function;
    SlotIndexes *Indexes = nullptr;
    std::map<RegisterOperand, LiveInterval *> VirtRegIntervals;

   public:
    void analyze(Function &fn);
    void clear();

    // TODO: dom tree
    LiveIntervals(Function *fn, SlotIndexes *si) : function(fn), Indexes(si) {};

    SlotIndexes *getSlotIndexes() const { return Indexes; }

    // 获取寄存器的活跃区间（如果不存在会创建并计算）
    LiveInterval &getInterval(RegisterOperand Reg);
    const LiveInterval &getInterval(RegisterOperand Reg) const;

    // 检查是否已有区间
    bool hasInterval(RegisterOperand Reg) const;

    // 创建空的活跃区间
    LiveInterval &createEmptyInterval(RegisterOperand Reg);

    // 创建并计算虚拟寄存器区间
    LiveInterval &createAndComputeVirtRegInterval(RegisterOperand Reg);

    // 获取或创建空区间（不触发计算）
    LiveInterval &getOrCreateEmptyInterval(RegisterOperand Reg);

    // 移除区间
    void removeInterval(RegisterOperand Reg);

    // 添加段到基本块末尾
    LiveInterval::Segment addSegmentToEndOfBlock(RegisterOperand RegOp,
                                                 Instruction &startInst);

    // 收缩到实际使用点
    bool shrinkToUses(LiveInterval *li,
                      std::vector<Instruction *> *dead = nullptr);
    void shrinkToUses(LiveInterval::SubRange &SR, RegisterOperand RegOp);

    // 扩展区间到指定索引
    void extendToIndices(LiveInterval &LI, std::vector<SlotIndex> Indices,
                         std::vector<SlotIndex> Undefs);
    void extendToIndices(LiveInterval &LI, std::vector<SlotIndex> Indices);

    // 修剪区间
    void pruneValue(LiveInterval &LI, SlotIndex Kill,
                    std::vector<SlotIndex> *EndPoints);
    // 检查指令是否在映射中
    bool isNotInMIMap(const Instruction &Instr) const;

    // 获取指令索引
    SlotIndex getInstructionIndex(const Instruction &Instr) const;

    // 从索引获取指令
    Instruction *getInstructionFromIndex(SlotIndex index) const;
    // 获取基本块的开始和结束索引
    SlotIndex getBBStartIdx(const BasicBlock *bb) const;
    SlotIndex getBBEndIdx(const BasicBlock *bb) const;

    // 检查活跃性
    bool isLiveInToBB(const LiveInterval &LI, const BasicBlock *bb) const;
    bool isLiveOutOfBB(const LiveInterval &LI, const BasicBlock *bb) const;

    // 从索引获取基本块
    BasicBlock *getBBFromIndex(SlotIndex index) const;
    // 插入基本块
    void insertBBInMaps(BasicBlock *BB);

    // 插入指令
    SlotIndex InsertInstructionInMaps(Instruction &I);
    void InsertInstructionRangeInMaps(BasicBlock::iterator B,
                                      BasicBlock::iterator E);

    // 移除指令
    void RemoveInstructionFromMaps(Instruction &I);

    // 替换指令
    SlotIndex ReplaceInstructionInMaps(Instruction &I, Instruction &NewI);
    // // 获取寄存器单元的活跃范围
    // LiveInterval &getRegUnit(unsigned Unit);

    // // 获取已缓存的寄存器单元
    // LiveInterval *getCachedRegUnit(unsigned Unit);
    // const LiveInterval *getCachedRegUnit(unsigned Unit) const;

    // // 移除寄存器单元
    // void removeRegUnit(unsigned Unit);
    // void removeAllRegUnitsForPhysReg(RegisterOperand RegOp);

    // 移除物理寄存器定义
    void removePhysRegDefAt(RegisterOperand RegOp, SlotIndex Pos);
    void removeVRegDefAt(LiveInterval &LI, SlotIndex Pos);

    // 静态方法：计算溢出权重
    static float getSpillWeight(bool isDef, bool isUse,
                                //    const MachineBlockFrequencyInfo *MBFI,
                                const Instruction &Instr
                                //    ProfileSummaryInfo *PSI = nullptr
    );

    static float getSpillWeight(bool isDef, bool isUse,
                                //    const MachineBlockFrequencyInfo *MBFI,
                                const BasicBlock *BB
                                //    ProfileSummaryInfo *PSI = nullptr
    );

    // 区间维护方法
    void handleMove(Instruction &I, bool UpdateFlags = false);
    void repairIntervalsInRange(BasicBlock *BB, BasicBlock::iterator Begin,
                                BasicBlock::iterator End,
                                const std::vector<RegisterOperand> &OrigRegs);

    // 分析和打印方法
    void reanalyze(Function &F) {
        clear();
        analyze(F);
    }
    void print(std::ostream &O) const;
    void dump() const;
};

}  // namespace riscv64