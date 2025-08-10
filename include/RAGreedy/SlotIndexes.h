#pragma once

#include <cassert>
#include <map>
#include <ostream>
#include <unordered_map>
#include <utility>
#include <algorithm>
#include "Instructions/BasicBlock.h"
namespace riscv64 {

// 前向声明
class IndexListEntry;
class SlotIndexes;

/// SlotIndex - 指令索引的封装类
class SlotIndex {
    friend class SlotIndexes;
    friend class LiveInterval;

    enum Slot {
        /// 基本块边界槽位
        Slot_Block,
        /// 提前清除寄存器使用/定义槽位
        Slot_EarlyClobber,
        /// 正常寄存器使用/定义槽位
        Slot_RegisterOperand,
        /// 死定义终止点
        Slot_Dead,
        Slot_Count
    };

    // 使用pair代替PointerIntPair
    std::pair<IndexListEntry*, unsigned> entry_slot;

    IndexListEntry* listEntry() const {
        assert(isValid() && "Attempt to compare reserved index.");
        return entry_slot.first;
    }

    unsigned getIndex() const;

    /// 返回当前SlotIndex的槽位类型
    Slot getSlot() const { return static_cast<Slot>(entry_slot.second); }

   public:
    enum {
        /// 指令间的默认距离
        InstrDist = 4 * Slot_Count
    };

    /// 构造无效索引
    SlotIndex() : entry_slot(nullptr, 0) {}

    /// 从IndexListEntry和槽位构造SlotIndex
    SlotIndex(IndexListEntry* entry, unsigned slot) : entry_slot(entry, slot) {}

    /// 从现有SlotIndex构造新的SlotIndex并设置槽位
    SlotIndex(const SlotIndex& li, Slot s)
        : entry_slot(li.listEntry(), unsigned(s)) {
        assert(isValid() && "Attempt to construct index with null pointer.");
    }

    /// 检查索引是否有效
    bool isValid() const { return entry_slot.first != nullptr; }

    /// 转换为布尔值
    explicit operator bool() const { return isValid(); }

    /// 打印索引信息
    void print(std::ostream& os) const;

    /// 输出到stderr
    void dump() const;

    /// 比较操作符
    bool operator==(const SlotIndex& other) const {
        return entry_slot == other.entry_slot;
    }

    bool operator!=(const SlotIndex& other) const {
        return entry_slot != other.entry_slot;
    }

    bool operator<(const SlotIndex& other) const {
        return getIndex() < other.getIndex();
    }

    bool operator<=(const SlotIndex& other) const {
        return getIndex() <= other.getIndex();
    }

    bool operator>(const SlotIndex& other) const {
        return getIndex() > other.getIndex();
    }

    bool operator>=(const SlotIndex& other) const {
        return getIndex() >= other.getIndex();
    }

    /// 检查两个SlotIndex是否指向同一条指令
    static bool isSameInstr(const SlotIndex& A, const SlotIndex& B) {
        return A.listEntry() == B.listEntry();
    }

    /// 检查A是否指向比B更早的指令
    static bool isEarlierInstr(const SlotIndex& A, const SlotIndex& B);

    /// 检查A是否指向比B更早或相同的指令
    static bool isEarlierEqualInstr(const SlotIndex& A, const SlotIndex& B) {
        return !isEarlierInstr(B, A);
    }

    /// 计算到另一个索引的距离
    int distance(const SlotIndex& other) const {
        return other.getIndex() - getIndex();
    }

    /// 获取近似指令距离
    int getApproxInstrDistance(const SlotIndex& other) const;

    /// 槽位类型查询
    bool isBlock() const { return getSlot() == Slot_Block; }
    bool isEarlyClobber() const { return getSlot() == Slot_EarlyClobber; }
    bool isRegisterOperand() const { return getSlot() == Slot_RegisterOperand; }
    bool isDead() const { return getSlot() == Slot_Dead; }

    /// 获取不同类型的槽位索引
    SlotIndex getBaseIndex() const {
        return SlotIndex(listEntry(), Slot_Block);
    }

    SlotIndex getBoundaryIndex() const {
        return SlotIndex(listEntry(), Slot_Dead);
    }

    SlotIndex getRegSlot(bool EC = false) const {
        return SlotIndex(listEntry(),
                         EC ? Slot_EarlyClobber : Slot_RegisterOperand);
    }

    SlotIndex getDeadSlot() const { return SlotIndex(listEntry(), Slot_Dead); }

    /// 导航方法
    SlotIndex getNextSlot() const;
    SlotIndex getNextIndex() const;
    SlotIndex getPrevSlot() const;
    SlotIndex getPrevIndex() const;
};

/// 输出流操作符
inline std::ostream& operator<<(std::ostream& os, const SlotIndex& li) {
    li.print(os);
    return os;
}

/// 索引列表条目
class IndexListEntry {
    Instruction* instr;
    unsigned index;

   public:
    IndexListEntry(Instruction* instr, unsigned index) : instr(instr), index(index) {}

    Instruction* getInstr() const { return instr; }
    void setInstr(Instruction* instr) { this->instr = instr; }

    unsigned getIndex() const { return index; }
    void setIndex(unsigned index) { this->index = index; }

    // 用于在list中的迭代
    std::list<IndexListEntry>::iterator getIterator();
};

using IdxBBPair = std::pair<SlotIndex, BasicBlock*>;

/// SlotIndexes - 槽位索引管理类
class SlotIndexes {
   private:
    // 使用STL容器替代LLVM特有数据结构
    std::list<IndexListEntry> indexList;
    Function* func = nullptr;

    using Instr2IndexMap = std::unordered_map<const Instruction*, SlotIndex>;
    Instr2IndexMap instr2iMap;

    /// BB范围映射 - 基本块编号到(开始,结束)索引的映射
    std::unordered_map<std::string, std::pair<SlotIndex, SlotIndex>> BBRanges;

    /// 索引到基本块的映射 - 按索引排序的(索引,基本块)对列表
    std::vector<std::pair<SlotIndex, BasicBlock*>> idx2BBMap;

    void clear();
    void analyze(Function& F);

    IndexListEntry* createEntry(Instruction* instr, unsigned index) {
        indexList.emplace_back(instr, index);
        return &indexList.back();
    }

    /// 在插入curItr后重新编号
    void renumberIndexes(std::list<IndexListEntry>::iterator curItr);

   public:
    SlotIndexes() = default;
    SlotIndexes(Function& F) { analyze(F); }
    ~SlotIndexes() = default;

    // 移动构造函数
    SlotIndexes(SlotIndexes&&) = default;
    SlotIndexes& operator=(SlotIndexes&&) = default;

    void reanalyze(Function& F) {
        clear();
        func = &F;
        analyze(F);
    }

    void print(std::ostream& OS) const;
    void dump() const;

    /// 修复指定范围内的索引
    void repairIndexesInRange(BasicBlock* BB,
                              std::vector<Instruction*>::iterator Begin,
                              std::vector<Instruction*>::iterator End);

    /// 获取零索引
    SlotIndex getZeroIndex() {
        assert(!indexList.empty() && indexList.front().getIndex() == 0 &&
               "First index is not 0?");
        return SlotIndex(&indexList.front(), 0);
    }

    /// 获取最后索引
    SlotIndex getLastIndex() { return SlotIndex(&indexList.back(), 0); }

    /// 检查指令是否有索引
    bool hasIndex(const Instruction& instr) const {
        return instr2iMap.find(&instr) != instr2iMap.end();
    }

    /// 获取指令的索引
    SlotIndex getInstructionIndex(const Instruction& instr,
                                  bool IgnoreBundle = false) const {
        auto it = instr2iMap.find(&instr);
        assert(it != instr2iMap.end() && "Instruction not found in maps.");
        return it->second;
    }

    /// 根据索引获取指令
    Instruction* getInstructionFromIndex(const SlotIndex& index) const {
        return index.listEntry()->getInstr();
    }

    /// 获取下一个非空索引
    SlotIndex getNextNonNullIndex(const SlotIndex& Index);

    /// 获取指令前的索引
    SlotIndex getIndexBefore(const Instruction& instr) const;

    /// 获取指令后的索引
    SlotIndex getIndexAfter(const Instruction& instr) const;

    /// 基本块范围相关方法
    const std::pair<SlotIndex, SlotIndex>& getBBRange(const std::string& BBLabel) const {
        return BBRanges.at(BBLabel);
    }

    const std::pair<SlotIndex, SlotIndex>& getBBRange(
        const BasicBlock* BB) const;

    SlotIndex getBBStartIdx(const std::string& BBLabel) const {
        return getBBRange(BBLabel).first;
    }

    SlotIndex getBBStartIdx(const BasicBlock* bb) const {
        return getBBRange(bb).first;
    }

    SlotIndex getBBEndIdx(const std::string& BBLabel) const { return getBBRange(BBLabel).second; }

    SlotIndex getBBEndIdx(const BasicBlock* bb) const {
        return getBBRange(bb).second;
    }

    /// 基本块迭代器类型
    using BBIndexIterator = std::vector<std::pair<SlotIndex, BasicBlock*>>::const_iterator;

    /// 获取基本块迭代器
    BBIndexIterator getBBLowerBound(BBIndexIterator Start,
                                    const SlotIndex& Idx) const {
        return std::lower_bound(Start, BBIndexEnd(), Idx,
                                [](const IdxBBPair& IM, const SlotIndex& Idx) {
                                    return IM.first < Idx;
                                });
    }

    BBIndexIterator getBBLowerBound(const SlotIndex& Idx) const {
        return getBBLowerBound(BBIndexBegin(), Idx);
    }

    BBIndexIterator getBBUpperBound(const SlotIndex& Idx) const {
        return std::upper_bound(BBIndexBegin(), BBIndexEnd(), Idx,
                                [](const SlotIndex& Idx, const IdxBBPair& IM) {
                                    return Idx < IM.first;
                                });
    }

    BBIndexIterator BBIndexBegin() const { return idx2BBMap.begin(); }

    BBIndexIterator BBIndexEnd() const { return idx2BBMap.end(); }

    /// 根据索引获取基本块
    BasicBlock* getBBFromIndex(const SlotIndex& index) const;

    /// 指令操作方法
    SlotIndex insertInstrInMaps(Instruction& instr, bool Late = false);
    void removeInstrFromMaps(Instruction& instr, bool AllowBundled = false);
    void removeSingleInstrFromMaps(Instruction& instr);
    SlotIndex replaceInstrInMaps(Instruction& oldInstr, Instruction& newInstr);

    /// 基本块操作
    void insertBBInMaps(BasicBlock* bb);

    /// 重新打包所有索引
    void packIndexes();
};

class SlotIndexesWrapperPass {
private:
    SlotIndexes SI;

public:
    SlotIndexesWrapperPass() {
        // 构造函数初始化
    }

    void getAnalysisUsage() const {
        
    }

    // 运行pass，即为指定函数构造SlotIndexes分析
    bool runOnFunction(Function* fn) {
        SI.reanalyze(*fn);
        return false; // 表示没有修改函数
    }

    // 获取SlotIndexes实例
    SlotIndexes& getSI() {
        return SI;
    }
};


}  // namespace riscv64