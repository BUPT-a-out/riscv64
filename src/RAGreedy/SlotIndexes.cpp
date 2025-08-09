#include "RAGreedy/SlotIndexes.h"

#include "Instructions/Function.h"

namespace riscv64 {
// SlotIndex类的未实现方法

unsigned SlotIndex::getIndex() const {
    assert(isValid() && "Attempt to get index of invalid SlotIndex.");
    return listEntry()->getIndex() + getSlot();
}

void SlotIndex::print(std::ostream& os) const {
    if (!isValid()) {
        os << "invalid";
        return;
    }

    os << getIndex();
    switch (getSlot()) {
        case Slot_Block:
            os << "B";
            break;
        case Slot_EarlyClobber:
            os << "e";
            break;
        case Slot_RegisterOperand:
            os << "r";
            break;
        case Slot_Dead:
            os << "d";
            break;
        default:
            os << "?";
            break;
    }
}

void SlotIndex::dump() const {
    print(std::cerr);
    std::cerr << "\n";
}

bool SlotIndex::isEarlierInstr(const SlotIndex& A, const SlotIndex& B) {
    assert(A.isValid() && B.isValid() && "Cannot compare invalid indices.");
    if (A.listEntry() == B.listEntry()) return false;  // 同一条指令
    return A.listEntry()->getIndex() < B.listEntry()->getIndex();
}

int SlotIndex::getApproxInstrDistance(const SlotIndex& other) const {
    int dist = distance(other);
    return dist >= 0 ? (dist + InstrDist / 2) / InstrDist
                     : (dist - InstrDist / 2) / InstrDist;
}

SlotIndex SlotIndex::getNextSlot() const {
    assert(isValid() && "Cannot get next slot of invalid index.");
    if (getSlot() < Slot_Dead) {
        return SlotIndex(listEntry(), getSlot() + 1);
    }
    return getNextIndex();
}

SlotIndex SlotIndex::getNextIndex() const {
    // 这需要SlotIndexes的支持，暂时返回无效索引
    assert(false && "getNextIndex requires SlotIndexes context");
    return SlotIndex();
}

SlotIndex SlotIndex::getPrevSlot() const {
    assert(isValid() && "Cannot get prev slot of invalid index.");
    if (getSlot() > Slot_Block) {
        return SlotIndex(listEntry(), getSlot() - 1);
    }
    return getPrevIndex();
}

SlotIndex SlotIndex::getPrevIndex() const {
    // 这需要SlotIndexes的支持，暂时返回无效索引
    assert(false && "getPrevIndex requires SlotIndexes context");
    return SlotIndex();
}

// IndexListEntry类的未实现方法

std::list<IndexListEntry>::iterator IndexListEntry::getIterator() {
    // 这个实现需要访问包含它的list，通常通过SlotIndexes提供
    assert(false && "getIterator requires SlotIndexes context");
    return std::list<IndexListEntry>::iterator();
}

// SlotIndexes类的未实现方法

void SlotIndexes::clear() {
    indexList.clear();
    instr2iMap.clear();
    BBRanges.clear();
    idx2BBMap.clear();
    func = nullptr;
}

void SlotIndexes::analyze(Function& F) {
    func = &F;
    clear();

    unsigned CurIndex = 0;

    // 为每个基本块分配索引
    for (auto& BB : F) {
        SlotIndex StartIdx =
            SlotIndex(createEntry(nullptr, CurIndex), SlotIndex::Slot_Block);

        // 为基本块中的每条指令分配索引
        for (auto& Instr : *BB) {
            IndexListEntry* Entry = createEntry(Instr.get(), CurIndex);
            SlotIndex InstrIdx(Entry, SlotIndex::Slot_Block);
            instr2iMap[Instr.get()] = InstrIdx;
            CurIndex += SlotIndex::InstrDist;
        }

        SlotIndex EndIdx =
            SlotIndex(createEntry(nullptr, CurIndex), SlotIndex::Slot_Block);

        // 记录基本块范围 - 使用标签
        std::string BBLabel = BB->getLabel();
        BBRanges[BBLabel] = std::make_pair(StartIdx, EndIdx);

        // 添加到 idx2BBMap vector - 正确的方式
        idx2BBMap.emplace_back(StartIdx, BB.get());

        CurIndex += SlotIndex::InstrDist;
    }

    // 排序以保持按 SlotIndex 升序
    std::sort(idx2BBMap.begin(), idx2BBMap.end(),
              [](const std::pair<SlotIndex, BasicBlock*>& A,
                 const std::pair<SlotIndex, BasicBlock*>& B) {
                  return A.first < B.first;
              });
}



void SlotIndexes::print(std::ostream& OS) const {
    OS << "SlotIndexes for function: ";
    if (func) {
        OS << func->getName();
    } else {
        OS << "unknown";
    }
    OS << "\n";

    for (const auto& Entry : indexList) {
        SlotIndex SI(&const_cast<IndexListEntry&>(Entry),
                     SlotIndex::Slot_Block);
        OS << "  ";
        SI.print(OS);
        if (Entry.getInstr()) {
            OS << ": " << Entry.getInstr()->toString();
        } else {
            OS << ": <null>";
        }
        OS << "\n";
    }
}

void SlotIndexes::dump() const { print(std::cerr); }

void SlotIndexes::renumberIndexes(std::list<IndexListEntry>::iterator curItr) {
    unsigned NewIndex = 0;
    if (curItr != indexList.begin()) {
        auto PrevItr = curItr;
        --PrevItr;
        NewIndex = PrevItr->getIndex() + SlotIndex::InstrDist;
    }

    for (auto It = curItr; It != indexList.end(); ++It) {
        It->setIndex(NewIndex);
        NewIndex += SlotIndex::InstrDist;
    }
}

void SlotIndexes::repairIndexesInRange(
    BasicBlock* BB, std::vector<Instruction*>::iterator Begin,
    std::vector<Instruction*>::iterator End) {
    // 重新构建指定范围内的索引
    for (auto It = Begin; It != End; ++It) {
        if (hasIndex(**It)) {
            removeInstrFromMaps(**It);
        }
        insertInstrInMaps(**It);
    }
}

SlotIndex SlotIndexes::getNextNonNullIndex(const SlotIndex& Index) {
    auto Entry = Index.listEntry();
    for (auto& ListEntry : indexList) {
        if (&ListEntry == Entry) {
            auto It = std::find_if(indexList.begin(), indexList.end(),
                                   [&ListEntry](const IndexListEntry& E) {
                                       return &E == &ListEntry;
                                   });
            if (++It != indexList.end()) {
                return SlotIndex(&(*It), SlotIndex::Slot_Block);
            }
            break;
        }
    }
    return SlotIndex();  // 返回无效索引
}

SlotIndex SlotIndexes::getIndexBefore(const Instruction& instr) const {
    auto It = instr2iMap.find(&instr);
    assert(It != instr2iMap.end() && "Instruction not found");

    SlotIndex InstrIdx = It->second;
    if (InstrIdx.getSlot() == SlotIndex::Slot_Block) {
        // 需要找到前一个索引，这里简化处理
        unsigned PrevIndex = InstrIdx.getIndex() - SlotIndex::InstrDist;
        if (PrevIndex < InstrIdx.getIndex()) {  // 检查下溢
            // 寻找对应的IndexListEntry
            for (auto& Entry : indexList) {
                if (Entry.getIndex() == PrevIndex) {
                    return SlotIndex(&const_cast<IndexListEntry&>(Entry),
                                     SlotIndex::Slot_Dead);
                }
            }
        }
    }
    return SlotIndex(InstrIdx.listEntry(), SlotIndex::Slot_Block);
}

SlotIndex SlotIndexes::getIndexAfter(const Instruction& instr) const {
    auto It = instr2iMap.find(&instr);
    assert(It != instr2iMap.end() && "Instruction not found");

    SlotIndex InstrIdx = It->second;
    return SlotIndex(InstrIdx.listEntry(), SlotIndex::Slot_Dead);
}

const std::pair<SlotIndex, SlotIndex>& SlotIndexes::getBBRange(
    const BasicBlock* BB) const {
    return getBBRange(BB->getLabel());
}


BasicBlock* SlotIndexes::getBBFromIndex(const SlotIndex& index) const {
    auto It = getBBUpperBound(index);
    if (It != BBIndexBegin()) {
        --It;
        return It->second;
    }
    return nullptr;
}

SlotIndex SlotIndexes::insertInstrInMaps(Instruction& instr, bool Late) {
    assert(!hasIndex(instr) && "Instruction already has an index");

    // 找到合适的插入位置
    unsigned NewIndex = SlotIndex::InstrDist;  // 默认索引

    // 简化实现：在列表末尾添加
    IndexListEntry* Entry = createEntry(&instr, NewIndex);
    SlotIndex InstrIdx(Entry, SlotIndex::Slot_Block);
    instr2iMap[&instr] = InstrIdx;

    return InstrIdx;
}

void SlotIndexes::removeInstrFromMaps(Instruction& instr, bool AllowBundled) {
    auto It = instr2iMap.find(&instr);
    if (It != instr2iMap.end()) {
        SlotIndex InstrIdx = It->second;
        instr2iMap.erase(It);

        // 从索引列表中移除对应条目
        indexList.remove_if([&instr](const IndexListEntry& Entry) {
            return Entry.getInstr() == &instr;
        });
    }
}

void SlotIndexes::removeSingleInstrFromMaps(Instruction& instr) {
    removeInstrFromMaps(instr, false);
}

SlotIndex SlotIndexes::replaceInstrInMaps(Instruction& oldInstr,
                                          Instruction& newInstr) {
    auto It = instr2iMap.find(&oldInstr);
    assert(It != instr2iMap.end() && "Old instruction not found");

    SlotIndex OldIdx = It->second;
    instr2iMap.erase(It);

    // 更新IndexListEntry
    OldIdx.listEntry()->setInstr(&newInstr);
    instr2iMap[&newInstr] = OldIdx;

    return OldIdx;
}

void SlotIndexes::insertBBInMaps(BasicBlock* bb) {
    // 为新基本块分配索引范围
    unsigned NewIndex =
        indexList.empty() ? 0
                          : indexList.back().getIndex() + SlotIndex::InstrDist;

    SlotIndex StartIdx =
        SlotIndex(createEntry(nullptr, NewIndex), SlotIndex::Slot_Block);
    SlotIndex EndIdx =
        SlotIndex(createEntry(nullptr, NewIndex + SlotIndex::InstrDist),
                  SlotIndex::Slot_Block);

    // 使用标签更新 BBRanges
    std::string BBLabel = bb->getLabel();
    BBRanges[BBLabel] = std::make_pair(StartIdx, EndIdx);

    // 将 (SlotIndex, BasicBlock*) 对添加到 vector
    idx2BBMap.emplace_back(StartIdx, bb);

    // 重新排序以保持按 SlotIndex 升序
    std::sort(idx2BBMap.begin(), idx2BBMap.end(),
              [](const std::pair<SlotIndex, BasicBlock*>& A,
                 const std::pair<SlotIndex, BasicBlock*>& B) {
                  return A.first < B.first;
              });
}


void SlotIndexes::packIndexes() {
    unsigned CurIndex = 0;
    for (auto& Entry : indexList) {
        Entry.setIndex(CurIndex);
        CurIndex += SlotIndex::InstrDist;
    }

    // 更新所有映射
    for (auto& Pair : instr2iMap) {
        // 索引已在Entry中更新，SlotIndex会自动反映新值
    }
}

}  // namespace riscv64