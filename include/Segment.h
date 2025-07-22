#pragma once

#include <vector>

#include "GlobalVariable.h"

namespace riscv64 {

// 段的类型
enum class SegmentKind {
    DATA,   // .data
    BSS,    // .bss
    RODATA  // .rodata
};

class DataSegment {
   public:
    DataSegment(SegmentKind kind) : kind_(kind) {}

    void addItem(GlobalVariable var) { items_.push_back(std::move(var)); }

    std::string toString() const;

   private:
    SegmentKind kind_;
    std::vector<GlobalVariable> items_;
};

}  // namespace riscv64
