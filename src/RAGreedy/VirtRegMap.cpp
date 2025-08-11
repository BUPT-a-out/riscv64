#include "RAGreedy/VirtRegMap.h"

#include <iostream>

namespace riscv64 {

// 静态成员初始化
const std::vector<Register> VirtRegMap::emptySplitRegs_;

VirtRegMap::VirtRegMap(Function& function) : function_(function) {
    // 初始化映射表
    virt2PhysMap_.clear();
    virt2StackSlotMap_.clear();
    virt2SplitMap_.clear();
}

// =============== 虚拟寄存器 → 物理寄存器映射 ===============

void VirtRegMap::assignVirt2Phys(Register virtReg, Register physReg) {
    validateVirtReg(virtReg);
    validatePhysReg(physReg);

    // 检查是否已经分配给栈槽
    if (hasStackSlot(virtReg)) {
        // 可以选择清除栈槽分配或报错
        // 这里选择清除栈槽分配
        clearStackSlot(virtReg);
    }

    virt2PhysMap_[virtReg] = physReg;
}

Register VirtRegMap::getPhys(Register virtReg) const {
    validateVirtReg(virtReg);

    auto it = virt2PhysMap_.find(virtReg);
    if (it != virt2PhysMap_.end()) {
        return it->second;
    }
    return NO_PHYS_REG;
}

bool VirtRegMap::hasPhys(Register virtReg) const {
    validateVirtReg(virtReg);
    return virt2PhysMap_.find(virtReg) != virt2PhysMap_.end();
}

void VirtRegMap::clearVirt(Register virtReg) {
    validateVirtReg(virtReg);
    virt2PhysMap_.erase(virtReg);
}

// =============== 虚拟寄存器 → 栈槽映射 ===============

void VirtRegMap::assignVirt2StackSlot(Register virtReg, StackSlot stackSlot) {
    validateVirtReg(virtReg);
    assert(stackSlot != NO_STACK_SLOT && "Invalid stack slot");

    // 检查是否已经分配给物理寄存器
    if (hasPhys(virtReg)) {
        // 可以选择清除物理寄存器分配或报错
        // 这里选择清除物理寄存器分配
        clearVirt(virtReg);
    }

    virt2StackSlotMap_[virtReg] = stackSlot;
}

StackSlot VirtRegMap::getStackSlot(Register virtReg) const {
    validateVirtReg(virtReg);

    auto it = virt2StackSlotMap_.find(virtReg);
    if (it != virt2StackSlotMap_.end()) {
        return it->second;
    }
    return NO_STACK_SLOT;
}

bool VirtRegMap::hasStackSlot(Register virtReg) const {
    validateVirtReg(virtReg);
    return virt2StackSlotMap_.find(virtReg) != virt2StackSlotMap_.end();
}

void VirtRegMap::clearStackSlot(Register virtReg) {
    validateVirtReg(virtReg);
    virt2StackSlotMap_.erase(virtReg);
}

// =============== 虚拟寄存器 → 分割后虚拟寄存器映射 ===============

void VirtRegMap::assignVirt2Split(Register origVirtReg,
                                  const std::vector<Register>& splitRegs) {
    validateVirtReg(origVirtReg);
    assert(!splitRegs.empty() && "Split register list cannot be empty");

    // 验证所有分割后的寄存器
    for (Register splitReg : splitRegs) {
        validateVirtReg(splitReg);
    }

    virt2SplitMap_[origVirtReg] = splitRegs;
}

const std::vector<Register>& VirtRegMap::getSplitRegs(
    Register origVirtReg) const {
    validateVirtReg(origVirtReg);

    auto it = virt2SplitMap_.find(origVirtReg);
    if (it != virt2SplitMap_.end()) {
        return it->second;
    }
    return emptySplitRegs_;
}

bool VirtRegMap::hasSplit(Register origVirtReg) const {
    validateVirtReg(origVirtReg);
    return virt2SplitMap_.find(origVirtReg) != virt2SplitMap_.end();
}

void VirtRegMap::clearSplit(Register origVirtReg) {
    validateVirtReg(origVirtReg);
    virt2SplitMap_.erase(origVirtReg);
}

// =============== 辅助方法 ===============

bool VirtRegMap::isAssigned(Register virtReg) const {
    return hasPhys(virtReg) || hasStackSlot(virtReg);
}

void VirtRegMap::clear() {
    virt2PhysMap_.clear();
    virt2StackSlotMap_.clear();
    virt2SplitMap_.clear();
}

void VirtRegMap::dump() const {
    std::cout << "=== VirtRegMap Dump ===" << std::endl;

    // 打印虚拟寄存器 → 物理寄存器映射
    std::cout << "Virt2Phys Mapping:" << std::endl;
    for (const auto& pair : virt2PhysMap_) {
        std::cout << "  V" << pair.first << " -> P" << pair.second << std::endl;
    }

    // 打印虚拟寄存器 → 栈槽映射
    std::cout << "Virt2StackSlot Mapping:" << std::endl;
    for (const auto& pair : virt2StackSlotMap_) {
        std::cout << "  V" << pair.first << " -> S" << pair.second << std::endl;
    }

    // 打印虚拟寄存器 → 分割映射
    std::cout << "Virt2Split Mapping:" << std::endl;
    for (const auto& pair : virt2SplitMap_) {
        std::cout << "  V" << pair.first << " -> [";
        for (size_t i = 0; i < pair.second.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << "V" << pair.second[i];
        }
        std::cout << "]" << std::endl;
    }

    std::cout << "======================" << std::endl;
}

void VirtRegMap::validateVirtReg(Register virtReg) const {
    assert(virtReg >= 64 && "Invalid virtual reg");
}

void VirtRegMap::validatePhysReg(Register physReg) const {
    assert(physReg < 64 && "Invalid physical reg");
}
}  // namespace riscv64
