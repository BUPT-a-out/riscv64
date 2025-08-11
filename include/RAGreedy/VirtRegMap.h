#pragma once

#include <cassert>
#include <unordered_map>
#include <vector>

#include "Instructions/All.h"

namespace riscv64 {

// 寄存器和栈槽的类型定义
using StackSlot = int;
using Register = unsigned;

// 特殊值定义
static const Register NO_PHYS_REG = 0;
static const StackSlot NO_STACK_SLOT = -1;

class VirtRegMap {
   public:
    // 构造函数
    explicit VirtRegMap(Function& function);

    // 析构函数
    ~VirtRegMap() = default;

    // 删除拷贝构造和赋值
    VirtRegMap(const VirtRegMap&) = delete;
    VirtRegMap& operator=(const VirtRegMap&) = delete;

    // =============== 虚拟寄存器 → 物理寄存器映射 ===============

    /// 将虚拟寄存器分配给物理寄存器
    void assignVirt2Phys(Register virtReg, Register physReg);

    /// 获取虚拟寄存器对应的物理寄存器
    Register getPhys(Register virtReg) const;

    /// 检查虚拟寄存器是否有物理寄存器分配
    bool hasPhys(Register virtReg) const;

    /// 清除虚拟寄存器的物理寄存器分配
    void clearVirt(Register virtReg);

    // =============== 虚拟寄存器 → 栈槽映射 ===============

    /// 将虚拟寄存器分配给栈槽
    void assignVirt2StackSlot(Register virtReg, StackSlot stackSlot);

    /// 获取虚拟寄存器对应的栈槽
    StackSlot getStackSlot(Register virtReg) const;

    /// 检查虚拟寄存器是否有栈槽分配
    bool hasStackSlot(Register virtReg) const;

    /// 清除虚拟寄存器的栈槽分配
    void clearStackSlot(Register virtReg);

    // =============== 虚拟寄存器 → 分割后虚拟寄存器映射 ===============

    /// 记录虚拟寄存器分割映射
    void assignVirt2Split(Register origVirtReg,
                          const std::vector<Register>& splitRegs);

    /// 获取虚拟寄存器分割后的寄存器列表
    const std::vector<Register>& getSplitRegs(Register origVirtReg) const;

    /// 检查虚拟寄存器是否被分割
    bool hasSplit(Register origVirtReg) const;

    /// 清除虚拟寄存器的分割映射
    void clearSplit(Register origVirtReg);

    // =============== 辅助方法 ===============

    /// 检查虚拟寄存器是否已被分配（物理寄存器或栈槽）
    bool isAssigned(Register virtReg) const;

    /// 获取关联的函数
    Function& getFunction() const { return function_; }

    /// 清除所有映射
    void clear();

    /// 打印调试信息
    void dump() const;

   private:
    // 关联的函数
    Function& function_;

    // 虚拟寄存器 → 物理寄存器映射
    std::unordered_map<Register, Register> virt2PhysMap_;

    // 虚拟寄存器 → 栈槽映射
    std::unordered_map<Register, StackSlot> virt2StackSlotMap_;

    // 虚拟寄存器 → 分割后虚拟寄存器列表映射
    std::unordered_map<Register, std::vector<Register>> virt2SplitMap_;

    // 空的分割寄存器列表，用于返回引用
    static const std::vector<Register> emptySplitRegs_;

    // 验证虚拟寄存器的有效性
    void validateVirtReg(Register virtReg) const;

    // 验证物理寄存器的有效性
    void validatePhysReg(Register physReg) const;
};
}  // namespace riscv64
