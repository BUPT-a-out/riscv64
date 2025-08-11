#include "RAGreedy/LiveRegMatrix.h"
#include <algorithm>

namespace riscv64 {

void LiveRegMatrix::releaseMemory() {
    Matrix.clear();
    Queries.clear();
    LIS = nullptr;
    VRM = nullptr;
    UserTag = 0;
}

void LiveRegMatrix::init(Function &F, LiveIntervals &liveIntervals,
                         VirtRegMap &virtRegMap) {
    LIS = &liveIntervals;
    VRM = &virtRegMap;
    
    // TODO: float
    const unsigned NumPhysRegs = 32; // RISC-V有32个通用寄存器
    
    Matrix.clear();
    Matrix.resize(NumPhysRegs);
    
    Queries.clear();
    Queries.resize(NumPhysRegs);
    for (unsigned i = 0; i < NumPhysRegs; ++i) {
        Queries[i] = std::make_unique<LiveIntervalUnion::Query>();
    }
    
    UserTag = 1;
}

LiveRegMatrix::InterferenceKind 
LiveRegMatrix::checkInterference(const LiveInterval &VirtReg,
                                 const RegisterOperand &PhysReg) {
    if (PhysReg.getRegNum() >= Matrix.size()) {
        return IK_Free;
    }
    
    LiveIntervalUnion::Query &Q = query(VirtReg, PhysReg.getRegNum());
    
    if (Q.checkInterference()) {
        return IK_VirtReg;
    }
    
    return IK_Free;
}

bool LiveRegMatrix::checkInterference(SlotIndex Start, SlotIndex End,
                                      const RegisterOperand &PhysReg) {
    if (PhysReg.getRegNum() >= Matrix.size()) {
        return false;
    }
    
    // 创建临时的LiveInterval来表示这个区间
    // 这里简化实现，实际可能需要更复杂的逻辑
    const LiveIntervalUnion &LIU = Matrix[PhysReg.getRegNum()];
    
    // 检查是否与任何已存在的区间重叠
    for (auto it = LIU.begin(); it != LIU.end(); ++it) {
        if (it->first < End && Start < it->first) {
            return true; // 有重叠
        }
    }
    
    return false;
}

void LiveRegMatrix::assign(const LiveInterval &VirtReg, 
                           const RegisterOperand &PhysReg) {
    if (PhysReg.getRegNum() >= Matrix.size()) {
        return;
    }
    
    // 更新VirtRegMap
    VRM->assignVirt2Phys(VirtReg.reg().getRegNum(), PhysReg.getRegNum());
    
    // 将虚拟寄存器添加到对应物理寄存器的联合中
    auto VirtRegPtr = std::make_shared<const LiveInterval>(VirtReg);
    Matrix[PhysReg.getRegNum()].unify(VirtRegPtr, VirtReg);
    
    // 使查询缓存失效
    invalidateVirtRegs();
}

void LiveRegMatrix::unassign(const LiveInterval &VirtReg) {
    unsigned PhysReg = VRM->getPhys(VirtReg.reg().getRegNum());
    
    if (PhysReg >= Matrix.size()) {
        return;
    }
    
    // 从物理寄存器的联合中移除虚拟寄存器
    auto VirtRegPtr = std::make_shared<const LiveInterval>(VirtReg);
    Matrix[PhysReg].extract(VirtRegPtr, VirtReg);
    
    // 更新VirtRegMap
    VRM->clearVirt(VirtReg.reg().getRegNum());
    
    // 使查询缓存失效
    invalidateVirtRegs();
}

bool LiveRegMatrix::isPhysRegUsed(const RegisterOperand &PhysReg) const {
    if (PhysReg.getRegNum() >= Matrix.size()) {
        return false;
    }
    
    return !Matrix[PhysReg.getRegNum()].empty();
}

LiveIntervalUnion::Query &LiveRegMatrix::query(const LiveInterval &LI, 
                                               unsigned RegIndex) {
    if (RegIndex >= Queries.size()) {
        // 扩展查询数组
        Queries.resize(RegIndex + 1);
        for (unsigned i = Queries.size() - (RegIndex + 1 - Queries.size()); 
             i <= RegIndex; ++i) {
            if (!Queries[i]) {
                Queries[i] = std::make_unique<LiveIntervalUnion::Query>();
            }
        }
    }
    
    if (RegIndex >= Matrix.size()) {
        Matrix.resize(RegIndex + 1);
    }
    
    LiveIntervalUnion::Query &Q = *Queries[RegIndex];
    Q.init(UserTag, LI, Matrix[RegIndex]);
    return Q;
}

RegisterOperand LiveRegMatrix::getOneVReg(unsigned PhysReg) const {
    if (PhysReg >= Matrix.size()) {
        return RegisterOperand(0); // 返回无效寄存器
    }
    
    auto VirtRegPtr = Matrix[PhysReg].getOneVReg();
    if (!VirtRegPtr) {
        return RegisterOperand(0); // 返回无效寄存器
    }
    
    return VirtRegPtr->reg();
}

} // namespace riscv64
