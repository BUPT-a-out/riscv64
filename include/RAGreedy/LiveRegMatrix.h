#include <memory>
#include <unordered_map>
#include <vector>

#include "LiveIntervalUnion.h"
#include "LiveIntervals.h"
#include "VirtRegMap.h"

namespace riscv64 {

class LiveRegMatrix {
    LiveIntervals *LIS = nullptr;
    VirtRegMap *VRM = nullptr;

    unsigned UserTag = 0;

    // 使用 STL 容器替代原 LiveIntervalUnion::Array
    std::vector<LiveIntervalUnion> Matrix;

    // 对应每个寄存器索引的查询缓存
    std::vector<std::unique_ptr<LiveIntervalUnion::Query>> Queries;

    void releaseMemory();

   public:
    LiveRegMatrix() = default;
    LiveRegMatrix(LiveRegMatrix &&) = default;

    void init(Function &F, LiveIntervals &liveIntervals,
              VirtRegMap &virtRegMap);

    // 干扰类型
    enum InterferenceKind {
        IK_Free = 0,  // 无干扰
        IK_VirtReg    // 虚拟寄存器干扰
    };

    // 使缓存失效
    void invalidateVirtRegs() { ++UserTag; }

    // 检查在将虚拟寄存器分配给物理寄存器前是否有干扰
    InterferenceKind checkInterference(const LiveInterval &VirtReg,
                                       const RegisterOperand &PhysReg);

    // 检查区间[Start, End)内物理寄存器是否有干扰
    bool checkInterference(SlotIndex Start, SlotIndex End,
                           const RegisterOperand &PhysReg);

    // 将虚拟寄存器分配给物理寄存器
    void assign(const LiveInterval &VirtReg, const RegisterOperand &PhysReg);

    // 取消虚拟寄存器与物理寄存器的分配关联
    void unassign(const LiveInterval &VirtReg);

    // 判断物理寄存器是否已使用
    bool isPhysRegUsed(const RegisterOperand &PhysReg) const;

    // 访问底层 LiveIntervalUnion
    LiveIntervalUnion *getLiveUnions() { return Matrix.data(); }

    // 查询矩阵中指定物理寄存器的指定活跃区间查询
    LiveIntervalUnion::Query &query(const LiveInterval &LI, unsigned RegIndex);

    // 获取虚拟寄存器对应的物理寄存器
    RegisterOperand getOneVReg(unsigned PhysReg) const;
};

}  // namespace riscv64
