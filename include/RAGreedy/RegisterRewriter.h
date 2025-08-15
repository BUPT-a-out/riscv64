#pragma once

#include <unordered_map>
#include <vector>

#include "ABI.h"
#include "Instructions/All.h"
#include "RAGreedy/VirtRegMap.h"

namespace riscv64 {

class RegisterRewriter {
   private:
    Function* function;
    VirtRegMap* VRM;
    bool assigningFloat = false;

    // 跟踪需要的栈槽偏移
    std::unordered_map<StackSlot, int> stackSlot2FrameIndex;
    int currentStackOffset;

    // 生成spill/reload代码的辅助函数
    void insertSpillCode(BasicBlock* BB, BasicBlock::iterator insertPos,
                         unsigned virtReg, unsigned physReg);
    void insertReloadCode(BasicBlock* BB, BasicBlock::iterator insertPos,
                          unsigned virtReg, unsigned physReg);

    // 处理不同类型的指令
    void rewriteInstruction(Instruction* inst);
    void rewriteOperand(std::unique_ptr<MachineOperand>& operand,
                        Instruction* inst, bool isDef,
                        std::vector<Instruction*>& newInstructions);

    // 在头文件中的新方法签名
    std::vector<std::unique_ptr<Instruction>> createLoadInstructions(
        unsigned physReg, StackSlot slot);
    std::vector<std::unique_ptr<Instruction>> createStoreInstructions(
        unsigned physReg, StackSlot slot);

    unsigned allocateTemporaryRegister(unsigned virtReg);

    // 寄存器类型判断
    void updateRegisterInInstruction(Instruction* inst, unsigned oldReg,
                                     unsigned newReg, bool isFloat);
    unsigned selectAvailablePhysicalDataReg(Instruction* inst, bool isFloat);

   public:
    RegisterRewriter(Function* func, VirtRegMap* vrm, bool assigningFloat = false)
        : function(func), VRM(vrm), assigningFloat(assigningFloat), currentStackOffset(0) {}

    void rewrite();
    void mapStackSlotToFrameIndex();
    void print(std::ostream& OS) const;
};

}  // namespace riscv64