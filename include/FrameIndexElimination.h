#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "Instructions/Function.h"
#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"
#include "StackFrameManager.h"

namespace riscv64 {

// 第三阶段：Frame Index Elimination Pass
// 负责计算最终栈帧布局并消除所有Frame Index伪指令
class FrameIndexElimination {
   private:
    Function* function;
    StackFrameManager* stackManager;

    // 最终的栈帧布局信息
    struct FinalFrameLayout {
        int totalFrameSize = 0;
        int returnAddressOffset = 0;      // ra相对于sp的偏移
        int framePointerOffset = 0;       // s0相对于sp的偏移

        // FI到最终偏移的映射(相对于s0/fp)
        std::unordered_map<int, int> frameIndexToOffset;
    };

    FinalFrameLayout layout;

   public:
    explicit FrameIndexElimination(Function* func)
        : function(func), stackManager(func->getStackFrameManager()) {}

    // 主要接口
    void run();

   private:
    // 第三阶段：计算最终栈帧布局
    void computeFinalFrameLayout();

    // 为所有栈对象分配最终偏移
    void assignFinalOffsets();

    // 生成最终的序言和尾声代码
    void generateFinalPrologueEpilogue();

    // 消除所有frameaddr伪指令
    void eliminateFrameIndices();

    // 替换单个frameaddr指令
    void eliminateFrameIndexInstruction(
        BasicBlock* bb, std::list<std::unique_ptr<Instruction>>::iterator& it);

    // 辅助函数
    int alignTo(int value, int alignment) const;
    void printFinalLayout() const;

    // 计算保存寄存器所需的空间
    int calculateSavedRegisterSize();

    // 计算最大调用参数所需的栈空间
    int calculateMaxCallArgSize();

    // 收集需要保存的寄存器列表
    std::vector<int> collectSavedIntegerRegisters();
    std::vector<int> collectSavedFloatRegisters();

    // 处理大偏移量的辅助函数
    bool isValidImmediateOffset(int64_t offset) const;
    void generateAddWithLargeOffset(
        BasicBlock* bb, std::list<std::unique_ptr<Instruction>>::iterator& it,
        int destRegNum, bool destIsVirtual, int baseRegNum, bool baseIsVirtual,
        int64_t offset);
};

}  // namespace riscv64