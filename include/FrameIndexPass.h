#pragma once

#include <unordered_map>
#include <memory>
#include <vector>

#include "Instructions/Function.h"
#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"
#include "StackFrameManager.h"

namespace riscv64 {

// 栈帧布局Pass - 将frameaddr伪指令展开为具体的地址计算
class FrameIndexPass {
private:
    Function* function;
    StackFrameManager* stackManager;
    
    // 栈帧布局信息
    struct FrameLayout {
        int totalFrameSize = 0;              // 总栈帧大小
        int localVariableAreaOffset = 0;     // 局部变量区起始偏移(相对于fp)
        int savedRegisterAreaOffset = 0;     // 保存寄存器区偏移(相对于sp)
        int returnAddressOffset = 0;         // 返回地址偏移(相对于sp)
        int framePointerOffset = 0;          // 帧指针偏移(相对于sp)
        
        // FI到实际偏移的映射(相对于fp)
        std::unordered_map<int, int> frameIndexToOffset;
    };
    
    FrameLayout layout;
    
public:
    explicit FrameIndexPass(Function* func) 
        : function(func), stackManager(func->getStackFrameManager()) {}
    
    // 主要接口
    void run();
    
private:
    // 计算栈帧布局
    void computeFrameLayout();
    
    // 计算栈帧总大小和各区域偏移
    void calculateFrameOffsets();
    
    // 生成序言代码
    void generatePrologue();
    
    // 生成尾声代码  
    void generateEpilogue();
    
    // 展开frameaddr伪指令
    void expandFrameAddressInstructions();
    
    // 替换单个frameaddr指令
    void expandFrameAddressInstruction(BasicBlock* bb, 
                                     std::list<std::unique_ptr<Instruction>>::iterator& it);
    
    // 在基本块开头插入指令
    void insertInstructionAtBeginning(BasicBlock* bb, std::unique_ptr<Instruction> inst);
    
    // 在基本块结尾插入指令(ret指令之前)
    void insertInstructionBeforeRet(BasicBlock* bb, std::unique_ptr<Instruction> inst);
    
    // 辅助函数
    int alignTo(int value, int alignment) const;
    void printFrameLayout() const;
};

}
