#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <memory>
#include <algorithm>

#include "Instructions/Function.h"
#include "Instructions/Instruction.h"
#include "Instructions/MachineOperand.h"
#include "ABI.h"

namespace riscv64 {

// 栈帧中的对象类型
enum class StackObjectType {
    LocalVariable,      // 局部变量
    SpilledRegister,    // 溢出的寄存器
    SavedRegister,      // 保存的callee-saved寄存器
    Temporary,          // 临时变量
    Parameter,          // 参数（超过8个时）
    ReturnAddress,      // 返回地址
    AllocatedStackSlot, // 分配的栈槽
};

// 栈帧对象
struct StackObject {
    StackObjectType type;
    int size;           // 字节数
    int alignment;      // 对齐要求
    int offset;         // 相对于栈指针的偏移量
    unsigned regNum;    // 对于溢出寄存器，记录寄存器编号
    bool isFixed;       // 是否是固定位置的对象
    int identifier;     // 用于标识对象的唯一ID
    
    StackObject(StackObjectType t, int s, int a = 8, unsigned reg = 0)
        : type(t), size(s), alignment(a), offset(0), regNum(reg), isFixed(false) {}

    StackObject(int s, int a = 8,
                int id = 0)  // 这个用于记录 allocated stack slots
        : type(StackObjectType::AllocatedStackSlot),
          size(s),
          alignment(a),
          offset(0),
          regNum(0),
          isFixed(false),
          identifier(id) {}
};

// 栈帧布局信息
struct StackFrameInfo {
    int totalSize = 0;              // 栈帧总大小
    int localVariableSize = 0;      // 局部变量区大小
    int spilledRegisterSize = 0;    // 溢出寄存器区大小
    int savedRegisterSize = 0;      // 保存寄存器区大小
    int parameterSize = 0;          // 参数区大小
    int maxCalleeArgs = 0;          // 最大被调用函数参数数量
    bool hasStackPointerAdjustment = false;  // 是否需要调整栈指针
    bool hasVariableArgs = false;   // 是否有可变参数
    
    // 各区域在栈中的偏移量
    int localVariableOffset = 0;
    int spilledRegisterOffset = 0;
    int savedRegisterOffset = 0;
    int parameterOffset = 0;
    int returnAddressOffset = 0;
};

// 栈帧管理器
class StackFrameManager {
private:
    Function* function;
    StackFrameInfo frameInfo;
    std::vector<std::unique_ptr<StackObject>> stackObjects;
    std::unordered_map<unsigned, int> spilledRegToStackSlot;  // 溢出寄存器到栈槽的映射
    std::unordered_map<unsigned, int> localVarToStackSlot;    // 局部变量到栈槽的映射
    std::unordered_map<const midend::Value*, int> allocaToStackSlot;// alloca指令到栈槽的映射
    std::unordered_set<unsigned> calleeSavedRegs;             // 需要保存的callee-saved寄存器
    
    // RISC-V ABI定义的寄存器使用约定
    static const std::vector<unsigned> callerSavedRegs;
    static const std::vector<unsigned> calleeSavedRegs_static;
    static const std::vector<unsigned> argumentRegs;
    static const std::vector<unsigned> returnValueRegs;
    
public:
    explicit StackFrameManager(Function* func) : function(func) {}
    
    // 主要接口
    void computeStackFrame();
    void insertPrologueEpilogue();
    
    // 栈对象管理
    int allocateStackSlot(StackObjectType type, int size, int alignment = 8, unsigned regNum = 0);
    int getStackSlotOffset(int slotIndex) const;
    StackObject* getStackObject(int slotIndex) const;

    // alloca 对象管理
    int getNewStackObjectIdentifier() {
        static int nextId = 0;
        return nextId++;
    }
    void addStackObject(std::unique_ptr<StackObject> obj) {
        stackObjects.push_back(std::move(obj));
    }
    StackObject* getStackObjectByIdentifier(int id) const {
        auto it = std::find_if(stackObjects.begin(), stackObjects.end(),
                               [&id](const std::unique_ptr<StackObject>& obj) {
                                   return obj->identifier == id;
                               });
        return it != stackObjects.end() ? it->get() : nullptr;
    }
    void mapAllocaToStackSlot(const midend::Value* alloca, int slotIndex) {
        allocaToStackSlot[alloca] = slotIndex;
    }
    int getAllocaStackSlotId(const midend::Value* alloca) const {
        auto it = allocaToStackSlot.find(alloca);
        return it != allocaToStackSlot.end() ? it->second : -1;
    }

    // 溢出寄存器管理
    int allocateSpillSlot(unsigned regNum);
    int getSpillSlotOffset(unsigned regNum) const;
    bool isRegisterSpilled(unsigned regNum) const;
    
    // 局部变量管理
    int allocateLocalVariable(int size, int alignment = 8);
    
    // 寄存器保存/恢复
    void markRegisterForSaving(unsigned regNum);
    bool needsRegisterSaving(unsigned regNum) const;
    
    // 栈帧信息查询
    const StackFrameInfo& getFrameInfo() const { return frameInfo; }
    int getFrameSize() const { return frameInfo.totalSize; }
    bool hasStackFrame() const { return frameInfo.totalSize > 0; }
    
    // 调试和输出
    void printStackLayout() const;
    std::string toString() const;
    
private:
    // 内部计算函数
    void analyzeRegisterUsage();
    void calculateStackLayout();
    void assignStackOffsets();
    
    // 序言和尾声生成
    void generatePrologue();
    void generateEpilogue();
    
    // 辅助函数
    bool isCallerSaved(unsigned regNum) const;
    bool isCalleeSaved(unsigned regNum) const;
    bool isArgumentRegister(unsigned regNum) const;
    bool isReturnRegister(unsigned regNum) const;
    int alignTo(int value, int alignment) const;
    
    // 指令生成辅助函数
    void insertInstructionAtBeginning(BasicBlock* bb, std::unique_ptr<Instruction> inst);
    void insertInstructionAtEnd(BasicBlock* bb, std::unique_ptr<Instruction> inst);
    std::unique_ptr<Instruction> createStackAdjustInstruction(int offset);
    std::unique_ptr<Instruction> createSaveInstruction(unsigned regNum, int offset);
    std::unique_ptr<Instruction> createRestoreInstruction(unsigned regNum, int offset);
};

}