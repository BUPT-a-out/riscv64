#pragma once

#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Instructions/Function.h"
#include "Instructions/MachineOperand.h"

namespace riscv64 {

// 栈帧中的对象类型
enum class StackObjectType {
    SpilledRegister,     // 溢出的寄存器
    AllocatedStackSlot,  // 分配的栈槽
};

// 栈帧对象
struct StackObject {
    StackObjectType type;
    int size;         // 字节数
    int alignment;    // 对齐要求
    unsigned regNum;  // 对于溢出寄存器，记录寄存器编号
    int identifier;   // 用于标识对象的唯一ID

    // 修正构造函数以确保所有成员都被初始化
    StackObject(StackObjectType t, int s, int a = 8, unsigned reg = 0)
        : type(t), size(s), alignment(a), regNum(reg), identifier(0) {}

    // 修正第二个构造函数
    StackObject(int s, int a = 8, int id = 0)
        : type(StackObjectType::AllocatedStackSlot),
          size(s),
          alignment(a),
          regNum(0),
          identifier(id) {}
};

// 栈帧管理器: 这个类只管理alloca的变量和spill的变量
class StackFrameManager {
   private:
    Function* function;

    std::vector<std::unique_ptr<StackObject>> stackObjects;

    std::unordered_map<const midend::Value*, int> allocaToStackSlot;

   public:
    explicit StackFrameManager(Function* func) : function(func) {}

    StackObject* getStackObject(int slotIndex) const;
    const std::vector<std::unique_ptr<StackObject>>& getAllStackObjects()
        const {
        return stackObjects;
    }

    // alloca 对象管理

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

    // 检查是否有任何栈对象需要处理
    bool hasStackObjects() const { return !stackObjects.empty(); }

    // 获取特定类型的栈对象数量
    int getStackObjectCount(StackObjectType type) const {
        int count = 0;
        for (const auto& obj : stackObjects) {
            if (obj->type == type) {
                count++;
            }
        }
        return count;
    }

    int createAllocaObject(const midend::Instruction* inst, int typeSize) {
        int fi_id = getNewStackObjectIdentifier();
        auto stack_obj = std::make_unique<StackObject>(
            StackObjectType::AllocatedStackSlot, static_cast<int>(typeSize),
            8,  // 8字节对齐
            0   // 第一阶段不设置regNum
        );
        stack_obj->identifier = fi_id;

        // 添加到栈帧管理器，但不计算偏移
        addStackObject(std::move(stack_obj));
        mapAllocaToStackSlot(inst, fi_id);

        std::cout << "Created abstract Frame Index FI(" << fi_id
                  << ") for alloca, size: " << typeSize << " bytes"
                  << std::endl;
        return fi_id;
    }

    int createSpillObject(unsigned reg) {
        int fi_id = getNewStackObjectIdentifier();
        auto spill_obj =
            std::make_unique<StackObject>(StackObjectType::SpilledRegister,
                                          8,  // 寄存器大小固定为8字节
                                          8,  // 8字节对齐
                                          reg  // 记录原始寄存器编号
            );
        spill_obj->identifier = fi_id;
        addStackObject(std::move(spill_obj));

        std::cout << "Created spill Frame Index FI(" << fi_id
                  << ") for register " << reg
                  << " [nextId: " << nextFrameIndexId << "]" << std::endl;
        return fi_id;
    }

   private:
    int nextFrameIndexId = 0;  // 每个函数实例独立的Frame Index计数器

    int getNewStackObjectIdentifier() { return nextFrameIndexId++; }
};

}  // namespace riscv64