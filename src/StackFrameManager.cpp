#include "StackFrameManager.h"


namespace riscv64 {

// 栈对象管理函数
int StackFrameManager::allocateStackSlot(StackObjectType type, int size,
                                         int alignment, unsigned regNum) {
    auto obj = std::make_unique<StackObject>(type, size, alignment, regNum);
    int index = stackObjects.size();
    stackObjects.push_back(std::move(obj));

    return index;
}

int StackFrameManager::allocateSpillSlot(unsigned regNum) {
    if (spilledRegToStackSlot.find(regNum) != spilledRegToStackSlot.end()) {
        return spilledRegToStackSlot[regNum];
    }

    return allocateStackSlot(StackObjectType::SpilledRegister, 8, 8, regNum);
}


StackObject* StackFrameManager::getStackObject(int slotIndex) const {
    if (slotIndex >= 0 && slotIndex < static_cast<int>(stackObjects.size())) {
        return stackObjects[slotIndex].get();
    }
    return nullptr;
}

}  // namespace riscv64