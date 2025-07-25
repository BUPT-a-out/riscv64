#include "StackFrameManager.h"

namespace riscv64 {

StackObject* StackFrameManager::getStackObject(int slotIndex) const {
    if (slotIndex >= 0 && slotIndex < static_cast<int>(stackObjects.size())) {
        return stackObjects[slotIndex].get();
    }
    return nullptr;
}

}  // namespace riscv64