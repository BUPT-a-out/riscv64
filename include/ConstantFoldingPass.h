#include "Instructions/Function.h"


namespace riscv64 {
class ConstantFolding {
   public:
    ConstantFolding() = default;

    void runOnFunction(Function* function);
};

}  // namespace riscv64