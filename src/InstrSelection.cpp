#include "CodeGen.h"
#include "IR/Instructions.h"

namespace riscv64 {

std::string CodeGenerator::generateInstruction(const midend::Instruction* inst) {
    if (inst->isUnaryOp()) {
        // TODO(rikka): ...
        return "";
    }

    return "# Not implemented instr: " + std::to_string(static_cast<int>(inst->getOpcode()));
}

}  // namespace riscv64