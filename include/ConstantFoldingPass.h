#include <map>

#include "Instructions/All.h"

namespace riscv64 {

class ConstantFolding {
   public:
    ConstantFolding() = default;

    void runOnFunction(Function* function);
    void runOnBasicBlock(BasicBlock* basicBlock);
    void handleInstruction(Instruction* inst, BasicBlock* parent_bb);

    // 尝试折叠
    void foldInstruction(Instruction* inst, BasicBlock* parent_bb);

    // 尝试窥孔优化
    void peepholeOptimize(Instruction* inst, BasicBlock* parent_bb);
    // 模式匹配
    void foldToITypeInst(Instruction* inst, BasicBlock* parent_bb);
    void algebraicIdentitySimplify(Instruction* inst, BasicBlock* parent_bb);
    void strengthReduction(Instruction* inst, BasicBlock* parent_bb);
    void bitwiseOperationSimplify(Instruction* inst, BasicBlock* parent_bb);
    void instructionReassociateAndCombine(Instruction* inst,
                                          BasicBlock* parent_bb);

    // 尝试常量传播
    void constantPropagate(Instruction* inst, BasicBlock* parent_bb);

    // 工具函数

    // 计算常数指令的值
    std::optional<int64_t> calculateInstructionValue(
        Opcode op, std::vector<int64_t>& source_operands);

   private:
    // 将虚拟寄存器映射到一个已知的常量值
    std::map<unsigned int, int64_t> virtualRegisterConstants;
    std::vector<Instruction*> instructionsToRemove;
};

}  // namespace riscv64