#pragma once

#include <memory>

#include "Instructions/Instruction.h"

namespace riscv64 {

class BasicBlock;  // 前向声明

// 魔数除法结构体
struct MagicConstants {
    int64_t magic;  // 魔数
    int shift;      // 移位数
    bool add_flag;  // 是否需要加法修正
};

// 魔数除法优化类
class MagicDivision {
   public:
    // 计算魔数常量
    static auto computeMagicConstants(int32_t divisor) -> MagicConstants;

    // 生成魔数除法指令序列
    // 返回结果寄存器的操作数
    static auto generateMagicDivision(
        std::unique_ptr<MachineOperand> dividend_operand, int32_t divisor,
        BasicBlock* parent_bb) -> std::unique_ptr<RegisterOperand>;

    // 生成魔数取模指令序列
    static auto generateMagicModulo(
        std::unique_ptr<MachineOperand> dividend_operand, int32_t divisor,
        BasicBlock* parent_bb) -> std::unique_ptr<RegisterOperand>;

    // 检查是否可以进行魔数除法优化
    static auto canOptimize(int32_t divisor) -> bool;

   private:
    // 辅助函数：将操作数转换为寄存器
    static auto ensureRegister(std::unique_ptr<MachineOperand> operand,
                               BasicBlock* parent_bb)
        -> std::unique_ptr<RegisterOperand>;

    // 生成立即数加载指令
    static auto generateLoadImmediate(int64_t value, BasicBlock* parent_bb)
        -> std::unique_ptr<RegisterOperand>;
};

}  // namespace riscv64
