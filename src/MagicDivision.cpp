#include "MagicDivision.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "CodeGen.h"
#include "Instructions/All.h"

namespace riscv64 {

// 常量定义
constexpr int INITIAL_PRECISION = 31;
constexpr int SHIFT_32_BITS = 32;
constexpr int SIGN_BIT_SHIFT = 31;
constexpr int VIRTUAL_REG_START = 10000;
constexpr uint32_t SIGN_BIT_MASK = 0x80000000U;

auto MagicDivision::computeMagicConstants(int32_t divisor) -> MagicConstants {
    if (divisor == 0) {
        throw std::runtime_error("Division by zero");
    }

    // 处理特殊情况
    if (divisor == 1) {
        return {1, 0, false};
    }
    if (divisor == -1) {
        return {-1, 0, false};
    }

    // 检查是否为2的幂
    if (isPowerOfTwo(divisor)) {
        int shift_amount = getPowerOfTwoShift(divisor);
        return {1, shift_amount, false};  // 对于2的幂，使用简单的移位
    }

    // 使用原始算法计算魔数
    int precision = INITIAL_PRECISION;
    const uint64_t two31 = 0x80000000ULL;  // 2**31

    auto divisor_abs = static_cast<uint32_t>(divisor >= 0 ? divisor : -divisor);
    auto divisor_unsigned = static_cast<uint32_t>(divisor);
    uint64_t temp =
        two31 + ((divisor_unsigned & SIGN_BIT_MASK) >> SIGN_BIT_SHIFT);
    uint64_t anc = temp - 1 - temp % divisor_abs;
    uint64_t quotient1 = two31 / anc;
    uint64_t remainder1 = two31 - quotient1 * anc;
    uint64_t quotient2 = two31 / divisor_abs;
    uint64_t remainder2 = two31 - quotient2 * divisor_abs;

    do {
        precision++;
        quotient1 = 2 * quotient1;
        remainder1 = 2 * remainder1;
        if (remainder1 >= anc) {
            quotient1++;
            remainder1 = remainder1 - anc;
        }
        quotient2 *= 2;
        remainder2 *= 2;
        if (remainder2 >= divisor_abs) {
            quotient2 += 1;
            remainder2 -= divisor_abs;
        }
        uint64_t delta = divisor_abs - remainder2;
        if (quotient1 >= delta && !(quotient1 == delta && remainder1 == 0)) {
            break;
        }
    } while (true);

    auto magic = static_cast<int64_t>(quotient2 + 1);

    return {magic, precision - SHIFT_32_BITS, magic < 0};
}

auto MagicDivision::canOptimize(int32_t divisor) -> bool {
    // 只对非零常量进行优化，且避免除1和-1的情况（这些已经有更简单的优化）
    return divisor != 0 && divisor != 1 && divisor != -1;
}

auto MagicDivision::ensureRegister(std::unique_ptr<MachineOperand> operand,
                                   BasicBlock* parent_bb)
    -> std::unique_ptr<RegisterOperand> {
    if (operand->getType() == OperandType::Register) {
        auto* reg_op = dynamic_cast<RegisterOperand*>(operand.get());
        return std::make_unique<RegisterOperand>(reg_op->getRegNum(),
                                                 reg_op->isVirtual());
    }

    // 如果是立即数，需要先加载到寄存器
    if (operand->getType() == OperandType::Immediate) {
        auto* imm_op = dynamic_cast<ImmediateOperand*>(operand.get());
        return generateLoadImmediate(imm_op->getValue(), parent_bb);
    }

    throw std::runtime_error(
        "Unsupported operand type for register conversion");
}

auto MagicDivision::generateLoadImmediate(int64_t value, BasicBlock* parent_bb)
    -> std::unique_ptr<RegisterOperand> {
    // 使用简单的虚拟寄存器编号生成策略
    static int next_reg_num = VIRTUAL_REG_START;
    auto new_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);

    auto instruction = std::make_unique<Instruction>(Opcode::LI, parent_bb);
    instruction->addOperand_(std::make_unique<RegisterOperand>(
        new_reg->getRegNum(), new_reg->isVirtual()));
    instruction->addOperand_(std::make_unique<ImmediateOperand>(value));

    parent_bb->addInstruction(std::move(instruction));

    return new_reg;
}

auto MagicDivision::isPowerOfTwo(int32_t value) -> bool {
    if (value <= 0) {
        return false;
    }
    // 检查是否只有一个位被设置
    auto unsigned_value = static_cast<uint32_t>(value);
    return (unsigned_value & (unsigned_value - 1U)) == 0U;
}

auto MagicDivision::getPowerOfTwoShift(int32_t value) -> int {
    if (!isPowerOfTwo(value)) {
        throw std::runtime_error("Value is not a power of two");
    }

    int shift = 0;
    int temp = value;
    while (temp > 1) {
        temp = temp / 2;
        shift++;
    }
    return shift;
}

auto MagicDivision::generatePowerOfTwoDivision(
    std::unique_ptr<RegisterOperand> dividend_reg, int32_t divisor,
    BasicBlock* parent_bb) -> std::unique_ptr<RegisterOperand> {
    static int next_reg_num = VIRTUAL_REG_START;
    int shift_amount = getPowerOfTwoShift(divisor);

    // 对于有符号除法，需要处理负数的舍入
    // 算法：如果被除数为负，先加上 (divisor - 1)，然后再右移

    // 1. 获取符号位：srai sign_reg, dividend, 31
    auto sign_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto sign_instr = std::make_unique<Instruction>(Opcode::SRAI, parent_bb);
    sign_instr->addOperand_(std::make_unique<RegisterOperand>(
        sign_reg->getRegNum(), sign_reg->isVirtual()));
    sign_instr->addOperand_(std::make_unique<RegisterOperand>(
        dividend_reg->getRegNum(), dividend_reg->isVirtual()));
    sign_instr->addOperand_(std::make_unique<ImmediateOperand>(SIGN_BIT_SHIFT));
    parent_bb->addInstruction(std::move(sign_instr));

    // 2. 计算调整值：andi adjust_reg, sign_reg, (divisor - 1)
    auto adjust_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto andi_instr = std::make_unique<Instruction>(Opcode::ANDI, parent_bb);
    andi_instr->addOperand_(std::make_unique<RegisterOperand>(
        adjust_reg->getRegNum(), adjust_reg->isVirtual()));
    andi_instr->addOperand_(std::make_unique<RegisterOperand>(
        sign_reg->getRegNum(), sign_reg->isVirtual()));
    andi_instr->addOperand_(std::make_unique<ImmediateOperand>(divisor - 1));
    parent_bb->addInstruction(std::move(andi_instr));

    // 3. 加法调整：add temp_reg, dividend, adjust_reg
    auto temp_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto add_instr = std::make_unique<Instruction>(Opcode::ADDW, parent_bb);
    add_instr->addOperand_(std::make_unique<RegisterOperand>(
        temp_reg->getRegNum(), temp_reg->isVirtual()));
    add_instr->addOperand_(std::make_unique<RegisterOperand>(
        dividend_reg->getRegNum(), dividend_reg->isVirtual()));
    add_instr->addOperand_(std::make_unique<RegisterOperand>(
        adjust_reg->getRegNum(), adjust_reg->isVirtual()));
    parent_bb->addInstruction(std::move(add_instr));

    // 4. 算术右移：srai result_reg, temp_reg, shift_amount
    auto result_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto srai_instr = std::make_unique<Instruction>(Opcode::SRAI, parent_bb);
    srai_instr->addOperand_(std::make_unique<RegisterOperand>(
        result_reg->getRegNum(), result_reg->isVirtual()));
    srai_instr->addOperand_(std::make_unique<RegisterOperand>(
        temp_reg->getRegNum(), temp_reg->isVirtual()));
    srai_instr->addOperand_(std::make_unique<ImmediateOperand>(shift_amount));
    parent_bb->addInstruction(std::move(srai_instr));

    return result_reg;
}

auto MagicDivision::generatePowerOfTwoModulo(
    std::unique_ptr<RegisterOperand> dividend_reg, int32_t divisor,
    BasicBlock* parent_bb) -> std::unique_ptr<RegisterOperand> {
    // 对于2的幂的模运算，我们使用位掩码优化而不是除法+乘法
    // remainder = dividend & (divisor - 1)，但需要处理负数情况

    static int next_reg_num = VIRTUAL_REG_START;

    // 1. 获取符号位：srai sign_reg, dividend, 31
    auto sign_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto sign_instr = std::make_unique<Instruction>(Opcode::SRAI, parent_bb);
    sign_instr->addOperand_(std::make_unique<RegisterOperand>(
        sign_reg->getRegNum(), sign_reg->isVirtual()));
    sign_instr->addOperand_(std::make_unique<RegisterOperand>(
        dividend_reg->getRegNum(), dividend_reg->isVirtual()));
    sign_instr->addOperand_(std::make_unique<ImmediateOperand>(SIGN_BIT_SHIFT));
    parent_bb->addInstruction(std::move(sign_instr));

    // 2. 计算调整值：andi adjust_reg, sign_reg, (divisor - 1)
    auto adjust_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto andi_instr = std::make_unique<Instruction>(Opcode::ANDI, parent_bb);
    andi_instr->addOperand_(std::make_unique<RegisterOperand>(
        adjust_reg->getRegNum(), adjust_reg->isVirtual()));
    andi_instr->addOperand_(std::make_unique<RegisterOperand>(
        sign_reg->getRegNum(), sign_reg->isVirtual()));
    andi_instr->addOperand_(std::make_unique<ImmediateOperand>(divisor - 1));
    parent_bb->addInstruction(std::move(andi_instr));

    // 3. 加法调整：addw temp_reg, dividend, adjust_reg
    auto temp_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto add_instr = std::make_unique<Instruction>(Opcode::ADDW, parent_bb);
    add_instr->addOperand_(std::make_unique<RegisterOperand>(
        temp_reg->getRegNum(), temp_reg->isVirtual()));
    add_instr->addOperand_(std::make_unique<RegisterOperand>(
        dividend_reg->getRegNum(), dividend_reg->isVirtual()));
    add_instr->addOperand_(std::make_unique<RegisterOperand>(
        adjust_reg->getRegNum(), adjust_reg->isVirtual()));
    parent_bb->addInstruction(std::move(add_instr));

    // 4. 位掩码：andi result_reg, temp_reg, ~(divisor - 1)
    auto masked_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto and_mask_instr =
        std::make_unique<Instruction>(Opcode::ANDI, parent_bb);
    and_mask_instr->addOperand_(std::make_unique<RegisterOperand>(
        masked_reg->getRegNum(), masked_reg->isVirtual()));
    and_mask_instr->addOperand_(std::make_unique<RegisterOperand>(
        temp_reg->getRegNum(), temp_reg->isVirtual()));
    and_mask_instr->addOperand_(std::make_unique<ImmediateOperand>(
        static_cast<int32_t>(~static_cast<uint32_t>(divisor - 1))));
    parent_bb->addInstruction(std::move(and_mask_instr));

    // 5. 计算最终结果：subw result_reg, dividend, masked_reg
    auto result_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto sub_instr = std::make_unique<Instruction>(Opcode::SUBW, parent_bb);
    sub_instr->addOperand_(std::make_unique<RegisterOperand>(
        result_reg->getRegNum(), result_reg->isVirtual()));
    sub_instr->addOperand_(std::make_unique<RegisterOperand>(
        dividend_reg->getRegNum(), dividend_reg->isVirtual()));
    sub_instr->addOperand_(std::make_unique<RegisterOperand>(
        masked_reg->getRegNum(), masked_reg->isVirtual()));
    parent_bb->addInstruction(std::move(sub_instr));

    return result_reg;
}

auto MagicDivision::generateMagicDivision(
    std::unique_ptr<MachineOperand> dividend_operand, int32_t divisor,
    BasicBlock* parent_bb) -> std::unique_ptr<RegisterOperand> {
    if (!canOptimize(divisor)) {
        throw std::runtime_error("Cannot optimize division by " +
                                 std::to_string(divisor));
    }

    auto constants = computeMagicConstants(divisor);

    // 确保被除数在寄存器中
    auto dividend_reg = ensureRegister(std::move(dividend_operand), parent_bb);

    // 处理特殊情况：除以1
    if (divisor == 1) {
        return std::make_unique<RegisterOperand>(dividend_reg->getRegNum(),
                                                 dividend_reg->isVirtual());
    }

    // 处理特殊情况：除以-1
    if (divisor == -1) {
        static int next_reg_num = VIRTUAL_REG_START;
        auto result_reg =
            std::make_unique<RegisterOperand>(next_reg_num++, true);

        // subw result, zero, dividend  (result = 0 - dividend)
        auto neg_instr = std::make_unique<Instruction>(Opcode::SUBW, parent_bb);
        neg_instr->addOperand_(std::make_unique<RegisterOperand>(
            result_reg->getRegNum(), result_reg->isVirtual()));
        neg_instr->addOperand_(
            std::make_unique<RegisterOperand>(0, false));  // zero reg
        neg_instr->addOperand_(std::make_unique<RegisterOperand>(
            dividend_reg->getRegNum(), dividend_reg->isVirtual()));
        parent_bb->addInstruction(std::move(neg_instr));

        return result_reg;
    }

    // 处理2的幂的特殊情况
    if (isPowerOfTwo(divisor)) {
        return generatePowerOfTwoDivision(std::move(dividend_reg), divisor,
                                          parent_bb);
    }

    // 一般情况的魔数除法
    static int next_reg_num = VIRTUAL_REG_START;

    // 1. 加载魔数到寄存器
    auto magic_reg = generateLoadImmediate(constants.magic, parent_bb);

    // 2. 执行乘法：magic * dividend
    auto mul_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto mul_instr = std::make_unique<Instruction>(Opcode::MUL, parent_bb);
    mul_instr->addOperand_(std::make_unique<RegisterOperand>(
        mul_reg->getRegNum(), mul_reg->isVirtual()));
    mul_instr->addOperand_(std::make_unique<RegisterOperand>(
        magic_reg->getRegNum(), magic_reg->isVirtual()));
    mul_instr->addOperand_(std::make_unique<RegisterOperand>(
        dividend_reg->getRegNum(), dividend_reg->isVirtual()));
    parent_bb->addInstruction(std::move(mul_instr));

    // 3. 提取高32位：srai result, mul_result, 32
    auto high_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto srai_instr = std::make_unique<Instruction>(Opcode::SRAI, parent_bb);
    srai_instr->addOperand_(std::make_unique<RegisterOperand>(
        high_reg->getRegNum(), high_reg->isVirtual()));
    srai_instr->addOperand_(std::make_unique<RegisterOperand>(
        mul_reg->getRegNum(), mul_reg->isVirtual()));
    srai_instr->addOperand_(std::make_unique<ImmediateOperand>(SHIFT_32_BITS));
    parent_bb->addInstruction(std::move(srai_instr));

    auto current_reg = std::move(high_reg);

    // 4. 如果魔数为负，需要加上被除数
    if (constants.add_flag) {
        auto add_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
        auto add_instr = std::make_unique<Instruction>(Opcode::ADD, parent_bb);
        add_instr->addOperand_(std::make_unique<RegisterOperand>(
            add_reg->getRegNum(), add_reg->isVirtual()));
        add_instr->addOperand_(std::make_unique<RegisterOperand>(
            current_reg->getRegNum(), current_reg->isVirtual()));
        add_instr->addOperand_(std::make_unique<RegisterOperand>(
            dividend_reg->getRegNum(), dividend_reg->isVirtual()));
        parent_bb->addInstruction(std::move(add_instr));

        current_reg = std::move(add_reg);
    }

    // 5. 如果有额外移位，执行右移
    if (constants.shift > 0) {
        auto shift_reg =
            std::make_unique<RegisterOperand>(next_reg_num++, true);
        auto shift_instr =
            std::make_unique<Instruction>(Opcode::SRAI, parent_bb);
        shift_instr->addOperand_(std::make_unique<RegisterOperand>(
            shift_reg->getRegNum(), shift_reg->isVirtual()));
        shift_instr->addOperand_(std::make_unique<RegisterOperand>(
            current_reg->getRegNum(), current_reg->isVirtual()));
        shift_instr->addOperand_(
            std::make_unique<ImmediateOperand>(constants.shift));
        parent_bb->addInstruction(std::move(shift_instr));

        current_reg = std::move(shift_reg);
    }

    // 6. 计算符号位：srai sign_reg, dividend, 31
    auto sign_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto sign_instr = std::make_unique<Instruction>(Opcode::SRAI, parent_bb);
    sign_instr->addOperand_(std::make_unique<RegisterOperand>(
        sign_reg->getRegNum(), sign_reg->isVirtual()));
    sign_instr->addOperand_(std::make_unique<RegisterOperand>(
        dividend_reg->getRegNum(), dividend_reg->isVirtual()));
    sign_instr->addOperand_(std::make_unique<ImmediateOperand>(SIGN_BIT_SHIFT));
    parent_bb->addInstruction(std::move(sign_instr));

    // 7. 最终调整
    auto final_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);

    if (divisor > 0) {
        // subw final, result, sign
        auto sub_instr = std::make_unique<Instruction>(Opcode::SUBW, parent_bb);
        sub_instr->addOperand_(std::make_unique<RegisterOperand>(
            final_reg->getRegNum(), final_reg->isVirtual()));
        sub_instr->addOperand_(std::make_unique<RegisterOperand>(
            current_reg->getRegNum(), current_reg->isVirtual()));
        sub_instr->addOperand_(std::make_unique<RegisterOperand>(
            sign_reg->getRegNum(), sign_reg->isVirtual()));
        parent_bb->addInstruction(std::move(sub_instr));
    } else {
        // subw final, sign, result
        auto sub_instr = std::make_unique<Instruction>(Opcode::SUBW, parent_bb);
        sub_instr->addOperand_(std::make_unique<RegisterOperand>(
            final_reg->getRegNum(), final_reg->isVirtual()));
        sub_instr->addOperand_(std::make_unique<RegisterOperand>(
            sign_reg->getRegNum(), sign_reg->isVirtual()));
        sub_instr->addOperand_(std::make_unique<RegisterOperand>(
            current_reg->getRegNum(), current_reg->isVirtual()));
        parent_bb->addInstruction(std::move(sub_instr));
    }

    return final_reg;
}

auto MagicDivision::generateMagicModulo(
    std::unique_ptr<MachineOperand> dividend_operand, int32_t divisor,
    BasicBlock* parent_bb) -> std::unique_ptr<RegisterOperand> {
    // 保存被除数寄存器信息
    auto dividend_reg_num =
        dynamic_cast<RegisterOperand*>(dividend_operand.get())->getRegNum();
    auto dividend_is_virtual =
        dynamic_cast<RegisterOperand*>(dividend_operand.get())->isVirtual();

    // 处理2的幂的特殊情况
    if (isPowerOfTwo(divisor)) {
        auto dividend_reg = std::make_unique<RegisterOperand>(
            dividend_reg_num, dividend_is_virtual);
        return generatePowerOfTwoModulo(std::move(dividend_reg), divisor,
                                        parent_bb);
    }

    // 先计算 dividend / divisor
    auto quotient_reg =
        generateMagicDivision(std::move(dividend_operand), divisor, parent_bb);

    // 加载除数到寄存器
    auto divisor_reg = generateLoadImmediate(divisor, parent_bb);

    // 计算 quotient * divisor
    static int next_reg_num = VIRTUAL_REG_START;
    auto product_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto mul_instr = std::make_unique<Instruction>(Opcode::MULW, parent_bb);
    mul_instr->addOperand_(std::make_unique<RegisterOperand>(
        product_reg->getRegNum(), product_reg->isVirtual()));
    mul_instr->addOperand_(std::make_unique<RegisterOperand>(
        quotient_reg->getRegNum(), quotient_reg->isVirtual()));
    mul_instr->addOperand_(std::make_unique<RegisterOperand>(
        divisor_reg->getRegNum(), divisor_reg->isVirtual()));
    parent_bb->addInstruction(std::move(mul_instr));

    // 计算 dividend - (quotient * divisor)
    auto result_reg = std::make_unique<RegisterOperand>(next_reg_num++, true);
    auto sub_instr = std::make_unique<Instruction>(Opcode::SUBW, parent_bb);
    sub_instr->addOperand_(std::make_unique<RegisterOperand>(
        result_reg->getRegNum(), result_reg->isVirtual()));
    sub_instr->addOperand_(std::make_unique<RegisterOperand>(
        dividend_reg_num, dividend_is_virtual));
    sub_instr->addOperand_(std::make_unique<RegisterOperand>(
        product_reg->getRegNum(), product_reg->isVirtual()));
    parent_bb->addInstruction(std::move(sub_instr));

    return result_reg;
}

}  // namespace riscv64
