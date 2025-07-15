#pragma once

#include <optional>
#include <string>

// riscv64::ABI 命名空间封装了所有与 RISC-V ABI 相关的功能
namespace riscv64::ABI {

/**
 * @brief 将 ABI 寄存器名称映射到其物理寄存器编号。
 *
 * 该函数能够识别标准 ABI 名称 (如 "ra", "sp", "a0")、
 * x-prefixed 名称 (如 "x1", "x2", "x10") 以及常用别名 ("fp")。
 *
 * @param name 寄存器的 ABI 名称。
 * @return 如果名称有效，则返回对应的寄存器编号；
 *         否则抛出 std::invalid_argument 异常。
 *
 * @example
 *   getRegNumFromABIName("a0");  // returns 10
 *   getRegNumFromABIName("x1");  // returns 1
 *   getRegNumFromABIName("fp");   // returns 8
 *   getRegNumFromABIName("invalid"); // throws std::invalid_argument
 */
unsigned getRegNumFromABIName(const std::string& name);

/**
 * @brief 将物理寄存器编号映射回其规范的 ABI 名称。
 *
 * @param num 寄存器编号 (0-31)。
 * @return 如果编号有效，则返回对应的 ABI 名称；
 *         否则抛出 std::out_of_range 异常。
 *
 * @example
 *   getABINameFromRegNum(10); // returns "a0"
 *   getABINameFromRegNum(8);  // returns "s0"
 *   getABINameFromRegNum(32); // throws std::out_of_range
 */
std::string getABINameFromRegNum(unsigned num);

}  // namespace riscv64::ABI