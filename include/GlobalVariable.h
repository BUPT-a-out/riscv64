#pragma once

#include <string>

#include "CompilerType.h"

namespace riscv64 {

// 用于标记显式的零初始化，以便放入 .bss 段
struct ZeroInitializer {};

// 使用 std::variant 来表示不同类型的初始化器
using ConstantInitializer = std::variant<int32_t,               // 单个 int
                                         float,                 // 单个 float
                                         std::vector<int32_t>,  // int 数组
                                         std::vector<float>,    // float 数组
                                         ZeroInitializer  // 显式零初始化
                                         >;

struct GlobalVariable {
    std::string name;
    CompilerType type;
    bool is_constant;  // 来自 LLVM IR 的 'constant' 关键字

    // 使用 std::optional 表示可能没有初始化器（针对 .bss 段）
    std::optional<ConstantInitializer> initializer;

    GlobalVariable(std::string n, CompilerType t, bool c,
                   std::optional<ConstantInitializer> i)
        : name(std::move(n)),
          type(t),
          is_constant(c),
          initializer(std::move(i)) {}
};

}  // namespace riscv64