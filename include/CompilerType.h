#pragma once

#include <cstddef>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace riscv64 {

// 基础类型枚举
enum class BaseType { INT32, FLOAT32 };

// 编译器内部的类型表示
struct CompilerType {
    BaseType base;
    size_t array_size = 0;  // 0 表示不是数组

    CompilerType(BaseType base_type, size_t array_size = 0)
        : base(base_type), array_size(array_size) {}

    bool isArray() const { return array_size > 0; }

    // 获取此类型占用的总字节数
    size_t getSizeInBytes() const {
        size_t base_size = (base == BaseType::INT32) ? 4 : 4;
        return isArray() ? base_size * array_size : base_size;
    }
};

}  // namespace riscv64