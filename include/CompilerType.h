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
    size_t array_size = 0;           // 0 表示不是数组
    std::vector<size_t> dimensions;  // 支持多维数组维度信息

    CompilerType(BaseType base_type, size_t array_size = 0)
        : base(base_type), array_size(array_size) {
        if (array_size > 0) {
            dimensions.push_back(array_size);
        }
    }

    CompilerType(BaseType base_type, const std::vector<size_t>& dims)
        : base(base_type), dimensions(dims) {
        array_size = 1;
        for (size_t dim : dims) {
            array_size *= dim;
        }
    }

    bool isArray() const { return array_size > 0; }

    bool isMultiDimensional() const { return dimensions.size() > 1; }

    // 获取此类型占用的总字节数
    size_t getSizeInBytes() const {
        size_t base_size = (base == BaseType::INT32) ? 4 : 4;
        return isArray() ? base_size * array_size : base_size;
    }

    // 获取维度信息的字符串表示
    std::string getDimensionString() const {
        if (!isArray()) return "";

        std::string result;
        for (size_t dim : dimensions) {
            result += "[" + std::to_string(dim) + "]";
        }
        return result;
    }
};

}  // namespace riscv64