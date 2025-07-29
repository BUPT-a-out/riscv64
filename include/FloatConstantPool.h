#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "Instructions/MachineOperand.h"

namespace riscv64 {

class FloatConstantPool {
   public:
    FloatConstantPool() = default;
    ~FloatConstantPool() = default;

    // 获取或创建浮点常量的标签
    std::string getOrCreateFloatConstant(float value);

    // 获取所有常量（用于代码生成阶段输出）
    const std::unordered_map<float, std::string>& getAllConstants() const;

    // 清空常量池
    void clear();

   private:
    std::unordered_map<float, std::string> floatConstants_;
    int nextLabelId_ = 0;

    // 生成唯一的标签名
    std::string generateLabel();
};

}  // namespace riscv64
