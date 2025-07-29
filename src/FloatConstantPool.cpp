#include "FloatConstantPool.h"

#include <iomanip>
#include <sstream>

namespace riscv64 {

std::string FloatConstantPool::getOrCreateFloatConstant(float value) {
    auto it = floatConstants_.find(value);
    if (it != floatConstants_.end()) {
        return it->second;
    }

    // 创建新的标签
    std::string label = generateLabel();
    floatConstants_[value] = label;
    return label;
}

const std::unordered_map<float, std::string>&
FloatConstantPool::getAllConstants() const {
    return floatConstants_;
}

void FloatConstantPool::clear() {
    floatConstants_.clear();
    nextLabelId_ = 0;
}

std::string FloatConstantPool::generateLabel() {
    std::stringstream ss;
    ss << ".LCPI_float_" << nextLabelId_++;
    return ss.str();
}

}  // namespace riscv64
