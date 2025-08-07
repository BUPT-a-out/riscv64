#pragma once
#include <memory>
#include <vector>

#include "Function.h"
#include "MachineOperand.h"
#include "Segment.h"

namespace riscv64 {

class Module {
   public:
    // 默认构造函数
    Module()
        : rodata_segment_(SegmentKind::RODATA),
          data_segment_(SegmentKind::DATA),
          bss_segment_(SegmentKind::BSS){};

    // 禁用拷贝构造函数和拷贝赋值运算符
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    // 启用移动构造函数和移动赋值运算符
    Module(Module&&) = default;
    Module& operator=(Module&&) = default;
    void addFunction(std::unique_ptr<Function> func) {
        functions.push_back(std::move(func));
    }

    // 提供访问函数的方法
    Function* getFunction(size_t index) const { return functions[index].get(); }

    size_t getFunctionCount() const { return functions.size(); }

    // 迭代器支持
    using iterator = std::vector<std::unique_ptr<Function>>::iterator;
    using const_iterator =
        std::vector<std::unique_ptr<Function>>::const_iterator;

    iterator begin() { return functions.begin(); }
    iterator end() { return functions.end(); }
    const_iterator begin() const { return functions.begin(); }
    const_iterator end() const { return functions.end(); }

    // 添加全局变量
    void addGlobal(GlobalVariable var) {
        std::cout << "Adding global variable: " << var.name
                  << ", constant: " << var.is_constant
                  << ", has_init: " << var.initializer.has_value() << std::endl;

        if (!var.initializer.has_value()) {
            // 没有初始化器 -> BSS 段
            bss_segment_.addItem(std::move(var));
            std::cout << "Added to BSS segment" << std::endl;
        } else {
            // 检查初始化器类型
            bool is_zero_init = std::holds_alternative<ZeroInitializer>(
                var.initializer.value());

            if (is_zero_init) {
                // 零初始化 -> BSS 段
                bss_segment_.addItem(std::move(var));
                std::cout << "Added to BSS segment (zero init)" << std::endl;
            } else {
                // 非零初始化
                if (var.is_constant) {
                    // 常量 -> RODATA 段
                    rodata_segment_.addItem(std::move(var));
                    std::cout << "Added to RODATA segment" << std::endl;
                } else {
                    // 变量 -> DATA 段
                    data_segment_.addItem(std::move(var));
                    std::cout << "Added to DATA segment" << std::endl;
                }
            }
        }
    }

    // 函数映射相关
    void addMidendFunctionMapping(const std::string& name,
                                  midend::Function* midend_func) {
        nameToMidendFunctionMap_[name] = midend_func;
    }

    midend::Function* getMidendFunction(const std::string& name) const {
        auto it = nameToMidendFunctionMap_.find(name);
        if (it != nameToMidendFunctionMap_.end()) {
            return it->second;
        }
        return nullptr;  // 如果没有找到，返回 nullptr
    }

    std::string toString() const;

   private:
    std::vector<std::unique_ptr<Function>> functions;
    // std::vector<std::unique_ptr<GlobalVariable>> globals;
    DataSegment rodata_segment_;
    DataSegment data_segment_;
    DataSegment bss_segment_;

    std::unordered_map<std::string, midend::Function*> nameToMidendFunctionMap_;
};

}  // namespace riscv64