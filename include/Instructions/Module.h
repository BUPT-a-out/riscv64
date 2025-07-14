#pragma once
#include <memory>
#include <vector>

#include "Function.h"
#include "MachineOperand.h"

namespace riscv64 {

class Module {
   public:
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

    std::string toString() const;

   private:
    std::vector<std::unique_ptr<Function>> functions;
    // std::vector<std::unique_ptr<GlobalVariable>> globals;
};

}  // namespace riscv64