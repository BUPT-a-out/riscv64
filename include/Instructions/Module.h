#pragma once
#include "Function.h"
#include <vector>

namespace riscv64 {

class Module {
   public:
    void addFunction(Function* func) { functions.push_back(func); }
    // ... 还可以添加全局变量、常量字符串等

   private:
    std::vector<Function*> functions;
    // std::vector<GlobalVariable*> globals;
};

}  // namespace riscv64