#pragma once
#include "IR/Module.h"
#include <string>
#include <vector>


namespace riscv64 {


class RISCV64Target {
public:
    RISCV64Target() = default;
    ~RISCV64Target() = default;

    std::vector<std::string> compileToAssembly(const midend::Module& module);
};

}   // namespace riscv64