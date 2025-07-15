#pragma once
#include "IR/Module.h"
#include "Instructions/Module.h"
#include <string>
#include <vector>


namespace riscv64 {


class RISCV64Target {
public:
    RISCV64Target() = default;
    ~RISCV64Target() = default;

    std::string compileToAssembly(const midend::Module& module);
    Module instructionSelectionPass(const midend::Module& module); // 指令选择，生成的都是虚拟寄存器
    Module registerAllocationPass(riscv64::Module& module);   // 寄存器分配，把虚拟寄存器分配到物理寄存器上

};

}   // namespace riscv64