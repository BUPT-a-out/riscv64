#pragma once

#include <string>
#include <vector>

// riscv64::ABI 命名空间封装了所有与 RISC-V ABI 相关的功能
namespace riscv64::ABI {

// 根据ABI名称获取寄存器编号
    unsigned getRegNumFromABIName(const std::string& name);
    
    // 根据寄存器编号获取ABI名称
    std::string getABINameFromRegNum(unsigned num);
    
    // 判断是否为调用者保存寄存器 (Caller-saved/Temporary registers)
    bool isCallerSaved(unsigned physreg, bool isFloat);
    
    // 判断是否为被调用者保存寄存器 (Callee-saved/Saved registers)
    bool isCalleeSaved(unsigned physreg, bool isFloat);
    
    // 判断是否为参数寄存器 (Argument registers)
    bool isArgumentReg(unsigned physreg, bool isFloat);
    
    // 判断是否为返回值寄存器 (Return value registers)
    bool isReturnReg(unsigned physreg, bool isFloat);

    bool isReservedReg(unsigned physreg, bool isFloat);

    std::vector<unsigned> getCallerSavedRegs(bool isFloat);
}  // namespace riscv64::ABI