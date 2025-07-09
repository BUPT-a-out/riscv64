#include "Target.h"
#include "CodeGen.h"
#include "IR/Function.h"

namespace riscv64 {

std::vector<std::string> RISCV64Target::compileToAssembly(const midend::Module& module) {
    std::vector<std::string> assembly;
    CodeGenerator codegen;
    
    // 添加汇编头部
    assembly.push_back(".text");
    assembly.push_back(".global _start");
    
    // 为每个函数生成代码
    for (const auto* func : module) {
        assembly.push_back("");
        assembly.push_back(func->getName() + ":");
        
        auto funcCode = codegen.generateFunction(func);
        assembly.insert(assembly.end(), funcCode.begin(), funcCode.end());
    }
    
    return assembly;
}

}  // namespace riscv64