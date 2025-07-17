#include "Target.h"

#include "CodeGen.h"
#include "IR/Function.h"
#include "Visit.h"

#include "RegAllocChaitin.h"

namespace riscv64 {

std::string RISCV64Target::compileToAssembly(const midend::Module& module) {
    std::vector<std::string> assembly;
    // CodeGenerator codegen;

    // 添加汇编头部
    assembly.emplace_back(".text");
    assembly.emplace_back(".global _start");

    // 为每个函数生成代码
    // for (const auto* func : module) {
    //     assembly.emplace_back("");
    //     assembly.push_back(func->getName() + ":");

    //     auto funcCode = codegen.generateFunction(func);
    //     assembly.insert(assembly.end(), funcCode.begin(), funcCode.end());
    // }
    auto riscv_module = instructionSelectionPass(module);

    return riscv_module.toString();
}

Module RISCV64Target::instructionSelectionPass(const midend::Module& module) {
    // 创建一个访客
    CodeGenerator codegen;
    // 使用访客访问模块
    auto riscv_module = codegen.visitor_->visit(&module);

    return riscv_module;
}

Module& RISCV64Target::registerAllocationPass(riscv64::Module& module) {
    // 这里实现寄存器分配逻辑
    for (auto& function: module) {
        RegAllocChaitin allocator(function.get());
        allocator.allocateRegisters();
    }

    return module;
}

}  // namespace riscv64