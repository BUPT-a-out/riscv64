#include "Target.h"

#include "CodeGen.h"
#include "IR/Function.h"
#include "Visit.h"
#include "FrameIndexPass.h"
#include "RegAllocChaitin.h"

namespace riscv64 {

std::string RISCV64Target::compileToAssembly(const midend::Module& module) {
    std::vector<std::string> assembly;

    // 添加汇编头部
    assembly.emplace_back(".text");
    assembly.emplace_back(".global _start");

    // 执行完整的编译流程
    auto riscv_module = instructionSelectionPass(module);
    registerAllocationPass(riscv_module);
    frameIndexPass(riscv_module);

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
    for (auto& function : module) {
        RegAllocChaitin allocator(function.get());
        allocator.allocateRegisters();
    }

    return module;
}

Module& RISCV64Target::frameIndexPass(riscv64::Module& module) {
    std::cout << "\n=== Running Frame Index Pass on Module ===" << std::endl;
    
    // 对模块中的每个函数运行栈帧布局Pass
    for (auto& function : module) {
        if (function->empty()) {
            // 跳过空函数
            std::cout << "Skipping empty function: " << function->getName() << std::endl;
            continue;
        }
        
        std::cout << "Processing function: " << function->getName() << std::endl;
        
        // 创建并运行FrameIndexPass
        FrameIndexPass framePass(function.get());
        framePass.run();
    }
    
    std::cout << "=== Frame Index Pass Completed ===" << std::endl;
    return module;
}

}  // namespace riscv64