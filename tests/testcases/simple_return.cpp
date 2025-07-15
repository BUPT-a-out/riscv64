#include <memory>

#include "IR/BasicBlock.h"
#include "IR/Constant.h"
#include "IR/Function.h"
#include "IR/IRBuilder.h"
#include "IR/Module.h"
#include "IR/Type.h"

// 简单的返回常量测试用例
// 生成一个返回42的main函数
std::unique_ptr<midend::Module> createModule() {
    auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("simple_return", context.get());

    // 创建函数类型: i32 main()
    auto* i32Type = context->getInt32Type();
    auto* funcType = midend::FunctionType::get(i32Type, {});

    // 创建main函数
    auto* func = midend::Function::Create(funcType, "main", module.get());

    // 创建基本块
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);

    // 创建IRBuilder
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 创建返回指令: ret i32 42
    auto* retVal = midend::ConstantInt::get(i32Type, 42);
    builder.createRet(retVal);

    return module;
}
