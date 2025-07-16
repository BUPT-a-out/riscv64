#include <memory>

#include "IR/BasicBlock.h"
#include "IR/Constant.h"
#include "IR/Function.h"
#include "IR/IRBuilder.h"
#include "IR/Module.h"
#include "IR/Type.h"

// 简单的两个整数相加测试用例
std::unique_ptr<midend::Module> createModule() {
    auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("simple_imm_add", context.get());

    // 创建函数类型: i32 add(i32, i32)
    auto* i32Type = context->getInt32Type();
    auto* funcType = midend::FunctionType::get(i32Type, {});

    // 创建函数
    auto* func = midend::Function::Create(funcType, "main", module.get());

    // 创建基本块
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);

    // 创建IRBuilder
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 创建加法指令: %result = add i32 %a, %b
    auto* result =
        builder.createAdd(midend::ConstantInt::get(i32Type, 1),
                          midend::ConstantInt::get(i32Type, 3), "result");

    // 创建返回指令: ret i32 %result
    builder.createRet(result);

    return module;
}
