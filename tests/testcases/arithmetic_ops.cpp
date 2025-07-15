#include <memory>

#include "IR/BasicBlock.h"
#include "IR/Constant.h"
#include "IR/Function.h"
#include "IR/IRBuilder.h"
#include "IR/Module.h"
#include "IR/Type.h"

// 多个算术运算的测试用例
// 生成一个函数: i32 arithmetic(i32 a, i32 b) { return (a + b) * (a - b); }
std::unique_ptr<midend::Module> createModule() {
    auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("arithmetic_ops", context.get());

    // 创建函数类型: i32 arithmetic(i32, i32)
    auto* i32Type = context->getInt32Type();
    auto* funcType = midend::FunctionType::get(i32Type, {i32Type, i32Type});

    // 创建函数
    auto* func = midend::Function::Create(funcType, "arithmetic", module.get());

    // 获取参数
    auto* arg1 = func->getArg(0);
    auto* arg2 = func->getArg(1);
    arg1->setName("a");
    arg2->setName("b");

    // 创建基本块
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);

    // 创建IRBuilder
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 创建算术指令序列
    // %add_result = add i32 %a, %b
    auto* addResult = builder.createAdd(arg1, arg2, "add_result");

    // %sub_result = sub i32 %a, %b
    auto* subResult = builder.createSub(arg1, arg2, "sub_result");

    // %final_result = mul i32 %add_result, %sub_result
    auto* finalResult = builder.createMul(addResult, subResult, "final_result");

    // ret i32 %final_result
    builder.createRet(finalResult);

    return module;
}
