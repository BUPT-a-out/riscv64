#include <memory>

#include "IR/BasicBlock.h"
#include "IR/Constant.h"
#include "IR/Function.h"
#include "IR/IRBuilder.h"
#include "IR/Module.h"
#include "IR/Type.h"

// 带有条件分支的测试用例
// 生成一个max函数: i32 max(i32 a, i32 b) { return a > b ? a : b; }
std::unique_ptr<midend::Module> createModule() {
    auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("conditional_branch", context.get());

    // 创建函数类型: i32 max(i32, i32)
    auto* i32Type = context->getInt32Type();
    auto* funcType = midend::FunctionType::get(i32Type, {i32Type, i32Type});

    // 创建max函数
    auto* func = midend::Function::Create(funcType, "max", module.get());

    // 获取参数
    auto* arg1 = func->getArg(0);
    auto* arg2 = func->getArg(1);
    arg1->setName("a");
    arg2->setName("b");

    // 创建基本块
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    auto* thenBB = midend::BasicBlock::Create(context.get(), "then", func);
    auto* elseBB = midend::BasicBlock::Create(context.get(), "else", func);
    auto* mergeBB = midend::BasicBlock::Create(context.get(), "merge", func);

    // 创建IRBuilder
    midend::IRBuilder builder(context.get());

    // entry块: 比较a和b
    builder.setInsertPoint(entry);
    auto* cmp = builder.createICmpSGT(arg1, arg2, "cmp");  // a > b
    builder.createCondBr(cmp, thenBB, elseBB);

    // then块: 返回a
    builder.setInsertPoint(thenBB);
    builder.createBr(mergeBB);

    // else块: 返回b
    builder.setInsertPoint(elseBB);
    builder.createBr(mergeBB);

    // merge块: 使用phi节点选择结果
    builder.setInsertPoint(mergeBB);
    auto* phi = builder.createPHI(i32Type, "result");
    phi->addIncoming(arg1, thenBB);  // 来自then块的值是a
    phi->addIncoming(arg2, elseBB);  // 来自else块的值是b
    builder.createRet(phi);

    return module;
}
