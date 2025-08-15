#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "IR/BasicBlock.h"
#include "IR/Constant.h"
#include "IR/Function.h"
#include "IR/IRBuilder.h"
#include "IR/IRPrinter.h"
#include "IR/Module.h"
#include "IR/Type.h"
#include "Instructions/All.h"
#include "Target.h"

namespace riscv64::test {

// 测试用例生成器类型定义
using TestCaseGenerator = std::function<std::unique_ptr<midend::Module>()>;

// 前置声明测试用例创建函数
namespace testcases {
std::unique_ptr<midend::Module> createSimpleReturnTest();
std::unique_ptr<midend::Module> createSimpleAddTest();
std::unique_ptr<midend::Module> createArithmeticOpsTest();
std::unique_ptr<midend::Module> createConditionalBranchTest();
std::unique_ptr<midend::Module> createSimpleImmAddTest();
std::unique_ptr<midend::Module> createVariableAssignmentTest();
std::unique_ptr<midend::Module> createComplexAssignmentTest();
std::unique_ptr<midend::Module> createComplexBranchTest();
std::unique_ptr<midend::Module> createComplexFunctionCallTest();
std::unique_ptr<midend::Module> createSimpleArray1DTest();
std::unique_ptr<midend::Module> createSimpleArray2DTest();
std::unique_ptr<midend::Module> createComplexMemoryArrayTest();
std::unique_ptr<midend::Module> createPeepholeOptimizationCoverageTest();
}  // namespace testcases

class TestRunner {
   public:
    TestRunner();
    ~TestRunner() = default;

    // 禁用拷贝构造函数和拷贝赋值运算符
    TestRunner(const TestRunner&) = delete;
    TestRunner& operator=(const TestRunner&) = delete;

    // 启用移动构造函数和移动赋值运算符
    TestRunner(TestRunner&&) = default;
    TestRunner& operator=(TestRunner&&) = default;

    // 运行指定的测试用例
    bool runTestCase(const std::string& testCaseName);

    // 运行所有测试用例
    void runAllTests();

    // 列出所有可用的测试用例
    void listTestCases();

   private:
    // 从注册的测试用例生成器中加载测试用例
    std::unique_ptr<midend::Module> loadTestCase(
        const std::string& testCaseName);

    // 执行代码生成并打印结果
    static void executeCodeGeneration(const std::string& testCaseName,
                                      std::unique_ptr<midend::Module> module);

    // 获取测试用例的完整路径（保留用于调试）
    std::string getTestCasePath(const std::string& testCaseName);

    // 获取所有注册的测试用例的列表
    std::vector<std::string> getAvailableTestCases();

    // 注册的测试用例映射
    std::map<std::string, TestCaseGenerator> testCases_;

    // 测试用例目录（保留用于调试）
    const std::string testCaseDir =
        "/home/rikka/compiler/modules/riscv64/tests/testcases/";
};

std::string TestRunner::getTestCasePath(const std::string& testCaseName) {
    return testCaseDir + testCaseName + ".cpp";
}

std::vector<std::string> TestRunner::getAvailableTestCases() {
    std::vector<std::string> testCases;

    // 返回所有注册的测试用例名称
    for (const auto& [testName, generator] : testCases_) {
        testCases.push_back(testName);
    }

    return testCases;
}

// 测试用例创建函数的实现
namespace testcases {
std::unique_ptr<midend::Module> createSimpleReturnTest() {
    static auto context = std::make_unique<midend::Context>();
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
    constexpr int RETURN_VALUE = 42;
    auto* retVal = midend::ConstantInt::get(i32Type, RETURN_VALUE);
    builder.createRet(retVal);

    return module;
}

std::unique_ptr<midend::Module> createSimpleAddTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module = std::make_unique<midend::Module>("simple_add", context.get());

    // 创建函数类型: i32 add(i32, i32)
    auto* i32Type = context->getInt32Type();
    auto* funcType = midend::FunctionType::get(i32Type, {i32Type, i32Type});

    // 创建add函数
    auto* func = midend::Function::Create(funcType, "add", module.get());

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

    // 创建加法指令: %result = add i32 %a, %b
    auto* result = builder.createAdd(arg1, arg2, "result");

    // 创建返回指令: ret i32 %result
    builder.createRet(result);

    return module;
}

std::unique_ptr<midend::Module> createArithmeticOpsTest() {
    static auto context = std::make_unique<midend::Context>();
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

    // 创建算术指令序列: (a + b) * (a - b)
    auto* addResult = builder.createAdd(arg1, arg2, "add_result");
    auto* subResult = builder.createSub(arg1, arg2, "sub_result");
    auto* finalResult = builder.createMul(addResult, subResult, "final_result");

    builder.createRet(finalResult);

    return module;
}

std::unique_ptr<midend::Module> createConditionalBranchTest() {
    static auto context = std::make_unique<midend::Context>();
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

    // then块: 跳转到merge
    builder.setInsertPoint(thenBB);
    builder.createBr(mergeBB);

    // else块: 跳转到merge
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

std::unique_ptr<midend::Module> createSimpleImmAddTest() {
    static auto context = std::make_unique<midend::Context>();
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

std::unique_ptr<midend::Module> createVariableAssignmentTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("variable_assignment", context.get());

    // 创建函数类型: i32 test_var(i32)
    auto* i32Type = context->getInt32Type();
    auto* funcType = midend::FunctionType::get(i32Type, {i32Type});

    // 创建函数
    auto* func = midend::Function::Create(funcType, "test_var", module.get());

    // 获取参数
    auto* arg = func->getArg(0);
    arg->setName("input");

    // 创建基本块
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);

    // 创建IRBuilder
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 分配局部变量: %x = alloca i32
    auto* varX = builder.createAlloca(i32Type, nullptr, "x");

    // 分配另一个局部变量: %y = alloca i32
    auto* varY = builder.createAlloca(i32Type, nullptr, "y");

    // 存储输入参数到 x: store i32 %input, i32* %x
    builder.createStore(arg, varX);

    // 加载 x 的值: %temp1 = load i32, i32* %x
    auto* temp1 = builder.createLoad(varX, "temp1");

    // 计算 temp1 + 10: %temp2 = add i32 %temp1, 10
    auto* constant10 = midend::ConstantInt::get(i32Type, 10);
    auto* temp2 = builder.createAdd(temp1, constant10, "temp2");

    // 存储结果到 y: store i32 %temp2, i32* %y
    builder.createStore(temp2, varY);

    // 加载 y 的值: %result = load i32, i32* %y
    auto* result = builder.createLoad(varY, "result");

    // 返回结果: ret i32 %result
    builder.createRet(result);

    return module;
}

std::unique_ptr<midend::Module> createComplexAssignmentTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("complex_assignment", context.get());

    // 创建函数类型: i32 complex_assign(i32, i32, i32)
    auto* i32Type = context->getInt32Type();
    auto* funcType =
        midend::FunctionType::get(i32Type, {i32Type, i32Type, i32Type});

    // 创建函数
    auto* func =
        midend::Function::Create(funcType, "complex_assign", module.get());

    // 获取参数
    auto* arg1 = func->getArg(0);
    auto* arg2 = func->getArg(1);
    auto* arg3 = func->getArg(2);
    arg1->setName("a");
    arg2->setName("b");
    arg3->setName("c");

    // 创建基本块
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);

    // 创建IRBuilder
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 分配多个局部变量
    auto* varX = builder.createAlloca(i32Type, nullptr, "x");
    auto* varY = builder.createAlloca(i32Type, nullptr, "y");
    auto* varZ = builder.createAlloca(i32Type, nullptr, "z");
    auto* varW = builder.createAlloca(i32Type, nullptr, "w");
    auto* varResult = builder.createAlloca(i32Type, nullptr, "result");

    // 复杂的赋值序列
    // x = a + b
    auto* temp1 = builder.createAdd(arg1, arg2, "temp1");
    builder.createStore(temp1, varX);

    // y = a * c
    auto* temp2 = builder.createMul(arg1, arg3, "temp2");
    builder.createStore(temp2, varY);

    // z = b - c
    auto* temp3 = builder.createSub(arg2, arg3, "temp3");
    builder.createStore(temp3, varZ);

    // 加载变量进行更复杂的计算
    auto* loadX = builder.createLoad(varX, "load_x");
    auto* loadY = builder.createLoad(varY, "load_y");
    auto* loadZ = builder.createLoad(varZ, "load_z");

    // w = (x + y) * z
    auto* temp4 = builder.createAdd(loadX, loadY, "temp4");
    auto* temp5 = builder.createMul(temp4, loadZ, "temp5");
    builder.createStore(temp5, varW);

    // 最终计算: result = w + (x - y) + 100
    auto* loadW = builder.createLoad(varW, "load_w");
    auto* temp6 = builder.createSub(loadX, loadY, "temp6");
    auto* constant100 = midend::ConstantInt::get(i32Type, 100);
    auto* temp7 = builder.createAdd(loadW, temp6, "temp7");
    auto* finalResult = builder.createAdd(temp7, constant100, "final_result");

    builder.createStore(finalResult, varResult);

    // 返回结果
    auto* returnValue = builder.createLoad(varResult, "return_value");
    builder.createRet(returnValue);

    return module;
}

std::unique_ptr<midend::Module> createComplexBranchTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("complex_branch", context.get());

    // 创建函数类型: i32 complex_branch(i32, i32)
    auto* i32Type = context->getInt32Type();
    auto* funcType = midend::FunctionType::get(i32Type, {i32Type, i32Type});

    // 创建函数
    auto* func =
        midend::Function::Create(funcType, "complex_branch", module.get());

    // 获取参数
    auto* arg1 = func->getArg(0);
    auto* arg2 = func->getArg(1);
    arg1->setName("x");
    arg2->setName("y");

    // 创建基本块
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    auto* cond1BB = midend::BasicBlock::Create(context.get(), "cond1", func);
    auto* cond2BB = midend::BasicBlock::Create(context.get(), "cond2", func);
    auto* case1BB = midend::BasicBlock::Create(context.get(), "case1", func);
    auto* case2BB = midend::BasicBlock::Create(context.get(), "case2", func);
    auto* case3BB = midend::BasicBlock::Create(context.get(), "case3", func);
    auto* case4BB = midend::BasicBlock::Create(context.get(), "case4", func);
    auto* merge1BB = midend::BasicBlock::Create(context.get(), "merge1", func);
    auto* merge2BB = midend::BasicBlock::Create(context.get(), "merge2", func);
    auto* finalBB = midend::BasicBlock::Create(context.get(), "final", func);
    auto* adjustBB = midend::BasicBlock::Create(context.get(), "adjust", func);
    auto* noAdjustBB =
        midend::BasicBlock::Create(context.get(), "no_adjust", func);
    auto* exitBB = midend::BasicBlock::Create(context.get(), "exit", func);

    // 创建IRBuilder
    midend::IRBuilder builder(context.get());

    // entry块: 第一个条件判断 x > 0
    builder.setInsertPoint(entry);
    auto* zero = midend::ConstantInt::get(i32Type, 0);
    auto* cmp1 = builder.createICmpSGT(arg1, zero, "x_gt_0");
    builder.createCondBr(cmp1, cond1BB, cond2BB);

    // cond1块: x > 0 时，判断 y > 10
    builder.setInsertPoint(cond1BB);
    auto* ten = midend::ConstantInt::get(i32Type, 10);
    auto* cmp2 = builder.createICmpSGT(arg2, ten, "y_gt_10");
    builder.createCondBr(cmp2, case1BB, case2BB);

    // cond2块: x <= 0 时，判断 y < -5
    builder.setInsertPoint(cond2BB);
    auto* minusFive = midend::ConstantInt::get(i32Type, -5);
    auto* cmp3 = builder.createICmpSLT(arg2, minusFive, "y_lt_minus5");
    builder.createCondBr(cmp3, case3BB, case4BB);

    // case1块: x > 0 && y > 10, 计算 x * y + 100
    builder.setInsertPoint(case1BB);
    auto* mul1 = builder.createMul(arg1, arg2, "mul1");
    auto* hundred = midend::ConstantInt::get(i32Type, 100);
    auto* result1 = builder.createAdd(mul1, hundred, "result1");
    builder.createBr(merge1BB);

    // case2块: x > 0 && y <= 10, 计算 x + y * 2
    builder.setInsertPoint(case2BB);
    auto* two = midend::ConstantInt::get(i32Type, 2);
    auto* mul2 = builder.createMul(arg2, two, "mul2");
    auto* result2 = builder.createAdd(arg1, mul2, "result2");
    builder.createBr(merge1BB);

    // case3块: x <= 0 && y < -5, 计算 x - y + 50
    builder.setInsertPoint(case3BB);
    auto* sub1 = builder.createSub(arg1, arg2, "sub1");
    auto* fifty = midend::ConstantInt::get(i32Type, 50);
    auto* result3 = builder.createAdd(sub1, fifty, "result3");
    builder.createBr(merge2BB);

    // case4块: x <= 0 && y >= -5, 计算 x * 3 - y
    builder.setInsertPoint(case4BB);
    auto* three = midend::ConstantInt::get(i32Type, 3);
    auto* mul3 = builder.createMul(arg1, three, "mul3");
    auto* result4 = builder.createSub(mul3, arg2, "result4");
    builder.createBr(merge2BB);

    // merge1块: 合并 case1 和 case2 的结果
    builder.setInsertPoint(merge1BB);
    auto* phi1 = builder.createPHI(i32Type, "phi1");
    phi1->addIncoming(result1, case1BB);
    phi1->addIncoming(result2, case2BB);
    builder.createBr(finalBB);

    // merge2块: 合并 case3 和 case4 的结果
    builder.setInsertPoint(merge2BB);
    auto* phi2 = builder.createPHI(i32Type, "phi2");
    phi2->addIncoming(result3, case3BB);
    phi2->addIncoming(result4, case4BB);
    builder.createBr(finalBB);

    // final块: 最终的 phi 节点选择结果
    builder.setInsertPoint(finalBB);
    auto* finalPhi = builder.createPHI(i32Type, "final_result");
    finalPhi->addIncoming(phi1, merge1BB);
    finalPhi->addIncoming(phi2, merge2BB);

    // 最后再做一次判断: 如果结果 > 50 则减 10，否则加 5（使用分支代替select）
    auto* fifty_threshold = midend::ConstantInt::get(i32Type, 50);
    auto* isGreaterThan50 =
        builder.createICmpSGT(finalPhi, fifty_threshold, "is_gt_50");
    builder.createCondBr(isGreaterThan50, adjustBB, noAdjustBB);

    // adjust块: 结果 > 50，减去 10
    builder.setInsertPoint(adjustBB);
    auto* minusTen = midend::ConstantInt::get(i32Type, -10);
    auto* adjustedResult1 =
        builder.createAdd(finalPhi, minusTen, "adjusted_result1");
    builder.createBr(exitBB);

    // no_adjust块: 结果 <= 50，加上 5
    builder.setInsertPoint(noAdjustBB);
    auto* five = midend::ConstantInt::get(i32Type, 5);
    auto* adjustedResult2 =
        builder.createAdd(finalPhi, five, "adjusted_result2");
    builder.createBr(exitBB);

    // exit块: 最终的 phi 节点选择调整后的结果
    builder.setInsertPoint(exitBB);
    auto* exitPhi = builder.createPHI(i32Type, "exit_result");
    exitPhi->addIncoming(adjustedResult1, adjustBB);
    exitPhi->addIncoming(adjustedResult2, noAdjustBB);

    builder.createRet(exitPhi);

    return module;
}

std::unique_ptr<midend::Module> createComplexFunctionCallTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module = std::make_unique<midend::Module>("complex_function_call",
                                                   context.get());

    auto* i32Type = context->getInt32Type();

    // 创建辅助函数 1: i32 add_three(i32 a, i32 b, i32 c)
    auto* addThreeFuncType =
        midend::FunctionType::get(i32Type, {i32Type, i32Type, i32Type});
    auto* addThreeFunc =
        midend::Function::Create(addThreeFuncType, "add_three", module.get());

    auto* addThreeArg1 = addThreeFunc->getArg(0);
    auto* addThreeArg2 = addThreeFunc->getArg(1);
    auto* addThreeArg3 = addThreeFunc->getArg(2);
    addThreeArg1->setName("a");
    addThreeArg2->setName("b");
    addThreeArg3->setName("c");

    auto* addThreeEntry =
        midend::BasicBlock::Create(context.get(), "entry", addThreeFunc);
    midend::IRBuilder addThreeBuilder(context.get());
    addThreeBuilder.setInsertPoint(addThreeEntry);

    auto* temp1 =
        addThreeBuilder.createAdd(addThreeArg1, addThreeArg2, "temp1");
    auto* result1 = addThreeBuilder.createAdd(temp1, addThreeArg3, "result");
    addThreeBuilder.createRet(result1);

    // 创建辅助函数 2: i32 multiply_by_two(i32 x)
    auto* multiplyFuncType = midend::FunctionType::get(i32Type, {i32Type});
    auto* multiplyFunc = midend::Function::Create(
        multiplyFuncType, "multiply_by_two", module.get());

    auto* multiplyArg = multiplyFunc->getArg(0);
    multiplyArg->setName("x");

    auto* multiplyEntry =
        midend::BasicBlock::Create(context.get(), "entry", multiplyFunc);
    midend::IRBuilder multiplyBuilder(context.get());
    multiplyBuilder.setInsertPoint(multiplyEntry);

    auto* two = midend::ConstantInt::get(i32Type, 2);
    auto* result2 = multiplyBuilder.createMul(multiplyArg, two, "result");
    multiplyBuilder.createRet(result2);

    // 创建辅助函数 3: i32 compute_formula(i32 a, i32 b)
    auto* formulaFuncType =
        midend::FunctionType::get(i32Type, {i32Type, i32Type});
    auto* formulaFunc = midend::Function::Create(
        formulaFuncType, "compute_formula", module.get());

    auto* formulaArg1 = formulaFunc->getArg(0);
    auto* formulaArg2 = formulaFunc->getArg(1);
    formulaArg1->setName("a");
    formulaArg2->setName("b");

    auto* formulaEntry =
        midend::BasicBlock::Create(context.get(), "entry", formulaFunc);
    midend::IRBuilder formulaBuilder(context.get());
    formulaBuilder.setInsertPoint(formulaEntry);

    // 在 compute_formula 中调用 multiply_by_two
    std::vector<midend::Value*> multiplyArgs = {formulaArg1};
    auto* doubledA =
        formulaBuilder.createCall(multiplyFunc, multiplyArgs, "doubled_a");

    // 计算 doubled_a + b * 3
    auto* three = midend::ConstantInt::get(i32Type, 3);
    auto* bTimesThree =
        formulaBuilder.createMul(formulaArg2, three, "b_times_3");
    auto* formulaResult =
        formulaBuilder.createAdd(doubledA, bTimesThree, "formula_result");
    formulaBuilder.createRet(formulaResult);

    // 创建主函数: i32 main(i32 x, i32 y, i32 z)
    auto* mainFuncType =
        midend::FunctionType::get(i32Type, {i32Type, i32Type, i32Type});
    auto* mainFunc =
        midend::Function::Create(mainFuncType, "main", module.get());

    auto* mainArg1 = mainFunc->getArg(0);
    auto* mainArg2 = mainFunc->getArg(1);
    auto* mainArg3 = mainFunc->getArg(2);
    mainArg1->setName("x");
    mainArg2->setName("y");
    mainArg3->setName("z");

    auto* mainEntry =
        midend::BasicBlock::Create(context.get(), "entry", mainFunc);
    auto* condBB =
        midend::BasicBlock::Create(context.get(), "condition", mainFunc);
    auto* thenBB = midend::BasicBlock::Create(context.get(), "then", mainFunc);
    auto* elseBB = midend::BasicBlock::Create(context.get(), "else", mainFunc);
    auto* mergeBB =
        midend::BasicBlock::Create(context.get(), "merge", mainFunc);

    midend::IRBuilder mainBuilder(context.get());

    // entry块: 调用 add_three(x, y, z)
    mainBuilder.setInsertPoint(mainEntry);
    std::vector<midend::Value*> addThreeArgs = {mainArg1, mainArg2, mainArg3};
    auto* sumResult =
        mainBuilder.createCall(addThreeFunc, addThreeArgs, "sum_result");
    mainBuilder.createBr(condBB);

    // condition块: 检查 sum_result > 20
    mainBuilder.setInsertPoint(condBB);
    auto* twenty = midend::ConstantInt::get(i32Type, 20);
    auto* cmp = mainBuilder.createICmpSGT(sumResult, twenty, "cmp");
    mainBuilder.createCondBr(cmp, thenBB, elseBB);

    // then块: 调用 compute_formula(sum_result, x)
    mainBuilder.setInsertPoint(thenBB);
    std::vector<midend::Value*> formulaArgs1 = {sumResult, mainArg1};
    auto* thenResult =
        mainBuilder.createCall(formulaFunc, formulaArgs1, "then_result");
    mainBuilder.createBr(mergeBB);

    // else块: 调用 multiply_by_two(sum_result) 然后再调用 compute_formula
    mainBuilder.setInsertPoint(elseBB);
    std::vector<midend::Value*> multiplyArgs2 = {sumResult};
    auto* doubledSum =
        mainBuilder.createCall(multiplyFunc, multiplyArgs2, "doubled_sum");

    // 再调用 compute_formula(doubled_sum, y + z)
    auto* yzSum = mainBuilder.createAdd(mainArg2, mainArg3, "yz_sum");
    std::vector<midend::Value*> formulaArgs2 = {doubledSum, yzSum};
    auto* elseResult =
        mainBuilder.createCall(formulaFunc, formulaArgs2, "else_result");
    mainBuilder.createBr(mergeBB);

    // merge块: 使用 phi 节点合并结果，然后进行最终计算
    mainBuilder.setInsertPoint(mergeBB);
    auto* phi = mainBuilder.createPHI(i32Type, "phi_result");
    phi->addIncoming(thenResult, thenBB);
    phi->addIncoming(elseResult, elseBB);

    // 最终调用: multiply_by_two(phi_result) + add_three(10, 20, 30)
    std::vector<midend::Value*> finalMultiplyArgs = {phi};
    auto* finalDoubled = mainBuilder.createCall(multiplyFunc, finalMultiplyArgs,
                                                "final_doubled");

    auto* ten = midend::ConstantInt::get(i32Type, 10);
    auto* thirty = midend::ConstantInt::get(i32Type, 30);
    std::vector<midend::Value*> finalAddArgs = {ten, twenty, thirty};
    auto* constantSum =
        mainBuilder.createCall(addThreeFunc, finalAddArgs, "constant_sum");

    auto* finalResult =
        mainBuilder.createAdd(finalDoubled, constantSum, "final_result");
    mainBuilder.createRet(finalResult);

    return module;
}

std::unique_ptr<midend::Module> createLargeRegisterSpillTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("register_spill_test", context.get());

    auto* i32Type = context->getInt32Type();

    // 创建一个会导致寄存器溢出的复杂函数
    // i32 complex_computation(i32 a, i32 b, i32 c, i32 d, i32 e, i32 f, i32 g,
    // i32 h)
    auto* funcType = midend::FunctionType::get(
        i32Type, {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type,
                  i32Type});
    auto* func =
        midend::Function::Create(funcType, "complex_computation", module.get());

    // 设置参数名称
    auto* arg1 = func->getArg(0);
    arg1->setName("a");
    auto* arg2 = func->getArg(1);
    arg2->setName("b");
    auto* arg3 = func->getArg(2);
    arg3->setName("c");
    auto* arg4 = func->getArg(3);
    arg4->setName("d");
    auto* arg5 = func->getArg(4);
    arg5->setName("e");
    auto* arg6 = func->getArg(5);
    arg6->setName("f");
    auto* arg7 = func->getArg(6);
    arg7->setName("g");
    auto* arg8 = func->getArg(7);
    arg8->setName("h");

    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 创建大量常量
    auto* const1 = midend::ConstantInt::get(i32Type, 1);
    auto* const2 = midend::ConstantInt::get(i32Type, 2);
    auto* const3 = midend::ConstantInt::get(i32Type, 3);
    auto* const4 = midend::ConstantInt::get(i32Type, 4);
    auto* const5 = midend::ConstantInt::get(i32Type, 5);
    auto* const7 = midend::ConstantInt::get(i32Type, 7);
    auto* const11 = midend::ConstantInt::get(i32Type, 11);
    auto* const13 = midend::ConstantInt::get(i32Type, 13);

    // 第一层计算 - 生成大量中间值
    auto* temp1 = builder.createAdd(arg1, arg2, "temp1");
    auto* temp2 = builder.createMul(arg3, arg4, "temp2");
    auto* temp3 = builder.createAdd(arg5, arg6, "temp3");
    auto* temp4 = builder.createMul(arg7, arg8, "temp4");
    auto* temp5 = builder.createAdd(temp1, temp2, "temp5");
    auto* temp6 = builder.createMul(temp3, temp4, "temp6");
    auto* temp7 = builder.createAdd(temp5, temp6, "temp7");
    auto* temp8 = builder.createMul(temp7, const2, "temp8");

    // 第二层计算 - 更多中间值
    auto* inter1 = builder.createAdd(arg1, const1, "inter1");
    auto* inter2 = builder.createMul(arg2, const2, "inter2");
    auto* inter3 = builder.createAdd(arg3, const3, "inter3");
    auto* inter4 = builder.createMul(arg4, const4, "inter4");
    auto* inter5 = builder.createAdd(arg5, const5, "inter5");
    auto* inter6 = builder.createMul(arg6, const7, "inter6");
    auto* inter7 = builder.createAdd(arg7, const11, "inter7");
    auto* inter8 = builder.createMul(arg8, const13, "inter8");

    // 第三层计算 - 交叉运算
    auto* cross1 = builder.createAdd(inter1, inter2, "cross1");
    auto* cross2 = builder.createMul(inter3, inter4, "cross2");
    auto* cross3 = builder.createAdd(inter5, inter6, "cross3");
    auto* cross4 = builder.createMul(inter7, inter8, "cross4");
    auto* cross5 = builder.createAdd(cross1, cross3, "cross5");
    auto* cross6 = builder.createMul(cross2, cross4, "cross6");
    auto* cross7 = builder.createAdd(cross5, cross6, "cross7");
    auto* cross8 = builder.createMul(cross7, temp8, "cross8");

    // 第四层计算 - 更复杂的表达式
    auto* complex1 = builder.createAdd(temp1, inter1, "complex1");
    auto* complex2 = builder.createMul(temp2, inter2, "complex2");
    auto* complex3 = builder.createAdd(temp3, inter3, "complex3");
    auto* complex4 = builder.createMul(temp4, inter4, "complex4");
    auto* complex5 = builder.createAdd(temp5, inter5, "complex5");
    auto* complex6 = builder.createMul(temp6, inter6, "complex6");
    auto* complex7 = builder.createAdd(temp7, inter7, "complex7");
    auto* complex8 = builder.createMul(temp8, inter8, "complex8");

    // 第五层计算 - 组合前面的结果
    auto* combo1 = builder.createAdd(complex1, complex2, "combo1");
    auto* combo2 = builder.createMul(complex3, complex4, "combo2");
    auto* combo3 = builder.createAdd(complex5, complex6, "combo3");
    auto* combo4 = builder.createMul(complex7, complex8, "combo4");
    auto* combo5 = builder.createAdd(combo1, combo3, "combo5");
    auto* combo6 = builder.createMul(combo2, combo4, "combo6");
    auto* combo7 = builder.createAdd(combo5, combo6, "combo7");

    // 第六层计算 - 最终复杂表达式
    auto* final1 = builder.createAdd(combo7, cross8, "final1");
    auto* final2 = builder.createMul(final1, temp7, "final2");
    auto* final3 = builder.createAdd(final2, cross7, "final3");
    auto* final4 = builder.createMul(final3, combo5, "final4");
    auto* final5 = builder.createAdd(final4, combo6, "final5");
    auto* final6 = builder.createMul(final5, temp5, "final6");
    auto* final7 = builder.createAdd(final6, inter5, "final7");
    auto* final8 = builder.createMul(final7, cross5, "final8");

    // 最终结果 - 使用所有中间值
    auto* result1 = builder.createAdd(final8, temp1, "result1");
    auto* result2 = builder.createAdd(result1, temp2, "result2");
    auto* result3 = builder.createAdd(result2, temp3, "result3");
    auto* result4 = builder.createAdd(result3, temp4, "result4");
    auto* result5 = builder.createAdd(result4, inter1, "result5");
    auto* result6 = builder.createAdd(result5, inter2, "result6");
    auto* result7 = builder.createAdd(result6, inter3, "result7");
    auto* finalResult = builder.createAdd(result7, inter4, "final_result");

    builder.createRet(finalResult);

    return module;
}

std::unique_ptr<midend::Module> createSmallRegisterSpillTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("register_spill_test", context.get());

    auto* i32Type = context->getInt32Type();

    // 创建一个在RISC-V64上会导致轻微寄存器溢出的函数
    // i32 riscv_spill_test(i32 a, i32 b, i32 c, i32 d, i32 e, i32 f, i32 g, i32
    // h)
    auto* funcType = midend::FunctionType::get(
        i32Type, {i32Type, i32Type, i32Type, i32Type, i32Type, i32Type, i32Type,
                  i32Type});
    auto* func =
        midend::Function::Create(funcType, "riscv_spill_test", module.get());

    // 设置参数名称
    auto* arg1 = func->getArg(0);
    arg1->setName("a");
    auto* arg2 = func->getArg(1);
    arg2->setName("b");
    auto* arg3 = func->getArg(2);
    arg3->setName("c");
    auto* arg4 = func->getArg(3);
    arg4->setName("d");
    auto* arg5 = func->getArg(4);
    arg5->setName("e");
    auto* arg6 = func->getArg(5);
    arg6->setName("f");
    auto* arg7 = func->getArg(6);
    arg7->setName("g");
    auto* arg8 = func->getArg(7);
    arg8->setName("h");

    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 创建常量
    auto* const1 = midend::ConstantInt::get(i32Type, 1);
    auto* const2 = midend::ConstantInt::get(i32Type, 2);
    auto* const3 = midend::ConstantInt::get(i32Type, 3);
    auto* const7 = midend::ConstantInt::get(i32Type, 7);

    // 第一层计算 - 基础运算 (8个参数 + 8个中间值 = 16个活跃变量)
    auto* temp1 = builder.createAdd(arg1, arg2, "temp1");
    auto* temp2 = builder.createMul(arg3, arg4, "temp2");
    auto* temp3 = builder.createAdd(arg5, arg6, "temp3");
    auto* temp4 = builder.createMul(arg7, arg8, "temp4");
    auto* temp5 = builder.createAdd(temp1, const1, "temp5");
    auto* temp6 = builder.createMul(temp2, const2, "temp6");
    auto* temp7 = builder.createAdd(temp3, const3, "temp7");
    auto* temp8 = builder.createMul(temp4, const7, "temp8");

    // 第二层计算 - 增加复杂度 (前面16个 + 6个新的 = 22个活跃变量)
    auto* inter1 = builder.createAdd(temp1, temp2, "inter1");
    auto* inter2 = builder.createMul(temp3, temp4, "inter2");
    auto* inter3 = builder.createAdd(temp5, temp6, "inter3");
    auto* inter4 = builder.createMul(temp7, temp8, "inter4");
    auto* inter5 = builder.createAdd(inter1, inter2, "inter5");
    auto* inter6 = builder.createMul(inter3, inter4, "inter6");

    // 第三层计算 - 达到溢出临界点 (前面22个 + 4个新的 = 26个活跃变量)
    auto* cross1 = builder.createAdd(inter5, arg1, "cross1");
    auto* cross2 = builder.createMul(inter6, arg2, "cross2");
    auto* cross3 = builder.createAdd(cross1, temp5, "cross3");
    auto* cross4 = builder.createMul(cross2, temp6, "cross4");

    // 最终计算 - 逐步释放一些变量，但仍保持高压力
    auto* final1 = builder.createAdd(cross3, cross4, "final1");
    auto* final2 = builder.createMul(final1, inter5, "final2");
    auto* final3 = builder.createAdd(final2, temp7, "final3");
    auto* result = builder.createMul(final3, temp8, "result");

    builder.createRet(result);

    return module;
}

std::unique_ptr<midend::Module> createSimpleArray1DTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("simple_array_1d", context.get());

    auto* i32Type = context->getInt32Type();
    auto* arrayType = midend::ArrayType::get(i32Type, 5);  // int arr[5]

    // 创建函数类型: i32 array_sum()
    auto* funcType = midend::FunctionType::get(i32Type, {});
    auto* func = midend::Function::Create(funcType, "array_sum", module.get());

    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 分配数组: %arr = alloca [5 x i32]
    auto* arrayAlloca = builder.createAlloca(arrayType, nullptr, "arr");

    // 分配求和变量: %sum = alloca i32
    auto* sumAlloca = builder.createAlloca(i32Type, nullptr, "sum");

    // 初始化sum为0: store i32 0, i32* %sum
    auto* zero = midend::ConstantInt::get(i32Type, 0);
    builder.createStore(zero, sumAlloca);

    // 初始化数组元素
    // arr[0] = 10
    auto* idx0 = midend::ConstantInt::get(i32Type, 0);
    auto* ptr0 = builder.createGEP(arrayType, arrayAlloca, {idx0}, "arr_0_ptr");

    // 继续完成这个测试用例的实现
    auto val0 = midend::ConstantInt::get(i32Type, 10);
    builder.createStore(val0, ptr0);

    // arr[1] = 20
    auto* idx1 = midend::ConstantInt::get(i32Type, 1);
    auto* ptr1 = builder.createGEP(arrayType, arrayAlloca, {idx1}, "arr_1_ptr");
    auto val1 = midend::ConstantInt::get(i32Type, 20);
    builder.createStore(val1, ptr1);

    // arr[2] = 30
    auto* idx2 = midend::ConstantInt::get(i32Type, 2);
    auto* ptr2 = builder.createGEP(arrayType, arrayAlloca, {idx2}, "arr_2_ptr");
    auto val2 = midend::ConstantInt::get(i32Type, 30);
    builder.createStore(val2, ptr2);

    // arr[3] = 40
    auto* idx3 = midend::ConstantInt::get(i32Type, 3);
    auto* ptr3 = builder.createGEP(arrayType, arrayAlloca, {idx3}, "arr_3_ptr");
    auto val3 = midend::ConstantInt::get(i32Type, 40);
    builder.createStore(val3, ptr3);

    // arr[4] = 50
    auto* idx4 = midend::ConstantInt::get(i32Type, 4);
    auto* ptr4 = builder.createGEP(arrayType, arrayAlloca, {idx4}, "arr_4_ptr");
    auto val4 = midend::ConstantInt::get(i32Type, 50);
    builder.createStore(val4, ptr4);

    // 计算数组元素之和
    // sum += arr[0]
    auto currentSum = builder.createLoad(sumAlloca, "current_sum");
    auto elem0 = builder.createLoad(ptr0, "elem_0");
    auto sum1 = builder.createAdd(currentSum, elem0, "sum_1");
    builder.createStore(sum1, sumAlloca);

    // sum += arr[1]
    auto currentSum2 = builder.createLoad(sumAlloca, "current_sum_2");
    auto elem1 = builder.createLoad(ptr1, "elem_1");
    auto sum2 = builder.createAdd(currentSum2, elem1, "sum_2");
    builder.createStore(sum2, sumAlloca);

    // sum += arr[2]
    auto currentSum3 = builder.createLoad(sumAlloca, "current_sum_3");
    auto elem2 = builder.createLoad(ptr2, "elem_2");
    auto sum3 = builder.createAdd(currentSum3, elem2, "sum_3");
    builder.createStore(sum3, sumAlloca);

    // sum += arr[3]
    auto currentSum4 = builder.createLoad(sumAlloca, "current_sum_4");
    auto elem3 = builder.createLoad(ptr3, "elem_3");
    auto sum4 = builder.createAdd(currentSum4, elem3, "sum_4");
    builder.createStore(sum4, sumAlloca);

    // sum += arr[4]
    auto currentSum5 = builder.createLoad(sumAlloca, "current_sum_5");
    auto elem4 = builder.createLoad(ptr4, "elem_4");
    auto* finalSum = builder.createAdd(currentSum5, elem4, "final_sum");

    // 返回总和
    builder.createRet(finalSum);

    return module;
}

std::unique_ptr<midend::Module> createSimpleArray2DTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("simple_array_2d", context.get());

    auto* i32Type = context->getInt32Type();
    // 创建 3x3 的二维数组类型: [3 x [3 x i32]]
    auto* innerArrayType = midend::ArrayType::get(i32Type, 3);
    auto* outerArrayType = midend::ArrayType::get(innerArrayType, 3);

    // 创建函数类型: i32 matrix_diagonal_sum()
    auto* funcType = midend::FunctionType::get(i32Type, {});
    auto* func =
        midend::Function::Create(funcType, "matrix_diagonal_sum", module.get());

    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 分配二维数组: %matrix = alloca [3 x [3 x i32]]
    auto* matrixAlloca =
        builder.createAlloca(outerArrayType, nullptr, "matrix");

    // 分配对角线和变量: %diag_sum = alloca i32
    auto* diagSumAlloca = builder.createAlloca(i32Type, nullptr, "diag_sum");

    // 初始化对角线和为0: store i32 0, i32* %diag_sum
    auto* zero = midend::ConstantInt::get(i32Type, 0);
    builder.createStore(zero, diagSumAlloca);

    // 初始化矩阵元素（只初始化对角线和一些其他元素）
    // matrix[0][0] = 1
    auto* idx0 = midend::ConstantInt::get(i32Type, 0);
    auto* idx1 = midend::ConstantInt::get(i32Type, 1);
    auto* idx2 = midend::ConstantInt::get(i32Type, 2);

    auto* val1 = midend::ConstantInt::get(i32Type, 1);
    auto* val5 = midend::ConstantInt::get(i32Type, 5);
    auto* val9 = midend::ConstantInt::get(i32Type, 9);
    auto* val2 = midend::ConstantInt::get(i32Type, 2);
    auto* val3 = midend::ConstantInt::get(i32Type, 3);

    // matrix[0][0] = 1 (对角线元素)
    auto* ptr00 = builder.createGEP(outerArrayType, matrixAlloca, {idx0, idx0},
                                    "matrix_0_0_ptr");
    builder.createStore(val1, ptr00);

    // matrix[0][1] = 2
    auto* ptr01 = builder.createGEP(outerArrayType, matrixAlloca, {idx0, idx1},
                                    "matrix_0_1_ptr");
    builder.createStore(val2, ptr01);

    // matrix[1][1] = 5 (对角线元素)
    auto* ptr11 = builder.createGEP(outerArrayType, matrixAlloca, {idx1, idx1},
                                    "matrix_1_1_ptr");
    builder.createStore(val5, ptr11);

    // matrix[1][0] = 3
    auto* ptr10 = builder.createGEP(outerArrayType, matrixAlloca, {idx1, idx0},
                                    "matrix_1_0_ptr");
    builder.createStore(val3, ptr10);

    // matrix[2][2] = 9 (对角线元素)
    auto* ptr22 = builder.createGEP(outerArrayType, matrixAlloca, {idx2, idx2},
                                    "matrix_2_2_ptr");
    builder.createStore(val9, ptr22);

    // 计算对角线元素之和
    // diag_sum += matrix[0][0]
    auto* currentSum = builder.createLoad(diagSumAlloca, "current_diag_sum");
    auto* elem00 = builder.createLoad(ptr00, "elem_0_0");
    auto* newSum = builder.createAdd(currentSum, elem00, "diag_sum_1");
    builder.createStore(newSum, diagSumAlloca);

    // diag_sum += matrix[1][1]
    currentSum = builder.createLoad(diagSumAlloca, "current_diag_sum_2");
    auto* elem11 = builder.createLoad(ptr11, "elem_1_1");
    newSum = builder.createAdd(currentSum, elem11, "diag_sum_2");
    builder.createStore(newSum, diagSumAlloca);

    // diag_sum += matrix[2][2]
    currentSum = builder.createLoad(diagSumAlloca, "current_diag_sum_3");
    auto* elem22 = builder.createLoad(ptr22, "elem_2_2");
    auto* finalSum = builder.createAdd(currentSum, elem22, "final_diag_sum");

    // 为了测试非对角线元素访问，再加上matrix[0][1] * matrix[1][0]
    auto* elem01 = builder.createLoad(ptr01, "elem_0_1");
    auto* elem10 = builder.createLoad(ptr10, "elem_1_0");
    auto* product = builder.createMul(elem01, elem10, "off_diag_product");
    auto* result = builder.createAdd(finalSum, product, "result");

    // 返回结果
    builder.createRet(result);

    return module;
}

std::unique_ptr<midend::Module> createComplexMemoryArrayTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("complex_memory_array", context.get());

    auto* i32Type = context->getInt32Type();
    auto* arrayType = midend::ArrayType::get(i32Type, 8);  // int arr[8]

    // 创建函数类型: i32 complex_array_ops(i32 n)
    auto* funcType = midend::FunctionType::get(i32Type, {i32Type});
    auto* func =
        midend::Function::Create(funcType, "complex_array_ops", module.get());

    auto* arg = func->getArg(0);
    arg->setName("n");

    // 创建基本块
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    auto* initLoopBB =
        midend::BasicBlock::Create(context.get(), "init_loop", func);
    auto* initCondBB =
        midend::BasicBlock::Create(context.get(), "init_cond", func);
    auto* processLoopBB =
        midend::BasicBlock::Create(context.get(), "process_loop", func);
    auto* processCondBB =
        midend::BasicBlock::Create(context.get(), "process_cond", func);
    auto* finalBB = midend::BasicBlock::Create(context.get(), "final", func);

    midend::IRBuilder builder(context.get());

    // entry块: 初始化
    builder.setInsertPoint(entry);

    // 分配数组和各种变量
    auto* arrayAlloca = builder.createAlloca(arrayType, nullptr, "arr");
    auto* iAlloca = builder.createAlloca(i32Type, nullptr, "i");
    auto* sumAlloca = builder.createAlloca(i32Type, nullptr, "sum");
    auto* tempAlloca = builder.createAlloca(i32Type, nullptr, "temp");
    auto* maxAlloca = builder.createAlloca(i32Type, nullptr, "max_val");

    // 初始化变量
    auto* zero = midend::ConstantInt::get(i32Type, 0);
    auto* one = midend::ConstantInt::get(i32Type, 1);
    auto* eight = midend::ConstantInt::get(i32Type, 8);
    auto* minusOne = midend::ConstantInt::get(i32Type, -1);

    builder.createStore(zero, iAlloca);
    builder.createStore(zero, sumAlloca);
    builder.createStore(minusOne, maxAlloca);  // 初始最大值为-1
    builder.createBr(initCondBB);

    // 初始化循环条件检查
    builder.setInsertPoint(initCondBB);
    auto* currentI = builder.createLoad(iAlloca, "current_i");
    auto* initCond = builder.createICmpSLT(currentI, eight, "init_cond");
    builder.createCondBr(initCond, initLoopBB, processCondBB);

    // 初始化循环体：arr[i] = i * n + (i % 3)
    builder.setInsertPoint(initLoopBB);
    currentI = builder.createLoad(iAlloca, "i_for_init");

    // 计算 i * n
    auto* iTimesN = builder.createMul(currentI, arg, "i_times_n");

    // 计算 i % 3
    auto* three = midend::ConstantInt::get(i32Type, 3);
    auto* iMod3 = builder.createRem(currentI, three, "i_mod_3");

    // 计算最终值
    auto* initValue = builder.createAdd(iTimesN, iMod3, "init_value");

    // 存储到数组
    auto* elemPtr =
        builder.createGEP(arrayType, arrayAlloca, {currentI}, "elem_ptr");
    builder.createStore(initValue, elemPtr);

    // i++
    auto* nextI = builder.createAdd(currentI, one, "next_i");
    builder.createStore(nextI, iAlloca);
    builder.createBr(initCondBB);

    // 重置i为0，开始处理循环
    builder.setInsertPoint(processCondBB);
    builder.createStore(zero, iAlloca);
    builder.createBr(processLoopBB);

    // 处理循环体：复杂的数组操作
    builder.setInsertPoint(processLoopBB);
    currentI = builder.createLoad(iAlloca, "i_for_process");

    // 检查循环条件 i < 8
    auto* processCond = builder.createICmpSLT(currentI, eight, "process_cond");
    auto* afterLoopBB =
        midend::BasicBlock::Create(context.get(), "after_loop", func);
    builder.createCondBr(processCond, afterLoopBB, finalBB);

    builder.setInsertPoint(afterLoopBB);
    currentI = builder.createLoad(iAlloca, "i_in_process");

    // 加载当前元素 arr[i]
    auto* currentElemPtr = builder.createGEP(arrayType, arrayAlloca, {currentI},
                                             "current_elem_ptr");
    auto* currentElem = builder.createLoad(currentElemPtr, "current_elem");

    // 更新sum: sum += arr[i]
    auto* currentSum = builder.createLoad(sumAlloca, "current_sum");
    auto* newSum = builder.createAdd(currentSum, currentElem, "new_sum");
    builder.createStore(newSum, sumAlloca);

    // 更新最大值
    auto* currentMax = builder.createLoad(maxAlloca, "current_max");
    auto* isGreater =
        builder.createICmpSGT(currentElem, currentMax, "is_greater");
    auto* updateMaxBB =
        midend::BasicBlock::Create(context.get(), "update_max", func);
    auto* keepMaxBB =
        midend::BasicBlock::Create(context.get(), "keep_max", func);
    auto* afterMaxBB =
        midend::BasicBlock::Create(context.get(), "after_max", func);

    builder.createCondBr(isGreater, updateMaxBB, keepMaxBB);

    // 更新最大值
    builder.setInsertPoint(updateMaxBB);
    builder.createStore(currentElem, maxAlloca);
    builder.createBr(afterMaxBB);

    // 保持原最大值
    builder.setInsertPoint(keepMaxBB);
    builder.createBr(afterMaxBB);

    // 继续执行
    builder.setInsertPoint(afterMaxBB);
    auto* newMax = builder.createLoad(maxAlloca, "new_max");
    builder.createStore(newMax, maxAlloca);

    // 如果i是偶数，将arr[i]乘以2
    auto* two = midend::ConstantInt::get(i32Type, 2);
    auto* iMod2 = builder.createRem(currentI, two, "i_mod_2");
    auto* isEven = builder.createICmpEQ(iMod2, zero, "is_even");
    auto* doubledElem = builder.createMul(currentElem, two, "doubled_elem");
    auto* evenBB = midend::BasicBlock::Create(context.get(), "even", func);
    auto* oddBB = midend::BasicBlock::Create(context.get(), "odd", func);
    auto* afterEvenOddBB =
        midend::BasicBlock::Create(context.get(), "after_even_odd", func);

    builder.createCondBr(isEven, evenBB, oddBB);

    // 偶数情况：元素乘以2
    builder.setInsertPoint(evenBB);
    auto* updatedElemEven = doubledElem;
    builder.createBr(afterEvenOddBB);

    // 奇数情况：保持原值
    builder.setInsertPoint(oddBB);
    auto* updatedElemOdd = currentElem;
    builder.createBr(afterEvenOddBB);

    // 合并后继续
    builder.setInsertPoint(afterEvenOddBB);
    auto* updatedElem = builder.createPHI(i32Type, "updated_elem");
    updatedElem->addIncoming(updatedElemEven, evenBB);
    updatedElem->addIncoming(updatedElemOdd, oddBB);
    builder.createStore(updatedElem, currentElemPtr);

    // 如果i > 0，将当前元素与前一个元素交换
    auto* iPrevious = builder.createSub(currentI, one, "i_previous");
    auto* hasPrevoius = builder.createICmpSGT(currentI, zero, "has_previous");

    // 创建条件块进行交换
    auto* swapBB = midend::BasicBlock::Create(context.get(), "swap", func);
    auto* noSwapBB = midend::BasicBlock::Create(context.get(), "no_swap", func);
    auto* afterSwapBB =
        midend::BasicBlock::Create(context.get(), "after_swap", func);

    builder.createCondBr(hasPrevoius, swapBB, noSwapBB);

    // 交换逻辑
    builder.setInsertPoint(swapBB);
    auto* prevElemPtr =
        builder.createGEP(arrayType, arrayAlloca, {iPrevious}, "prev_elem_ptr");
    auto* currentElemForSwap =
        builder.createLoad(currentElemPtr, "current_for_swap");
    auto* prevElem = builder.createLoad(prevElemPtr, "prev_elem");
    builder.createStore(currentElemForSwap, prevElemPtr);
    builder.createStore(prevElem, currentElemPtr);
    builder.createBr(afterSwapBB);

    // 不交换
    builder.setInsertPoint(noSwapBB);
    builder.createBr(afterSwapBB);

    // 交换后继续
    builder.setInsertPoint(afterSwapBB);

    // 存储临时计算结果
    auto* tempValue = builder.createAdd(currentI, newSum, "temp_value");
    builder.createStore(tempValue, tempAlloca);

    // i++
    nextI = builder.createAdd(currentI, one, "next_i_process");
    builder.createStore(nextI, iAlloca);
    builder.createBr(processLoopBB);

    // 最终计算
    builder.setInsertPoint(finalBB);
    auto* finalSum = builder.createLoad(sumAlloca, "final_sum");
    auto* finalMax = builder.createLoad(maxAlloca, "final_max");
    auto* tempVal = builder.createLoad(tempAlloca, "final_temp");

    // 计算最终结果: sum + max + temp - n
    auto* intermediate1 =
        builder.createAdd(finalSum, finalMax, "intermediate1");
    auto* intermediate2 =
        builder.createAdd(intermediate1, tempVal, "intermediate2");
    auto* result = builder.createSub(intermediate2, arg, "result");

    builder.createRet(result);

    return module;
}

std::unique_ptr<midend::Module> createComprehensiveBinaryOpsTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module = std::make_unique<midend::Module>("comprehensive_binary_ops",
                                                   context.get());

    auto* i32Type = context->getInt32Type();

    // 创建函数类型: i32 test_all_binary_ops(i32 a, i32 b, i32 c, i32 d)
    auto* funcType = midend::FunctionType::get(
        i32Type, {i32Type, i32Type, i32Type, i32Type});
    auto* func =
        midend::Function::Create(funcType, "test_all_binary_ops", module.get());

    // 获取参数
    auto* arg1 = func->getArg(0);
    arg1->setName("a");
    auto* arg2 = func->getArg(1);
    arg2->setName("b");
    auto* arg3 = func->getArg(2);
    arg3->setName("c");
    auto* arg4 = func->getArg(3);
    arg4->setName("d");

    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 创建常量用于测试立即数操作
    auto* const5 = midend::ConstantInt::get(i32Type, 5);
    auto* const3 = midend::ConstantInt::get(i32Type, 3);
    auto* const2 = midend::ConstantInt::get(i32Type, 2);
    auto* const10 = midend::ConstantInt::get(i32Type, 10);
    auto* const7 = midend::ConstantInt::get(i32Type, 7);
    auto* const15 = midend::ConstantInt::get(i32Type, 15);

    // === 算术运算测试 ===

    // 1. Add: 测试寄存器+寄存器, 寄存器+立即数, 立即数+立即数
    auto* add_reg_reg = builder.createAdd(arg1, arg2, "add_reg_reg");
    auto* add_reg_imm = builder.createAdd(arg1, const5, "add_reg_imm");
    auto* add_imm_reg = builder.createAdd(const3, arg2, "add_imm_reg");
    auto* add_imm_imm = builder.createAdd(const5, const3, "add_imm_imm");

    // 2. Sub: 测试各种组合
    auto* sub_reg_reg = builder.createSub(arg1, arg2, "sub_reg_reg");
    auto* sub_reg_imm = builder.createSub(arg1, const5, "sub_reg_imm");
    auto* sub_imm_reg = builder.createSub(const10, arg2, "sub_imm_reg");
    auto* sub_imm_imm = builder.createSub(const10, const3, "sub_imm_imm");

    // 3. Mul: 测试各种组合
    auto* mul_reg_reg = builder.createMul(arg1, arg2, "mul_reg_reg");
    auto* mul_reg_imm = builder.createMul(arg1, const2, "mul_reg_imm");
    auto* mul_imm_reg = builder.createMul(const3, arg2, "mul_imm_reg");
    auto* mul_imm_imm = builder.createMul(const3, const2, "mul_imm_imm");

    // 4. Div: 测试各种组合
    auto* div_reg_reg = builder.createDiv(arg3, arg4, "div_reg_reg");
    auto* div_reg_imm = builder.createDiv(arg3, const2, "div_reg_imm");
    auto* div_imm_reg = builder.createDiv(const10, arg4, "div_imm_reg");
    auto* div_imm_imm = builder.createDiv(const10, const2, "div_imm_imm");

    // 5. Rem: 测试各种组合
    auto* rem_reg_reg = builder.createRem(arg3, arg4, "rem_reg_reg");
    auto* rem_reg_imm = builder.createRem(arg3, const3, "rem_reg_imm");
    auto* rem_imm_reg = builder.createRem(const7, arg4, "rem_imm_reg");
    auto* rem_imm_imm = builder.createRem(const7, const3, "rem_imm_imm");

    // === 位运算测试 ===

    // 6. And: 测试各种组合
    auto* and_reg_reg = builder.createAnd(arg1, arg2, "and_reg_reg");
    auto* and_reg_imm = builder.createAnd(arg1, const15, "and_reg_imm");
    auto* and_imm_reg = builder.createAnd(const15, arg2, "and_imm_reg");
    auto* and_imm_imm = builder.createAnd(const15, const7, "and_imm_imm");

    // 7. Or: 测试各种组合
    auto* or_reg_reg = builder.createOr(arg1, arg2, "or_reg_reg");
    auto* or_reg_imm = builder.createOr(arg1, const15, "or_reg_imm");
    auto* or_imm_reg = builder.createOr(const15, arg2, "or_imm_reg");
    auto* or_imm_imm = builder.createOr(const15, const7, "or_imm_imm");

    // 8. Xor: 测试各种组合
    auto* xor_reg_reg = builder.createXor(arg1, arg2, "xor_reg_reg");
    auto* xor_reg_imm = builder.createXor(arg1, const15, "xor_reg_imm");
    auto* xor_imm_reg = builder.createXor(const15, arg2, "xor_imm_reg");
    auto* xor_imm_imm = builder.createXor(const15, const7, "xor_imm_imm");

    // 9. Shl: 测试各种组合
    // auto* shl_reg_reg = builder.createShl(arg1, arg2, "shl_reg_reg");
    // auto* shl_reg_imm = builder.createShl(arg1, const2, "shl_reg_imm");
    // auto* shl_imm_reg = builder.createShl(const10, arg2, "shl_imm_reg");
    // auto* shl_imm_imm = builder.createShl(const10, const2, "shl_imm_imm");

    // // 10. Shr: 测试各种组合
    // auto* shr_reg_reg = builder.createAShr(arg1, arg2, "shr_reg_reg");
    // auto* shr_reg_imm = builder.createAShr(arg1, const2, "shr_reg_imm");
    // auto* shr_imm_reg = builder.createAShr(const10, arg2, "shr_imm_reg");
    // auto* shr_imm_imm = builder.createAShr(const10, const2, "shr_imm_imm");

    // === 比较运算测试 ===

    // 11. ICmpSGT: 测试有符号大于比较
    auto* sgt_reg_reg = builder.createICmpSGT(arg1, arg2, "sgt_reg_reg");
    auto* sgt_reg_imm = builder.createICmpSGT(arg1, const5, "sgt_reg_imm");
    auto* sgt_imm_reg = builder.createICmpSGT(const10, arg2, "sgt_imm_reg");
    auto* sgt_imm_imm = builder.createICmpSGT(const10, const5, "sgt_imm_imm");

    // 12. ICmpEQ: 测试相等比较
    auto* eq_reg_reg = builder.createICmpEQ(arg1, arg2, "eq_reg_reg");
    auto* eq_reg_imm = builder.createICmpEQ(arg1, const5, "eq_reg_imm");
    auto* eq_imm_reg = builder.createICmpEQ(const5, arg2, "eq_imm_reg");
    auto* eq_imm_imm = builder.createICmpEQ(const5, const5, "eq_imm_imm");

    // 13. ICmpNE: 测试不等比较
    auto* ne_reg_reg = builder.createICmpNE(arg1, arg2, "ne_reg_reg");
    auto* ne_reg_imm = builder.createICmpNE(arg1, const5, "ne_reg_imm");
    auto* ne_imm_reg = builder.createICmpNE(const5, arg2, "ne_imm_reg");
    auto* ne_imm_imm = builder.createICmpNE(const5, const7, "ne_imm_imm");

    // 14. ICmpSLT: 测试有符号小于比较
    auto* slt_reg_reg = builder.createICmpSLT(arg1, arg2, "slt_reg_reg");
    auto* slt_reg_imm = builder.createICmpSLT(arg1, const5, "slt_reg_imm");
    auto* slt_imm_reg = builder.createICmpSLT(const3, arg2, "slt_imm_reg");
    auto* slt_imm_imm = builder.createICmpSLT(const3, const5, "slt_imm_imm");

    // 15. ICmpSLE: 测试有符号小于等于比较
    auto* sle_reg_reg = builder.createICmpSLE(arg1, arg2, "sle_reg_reg");
    auto* sle_reg_imm = builder.createICmpSLE(arg1, const5, "sle_reg_imm");
    auto* sle_imm_reg = builder.createICmpSLE(const5, arg2, "sle_imm_reg");
    auto* sle_imm_imm = builder.createICmpSLE(const5, const5, "sle_imm_imm");

    // 16. ICmpSGE: 测试有符号大于等于比较
    auto* sge_reg_reg = builder.createICmpSGE(arg1, arg2, "sge_reg_reg");
    auto* sge_reg_imm = builder.createICmpSGE(arg1, const5, "sge_reg_imm");
    auto* sge_imm_reg = builder.createICmpSGE(const5, arg2, "sge_imm_reg");
    auto* sge_imm_imm = builder.createICmpSGE(const5, const5, "sge_imm_imm");

    // === 组合计算：使用所有运算结果 ===

    // 将所有算术运算结果相加
    auto* arith_sum1 =
        builder.createAdd(add_reg_reg, sub_reg_reg, "arith_sum1");
    auto* arith_sum2 =
        builder.createAdd(mul_reg_reg, div_reg_reg, "arith_sum2");
    auto* arith_sum3 = builder.createAdd(rem_reg_reg, arith_sum1, "arith_sum3");
    auto* arith_total =
        builder.createAdd(arith_sum2, arith_sum3, "arith_total");

    // 将所有位运算结果相加
    auto* bit_sum1 = builder.createAdd(and_reg_reg, or_reg_reg, "bit_sum1");
    // auto* bit_sum2 = builder.createAdd(xor_reg_reg, shl_reg_reg, "bit_sum2");
    // auto* bit_sum3 = builder.createAdd(shr_reg_reg, bit_sum1, "bit_sum3");
    // auto* bit_total = builder.createAdd(bit_sum2, bit_sum3, "bit_total");

    // 将比较运算结果转换为整数并相加
    auto* cmp_sum1 = builder.createAdd(sgt_reg_reg, eq_reg_reg, "cmp_sum1");
    auto* cmp_sum2 = builder.createAdd(ne_reg_reg, slt_reg_reg, "cmp_sum2");
    auto* cmp_sum3 = builder.createAdd(sle_reg_reg, sge_reg_reg, "cmp_sum3");
    auto* cmp_sum4 = builder.createAdd(cmp_sum1, cmp_sum2, "cmp_sum4");
    auto* cmp_total = builder.createAdd(cmp_sum3, cmp_sum4, "cmp_total");

    // 混合使用立即数操作的结果
    auto* imm_sum1 = builder.createAdd(add_imm_imm, sub_imm_imm, "imm_sum1");
    auto* imm_sum2 = builder.createAdd(mul_imm_imm, div_imm_imm, "imm_sum2");
    auto* imm_sum3 = builder.createAdd(rem_imm_imm, and_imm_imm, "imm_sum3");
    auto* imm_sum4 = builder.createAdd(or_imm_imm, xor_imm_imm, "imm_sum4");
    // auto* imm_sum5 = builder.createAdd(shl_imm_imm, shr_imm_imm, "imm_sum5");
    auto* imm_sum6 = builder.createAdd(sgt_imm_imm, eq_imm_imm, "imm_sum6");
    auto* imm_sum7 = builder.createAdd(ne_imm_imm, slt_imm_imm, "imm_sum7");
    auto* imm_sum8 = builder.createAdd(sle_imm_imm, sge_imm_imm, "imm_sum8");

    auto* imm_total1 = builder.createAdd(imm_sum1, imm_sum2, "imm_total1");
    auto* imm_total2 = builder.createAdd(imm_sum3, imm_sum4, "imm_total2");
    // auto* imm_total3 = builder.createAdd(imm_sum5, imm_sum6, "imm_total3");
    auto* imm_total4 = builder.createAdd(imm_sum7, imm_sum8, "imm_total4");
    auto* imm_total5 = builder.createAdd(imm_total1, imm_total2, "imm_total5");
    // auto* imm_total6 = builder.createAdd(imm_total3, imm_total4,
    // "imm_total6"); auto* imm_total = builder.createAdd(imm_total5,
    // imm_total6, "imm_total");

    // 最终结果：综合所有类型的运算
    // auto* partial1 = builder.createAdd(arith_total, bit_total, "partial1");
    // auto* partial2 = builder.createAdd(cmp_total, imm_total, "partial2");
    // auto* final_result = builder.createAdd(partial1, partial2,
    // "final_result");

    // 额外测试：确保代码涵盖所有分支
    // 测试特殊情况：与0进行运算（测试immToReg中的零寄存器优化）
    auto* zero = midend::ConstantInt::get(i32Type, 0);
    auto* add_with_zero = builder.createAdd(imm_sum1, zero, "add_with_zero");
    auto* mul_with_zero = builder.createMul(arg1, zero, "mul_with_zero");
    auto* test_zero_optimizations =
        builder.createAdd(add_with_zero, mul_with_zero, "test_zero_opt");

    // 最终测试结果
    auto* ultimate_result =
        builder.createAdd(test_zero_optimizations, imm_sum2, "ultimate_result");

    builder.createRet(ultimate_result);
    return module;
}

std::unique_ptr<midend::Module> createComprehensiveUnaryOpsTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module = std::make_unique<midend::Module>("comprehensive_unary_ops",
                                                   context.get());

    auto* i32Type = context->getInt32Type();
    auto* boolType = context->getInt1Type();

    // 创建函数类型: i32 test_all_unary_ops(i32 a, i32 b, i32 c)
    auto* funcType =
        midend::FunctionType::get(i32Type, {i32Type, i32Type, i32Type});
    auto* func =
        midend::Function::Create(funcType, "test_all_unary_ops", module.get());

    // 获取参数
    auto* arg1 = func->getArg(0);
    arg1->setName("a");
    auto* arg2 = func->getArg(1);
    arg2->setName("b");
    auto* arg3 = func->getArg(2);
    arg3->setName("c");

    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 创建常量用于测试立即数操作
    auto* const0 = midend::ConstantInt::get(i32Type, 0);
    auto* const1 = midend::ConstantInt::get(i32Type, 1);
    auto* const5 = midend::ConstantInt::get(i32Type, 5);
    auto* const10 = midend::ConstantInt::get(i32Type, 10);
    auto* const42 = midend::ConstantInt::get(i32Type, 42);
    auto* constNeg7 = midend::ConstantInt::get(i32Type, -7);
    auto* const255 = midend::ConstantInt::get(i32Type, 255);
    auto* constMax =
        midend::ConstantInt::get(i32Type, 2147483647);  // INT32_MAX
    auto* constMin =
        midend::ConstantInt::get(i32Type, -2147483648);  // INT32_MIN

    // 创建布尔常量用于测试 Not 操作
    auto* boolTrue = midend::ConstantInt::getTrue(context.get());
    auto* boolFalse = midend::ConstantInt::getFalse(context.get());

    // === 测试 UAdd (一元加号) ===

    // 1. UAdd 寄存器操作数
    auto* uadd_reg1 = builder.createUAdd(arg1, "uadd_reg1");
    auto* uadd_reg2 = builder.createUAdd(arg2, "uadd_reg2");
    auto* uadd_reg3 = builder.createUAdd(arg3, "uadd_reg3");

    // 2. UAdd 立即数操作数（常量折叠）
    auto* uadd_imm_pos = builder.createUAdd(const42, "uadd_imm_pos");
    auto* uadd_imm_neg = builder.createUAdd(constNeg7, "uadd_imm_neg");
    auto* uadd_imm_zero = builder.createUAdd(const0, "uadd_imm_zero");
    auto* uadd_imm_max = builder.createUAdd(constMax, "uadd_imm_max");
    auto* uadd_imm_min = builder.createUAdd(constMin, "uadd_imm_min");

    // 3. UAdd 复杂表达式的结果
    auto* temp_add = builder.createAdd(arg1, const5, "temp_add");
    auto* uadd_complex = builder.createUAdd(temp_add, "uadd_complex");

    // === 测试 USub (一元减号) ===

    // 4. USub 寄存器操作数
    auto* usub_reg1 = builder.createUSub(arg1, "usub_reg1");
    auto* usub_reg2 = builder.createUSub(arg2, "usub_reg2");
    auto* usub_reg3 = builder.createUSub(arg3, "usub_reg3");

    // 5. USub 立即数操作数（常量折叠）
    auto* usub_imm_pos = builder.createUSub(const42, "usub_imm_pos");  // -42
    auto* usub_imm_neg =
        builder.createUSub(constNeg7, "usub_imm_neg");  // -(-7) = 7
    auto* usub_imm_zero =
        builder.createUSub(const0, "usub_imm_zero");                  // -0 = 0
    auto* usub_imm_one = builder.createUSub(const1, "usub_imm_one");  // -1
    auto* usub_imm_max =
        builder.createUSub(constMax, "usub_imm_max");  // -INT32_MAX

    // 6. USub 复杂表达式的结果
    auto* temp_mul = builder.createMul(arg2, const10, "temp_mul");
    auto* usub_complex = builder.createUSub(temp_mul, "usub_complex");

    // 7. USub 的嵌套使用（双重否定）
    auto* usub_nested =
        builder.createUSub(usub_reg1, "usub_nested");  // -(-arg1) = arg1

    // === 测试 Not (按位取反) ===

    // 8. Not 寄存器操作数
    auto* not_reg1 = builder.createNot(arg1, "not_reg1");
    auto* not_reg2 = builder.createNot(arg2, "not_reg2");
    auto* not_reg3 = builder.createNot(arg3, "not_reg3");

    // 9. Not 立即数操作数（常量折叠）
    auto* not_imm_zero = builder.createNot(const0, "not_imm_zero");   // ~0 = -1
    auto* not_imm_one = builder.createNot(const1, "not_imm_one");     // ~1 = -2
    auto* not_imm_pos = builder.createNot(const42, "not_imm_pos");    // ~42
    auto* not_imm_neg = builder.createNot(constNeg7, "not_imm_neg");  // ~(-7)
    auto* not_imm_255 = builder.createNot(const255, "not_imm_255");   // ~255
    auto* not_imm_max =
        builder.createNot(constMax, "not_imm_max");  // ~INT32_MAX
    auto* not_imm_min =
        builder.createNot(constMin, "not_imm_min");  // ~INT32_MIN

    // 10. Not 复杂表达式的结果
    auto* temp_and = builder.createAnd(arg1, const255, "temp_and");
    auto* not_complex = builder.createNot(temp_and, "not_complex");

    // 11. Not 的嵌套使用（双重取反）
    auto* not_nested =
        builder.createNot(not_reg1, "not_nested");  // ~~arg1 = arg1

    // === 测试边界情况和特殊值 ===

    // 12. 测试零值的特殊处理（immToReg中的零寄存器优化）
    auto* zero_uadd = builder.createUAdd(const0, "zero_uadd");
    auto* zero_usub = builder.createUSub(const0, "zero_usub");
    auto* zero_not = builder.createNot(const0, "zero_not");

    // 13. 测试一元运算符的组合使用
    auto* combo1 = builder.createUSub(uadd_reg1, "combo1");  // -(+arg1) = -arg1
    auto* combo2 = builder.createUAdd(usub_reg2, "combo2");  // +(-arg2) = -arg2
    auto* combo3 = builder.createNot(usub_reg3, "combo3");   // ~(-arg3)
    auto* combo4 = builder.createUSub(not_reg1, "combo4");   // -(~arg1)

    // 14. 一元运算符与二元运算符的混合使用
    auto* mixed1 =
        builder.createAdd(usub_reg1, uadd_reg2, "mixed1");  // (-arg1) + (+arg2)
    auto* mixed2 =
        builder.createMul(not_reg1, const10, "mixed2");  // (~arg1) * 10
    auto* mixed3 =
        builder.createAnd(not_imm_255, arg3, "mixed3");  // (~255) & arg3
    auto* mixed4 =
        builder.createSub(zero_usub, usub_imm_pos, "mixed4");  // (-0) - (-42)

    // 15. 复杂的嵌套一元运算
    auto* complex1 = builder.createNot(
        builder.createUSub(builder.createUAdd(arg1, "inner_uadd"),
                           "inner_usub"),
        "complex1");  // ~(-(+arg1))

    auto* temp_expr = builder.createAdd(arg2, arg3, "temp_expr");
    auto* complex2 =
        builder.createUSub(builder.createNot(temp_expr, "inner_not"),
                           "complex2");  // -(~(arg2 + arg3))

    // === 累积所有结果进行最终计算 ===

    // 累积 UAdd 结果
    auto* uadd_sum1 = builder.createAdd(uadd_reg1, uadd_reg2, "uadd_sum1");
    auto* uadd_sum2 = builder.createAdd(uadd_reg3, uadd_imm_pos, "uadd_sum2");
    auto* uadd_sum3 =
        builder.createAdd(uadd_imm_neg, uadd_imm_zero, "uadd_sum3");
    auto* uadd_sum4 = builder.createAdd(uadd_complex, zero_uadd, "uadd_sum4");
    auto* uadd_total1 = builder.createAdd(uadd_sum1, uadd_sum2, "uadd_total1");
    auto* uadd_total2 = builder.createAdd(uadd_sum3, uadd_sum4, "uadd_total2");
    auto* uadd_total =
        builder.createAdd(uadd_total1, uadd_total2, "uadd_total");

    // 累积 USub 结果
    auto* usub_sum1 = builder.createAdd(usub_reg1, usub_reg2, "usub_sum1");
    auto* usub_sum2 = builder.createAdd(usub_reg3, usub_imm_pos, "usub_sum2");
    auto* usub_sum3 =
        builder.createAdd(usub_imm_neg, usub_imm_zero, "usub_sum3");
    auto* usub_sum4 = builder.createAdd(usub_complex, usub_nested, "usub_sum4");
    auto* usub_sum5 = builder.createAdd(zero_usub, usub_imm_one, "usub_sum5");
    auto* usub_total1 = builder.createAdd(usub_sum1, usub_sum2, "usub_total1");
    auto* usub_total2 = builder.createAdd(usub_sum3, usub_sum4, "usub_total2");
    auto* usub_total3 =
        builder.createAdd(usub_total1, usub_total2, "usub_total3");
    auto* usub_total = builder.createAdd(usub_total3, usub_sum5, "usub_total");

    // 累积 Not 结果
    auto* not_sum1 = builder.createAdd(not_reg1, not_reg2, "not_sum1");
    auto* not_sum2 = builder.createAdd(not_reg3, not_imm_zero, "not_sum2");
    auto* not_sum3 = builder.createAdd(not_imm_one, not_imm_pos, "not_sum3");
    auto* not_sum4 = builder.createAdd(not_complex, not_nested, "not_sum4");
    auto* not_sum5 = builder.createAdd(zero_not, not_imm_255, "not_sum5");
    auto* not_total1 = builder.createAdd(not_sum1, not_sum2, "not_total1");
    auto* not_total2 = builder.createAdd(not_sum3, not_sum4, "not_total2");
    auto* not_total3 = builder.createAdd(not_total1, not_total2, "not_total3");
    auto* not_total = builder.createAdd(not_total3, not_sum5, "not_total");

    // 累积组合和混合结果
    auto* combo_sum1 = builder.createAdd(combo1, combo2, "combo_sum1");
    auto* combo_sum2 = builder.createAdd(combo3, combo4, "combo_sum2");
    auto* combo_total =
        builder.createAdd(combo_sum1, combo_sum2, "combo_total");

    auto* mixed_sum1 = builder.createAdd(mixed1, mixed2, "mixed_sum1");
    auto* mixed_sum2 = builder.createAdd(mixed3, mixed4, "mixed_sum2");
    auto* mixed_total =
        builder.createAdd(mixed_sum1, mixed_sum2, "mixed_total");

    auto* complex_sum = builder.createAdd(complex1, complex2, "complex_sum");

    // 最终综合计算
    auto* partial1 = builder.createAdd(uadd_total, usub_total, "partial1");
    auto* partial2 = builder.createAdd(not_total, combo_total, "partial2");
    auto* partial3 = builder.createAdd(mixed_total, complex_sum, "partial3");
    auto* partial4 = builder.createAdd(partial1, partial2, "partial4");
    auto* final_result = builder.createAdd(partial3, partial4, "final_result");

    // 最后测试：确保覆盖所有优化路径
    // 测试与立即数零的运算（触发零寄存器优化）
    auto* zero_test1 = builder.createAdd(final_result, const0, "zero_test1");
    auto* zero_test2 = builder.createMul(zero_uadd, const0, "zero_test2");
    auto* ultimate_result =
        builder.createAdd(zero_test1, zero_test2, "ultimate_result");

    builder.createRet(ultimate_result);
    return module;
}

std::unique_ptr<midend::Module> createSimpleGlobalConstantTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module = std::make_unique<midend::Module>("simple_global_constant",
                                                   context.get());
    auto* i32Type = context->getInt32Type();

    // 创建全局常量: const int g = 14;
    auto* g = midend::GlobalVariable::Create(
        i32Type,
        true,  // isConstant
        midend::GlobalVariable::ExternalLinkage,
        midend::ConstantInt::get(i32Type, 14), "g", module.get());

    // 创建全局常量: const int N = 10000;
    auto* N = midend::GlobalVariable::Create(
        i32Type,
        true,  // isConstant
        midend::GlobalVariable::ExternalLinkage,
        midend::ConstantInt::get(i32Type, 10000), "N", module.get());

    // 创建测试函数: int test() { return g + N; }
    auto* funcType = midend::FunctionType::get(i32Type, {});
    auto* func = midend::Function::Create(funcType, "test", module.get());
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 加载全局变量值
    auto* gVal = builder.createLoad(g, "g_val");
    auto* NVal = builder.createLoad(N, "N_val");
    auto* result = builder.createAdd(gVal, NVal, "result");
    builder.createRet(result);

    return module;
}

std::unique_ptr<midend::Module> createGlobalArrayTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("global_array", context.get());
    auto* i32Type = context->getInt32Type();

    // 创建全局数组: int small_data[5] = {0, 1, 2, 3, 4};
    auto* smallArrayTy = midend::ArrayType::get(i32Type, 5);
    std::vector<midend::Constant*> smallArrayInit = {
        midend::ConstantInt::get(i32Type, 0),
        midend::ConstantInt::get(i32Type, 1),
        midend::ConstantInt::get(i32Type, 2),
        midend::ConstantInt::get(i32Type, 3),
        midend::ConstantInt::get(i32Type, 4)};
    auto* smallArrayInitializer =
        midend::ConstantArray::get(smallArrayTy, smallArrayInit);
    auto* smallData = midend::GlobalVariable::Create(
        smallArrayTy,
        false,  // not constant
        midend::GlobalVariable::ExternalLinkage, smallArrayInitializer,
        "small_data", module.get());

    // 创建零初始化数组: int zero_data[10] = {0};
    auto* zeroArrayTy = midend::ArrayType::get(i32Type, 10);
    std::vector<midend::Constant*> zeroArrayInit(
        10, midend::ConstantInt::get(i32Type, 0));
    auto* zeroArrayInitializer =
        midend::ConstantArray::get(zeroArrayTy, zeroArrayInit);
    auto* zeroData = midend::GlobalVariable::Create(
        zeroArrayTy, false, midend::GlobalVariable::ExternalLinkage,
        zeroArrayInitializer, "zero_data", module.get());

    // 创建测试函数: int array_sum()
    auto* funcType = midend::FunctionType::get(i32Type, {});
    auto* func = midend::Function::Create(funcType, "array_sum", module.get());

    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    auto* zero = midend::ConstantInt::get(i32Type, 0);
    auto* one = midend::ConstantInt::get(i32Type, 1);

    // 访问 small_data[1]
    auto* smallDataPtr =
        builder.createGEP(smallArrayTy, smallData, {one}, "small_data_1_addr");
    auto* smallDataVal = builder.createLoad(smallDataPtr, "small_data_1");

    // 修改 small_data[2] = 42
    auto* two = midend::ConstantInt::get(i32Type, 2);
    auto* storePtr =
        builder.createGEP(smallArrayTy, smallData, {two}, "small_data_2_addr");
    auto* newVal = midend::ConstantInt::get(i32Type, 42);
    builder.createStore(newVal, storePtr);

    // 访问 zero_data[5]
    auto* five = midend::ConstantInt::get(i32Type, 5);
    auto* zeroDataPtr =
        builder.createGEP(zeroArrayTy, zeroData, {five}, "zero_data_5_addr");
    auto* zeroDataVal = builder.createLoad(zeroDataPtr, "zero_data_5");

    // 返回两个数组元素的和
    auto* result = builder.createAdd(smallDataVal, zeroDataVal, "result");
    builder.createRet(result);

    return module;
}

std::unique_ptr<midend::Module> createGlobalMatrix2DTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("global_matrix_2d", context.get());
    auto* i32Type = context->getInt32Type();

    // 创建全局2D数组: int matrix_data[2][3] = {{1,2,3}, {4,5,6}};
    auto* rowType = midend::ArrayType::get(i32Type, 3);
    auto* matrixTy = midend::ArrayType::get(rowType, 2);

    std::vector<midend::Constant*> row1 = {
        midend::ConstantInt::get(i32Type, 1),
        midend::ConstantInt::get(i32Type, 2),
        midend::ConstantInt::get(i32Type, 3)};
    std::vector<midend::Constant*> row2 = {
        midend::ConstantInt::get(i32Type, 4),
        midend::ConstantInt::get(i32Type, 5),
        midend::ConstantInt::get(i32Type, 6)};
    auto* row1Array = midend::ConstantArray::get(rowType, row1);
    auto* row2Array = midend::ConstantArray::get(rowType, row2);
    std::vector<midend::Constant*> matrixInit = {row1Array, row2Array};
    auto* matrixInitializer = midend::ConstantArray::get(matrixTy, matrixInit);

    auto* matrixData = midend::GlobalVariable::Create(
        matrixTy, false, midend::GlobalVariable::ExternalLinkage,
        matrixInitializer, "matrix_data", module.get());

    // 创建测试函数: int matrix_access()
    auto* funcType = midend::FunctionType::get(i32Type, {});
    auto* func =
        midend::Function::Create(funcType, "matrix_access", module.get());
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    auto* zero = midend::ConstantInt::get(i32Type, 0);
    auto* one = midend::ConstantInt::get(i32Type, 1);
    auto* two = midend::ConstantInt::get(i32Type, 2);

    // 访问 matrix_data[0][1] = 2
    auto* ptr01 =
        builder.createGEP(matrixTy, matrixData, {zero, one}, "matrix_0_1_addr");
    auto* val01 = builder.createLoad(ptr01, "matrix_0_1");

    // 访问 matrix_data[1][2] = 6
    auto* ptr12 =
        builder.createGEP(matrixTy, matrixData, {one, two}, "matrix_1_2_addr");
    auto* val12 = builder.createLoad(ptr12, "matrix_1_2");

    // 修改 matrix_data[0][0] = 100
    auto* ptr00 = builder.createGEP(matrixTy, matrixData, {zero, zero},
                                    "matrix_0_0_addr");
    auto* newVal = midend::ConstantInt::get(i32Type, 100);
    builder.createStore(newVal, ptr00);
    auto* val00 = builder.createLoad(ptr00, "matrix_0_0_new");

    // 返回三个元素的和
    auto* temp = builder.createAdd(val01, val12, "temp");
    auto* result = builder.createAdd(temp, val00, "result");
    builder.createRet(result);

    return module;
}

std::unique_ptr<midend::Module> createGlobal3DArrayTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("global_3d_array", context.get());
    auto* i32Type = context->getInt32Type();

    // 创建全局3D数组: int arr3d[2][3][4]
    auto* arr1DTy = midend::ArrayType::get(i32Type, 4);
    auto* arr2DTy = midend::ArrayType::get(arr1DTy, 3);
    auto* arr3DTy = midend::ArrayType::get(arr2DTy, 2);

    // 简单初始化（所有元素都是其线性索引值）
    std::vector<midend::Constant*> level1;
    for (int k = 0; k < 4; k++) {
        level1.push_back(midend::ConstantInt::get(i32Type, k));
    }
    auto* arr1DConst = midend::ConstantArray::get(arr1DTy, level1);

    std::vector<midend::Constant*> level2;
    for (int j = 0; j < 3; j++) {
        level2.push_back(arr1DConst);
    }
    auto* arr2DConst = midend::ConstantArray::get(arr2DTy, level2);

    std::vector<midend::Constant*> level3;
    for (int i = 0; i < 2; i++) {
        level3.push_back(arr2DConst);
    }
    auto* arr3DConst = midend::ConstantArray::get(arr3DTy, level3);

    auto* arr3D = midend::GlobalVariable::Create(
        arr3DTy, false, midend::GlobalVariable::ExternalLinkage, arr3DConst,
        "arr3d", module.get());

    // 创建测试函数: int test_3d_access()
    auto* funcType = midend::FunctionType::get(i32Type, {});
    auto* func =
        midend::Function::Create(funcType, "test_3d_access", module.get());
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    auto* zero = midend::ConstantInt::get(i32Type, 0);
    auto* one = midend::ConstantInt::get(i32Type, 1);
    auto* two = midend::ConstantInt::get(i32Type, 2);

    // 访问 arr3d[0][1][2]
    auto* gep1 =
        builder.createGEP(arr3DTy, arr3D, {zero, one, two}, "gep_0_1_2");
    auto* val1 = builder.createLoad(gep1, "val_0_1_2");

    // 访问 arr3d[1][0][1]
    auto* gep2 =
        builder.createGEP(arr3DTy, arr3D, {one, zero, one}, "gep_1_0_1");
    auto* val2 = builder.createLoad(gep2, "val_1_0_1");

    // 修改 arr3d[1][2][3] = 999
    auto* three = midend::ConstantInt::get(i32Type, 3);
    auto* gep3 =
        builder.createGEP(arr3DTy, arr3D, {one, two, three}, "gep_1_2_3");
    auto* newVal = midend::ConstantInt::get(i32Type, 999);
    builder.createStore(newVal, gep3);
    auto* val3 = builder.createLoad(gep3, "val_1_2_3_new");

    // 返回三个值的和
    auto* temp = builder.createAdd(val1, val2, "temp");
    auto* result = builder.createAdd(temp, val3, "result");
    builder.createRet(result);

    return module;
}

// std::unique_ptr<midend::Module> createGlobalInitOrderTest() {
//     static auto context = std::make_unique<midend::Context>();
//     auto module =
//         std::make_unique<midend::Module>("global_init_order", context.get());
//     auto* i32Type = context->getInt32Type();

//     // 创建相互依赖的全局变量
//     // int base_value = 10;
//     auto* baseValue = midend::GlobalVariable::Create(
//         i32Type, false, midend::GlobalVariable::ExternalLinkage,
//         midend::ConstantInt::get(i32Type, 10), "base_value", module.get());

//     // const int multiplier = 5;
//     auto* multiplier = midend::GlobalVariable::Create(
//         i32Type, true, midend::GlobalVariable::ExternalLinkage,
//         midend::ConstantInt::get(i32Type, 5), "multiplier", module.get());

//     // int computed_values[5] = {10, 20, 30, 40, 50};
//     auto* arrayTy = midend::ArrayType::get(i32Type, 5);
//     std::vector<midend::Constant*> arrayInit = {
//         midend::ConstantInt::get(i32Type, 10),
//         midend::ConstantInt::get(i32Type, 20),
//         midend::ConstantInt::get(i32Type, 30),
//         midend::ConstantInt::get(i32Type, 40),
//         midend::ConstantInt::get(i32Type, 50)};
//     auto* arrayInitializer = midend::ConstantArray::get(arrayTy, arrayInit);
//     auto* computedValues = midend::GlobalVariable::Create(
//         arrayTy, false, midend::GlobalVariable::ExternalLinkage,
//         arrayInitializer, "computed_values", module.get());

//     // 创建计算函数: int compute_result()
//     auto* funcType = midend::FunctionType::get(i32Type, {});
//     auto* func =
//         midend::Function::Create(funcType, "compute_result", module.get());
//     auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
//     midend::IRBuilder builder(context.get());
//     builder.setInsertPoint(entry);

//     // 使用所有全局变量进行计算
//     auto* baseVal = builder.createLoad(baseValue, "base_val");
//     auto* multVal = builder.createLoad(multiplier, "mult_val");

//     // 修改base_value
//     auto* newBase = builder.createMul(baseVal, multVal, "new_base");
//     builder.createStore(newBase, baseValue);

//     // 初始化累加器
//     auto* zero = midend::ConstantInt::get(i32Type, 0);
//     auto* sum = zero;

//     // 累加数组元素
//     constexpr int ARRAY_SIZE = 5;
//     for (int i = 0; i < ARRAY_SIZE; i++) {
//         auto* idx = midend::ConstantInt::get(i32Type, i);
//         auto* elemPtr = builder.createGEP(arrayTy, computedValues, {idx},
//                                           "elem_" + std::to_string(i) + "_ptr");
//         auto* elemVal =
//             builder.createLoad(elemPtr, "elem_" + std::to_string(i));

//         // 修改数组元素: computed_values[i] *= multiplier
//         auto* newElem = builder.createMul(elemVal, multVal,
//                                           "new_elem_" + std::to_string(i));
//         builder.createStore(newElem, elemPtr);

//         sum = builder.createAdd(sum, newElem, "sum_" + std::to_string(i));
//     }

//     // 添加修改后的base_value
//     auto* finalBase = builder.createLoad(baseValue, "final_base");
//     auto* result = builder.createAdd(sum, finalBase, "result");
//     builder.createRet(result);

//     return module;
// }

std::unique_ptr<midend::Module> createGlobalPointerArithmeticTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module = std::make_unique<midend::Module>("global_pointer_arithmetic",
                                                   context.get());
    auto* i32Type = context->getInt32Type();

    // 创建全局数组用于指针运算
    // int data_array[20] = {0, 1, 2, ..., 19};
    auto* arrayTy = midend::ArrayType::get(i32Type, 20);
    std::vector<midend::Constant*> arrayInit;
    for (int i = 0; i < 20; i++) {
        arrayInit.push_back(midend::ConstantInt::get(i32Type, i));
    }
    auto* arrayInitializer = midend::ConstantArray::get(arrayTy, arrayInit);
    auto* dataArray = midend::GlobalVariable::Create(
        arrayTy, false, midend::GlobalVariable::ExternalLinkage,
        arrayInitializer, "data_array", module.get());

    // 创建测试函数: int pointer_arithmetic_test()
    auto* funcType = midend::FunctionType::get(i32Type, {});
    auto* func = midend::Function::Create(funcType, "pointer_arithmetic_test",
                                          module.get());
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    auto* zero = midend::ConstantInt::get(i32Type, 0);
    auto* five = midend::ConstantInt::get(i32Type, 5);
    auto* ten = midend::ConstantInt::get(i32Type, 10);

    // 获取数组起始地址: &data_array[0]
    auto* basePtr =
        builder.createGEP(arrayTy, dataArray, {zero, zero}, "base_ptr");

    // 指针运算: ptr + 5
    auto* ptr5 = builder.createGEP(i32Type, basePtr, {five}, "ptr_plus_5");
    auto* val5 = builder.createLoad(ptr5, "val_at_5");

    // 指针运算: ptr + 10
    auto* ptr10 = builder.createGEP(i32Type, basePtr, {ten}, "ptr_plus_10");
    auto* val10 = builder.createLoad(ptr10, "val_at_10");

    // 直接索引访问对比: data_array[15]
    auto* fifteen = midend::ConstantInt::get(i32Type, 15);
    auto* directPtr =
        builder.createGEP(arrayTy, dataArray, {fifteen}, "direct_ptr_15");
    auto* val15 = builder.createLoad(directPtr, "val_at_15");

    // 通过指针修改值: *(ptr + 5) = 999
    auto* newVal = midend::ConstantInt::get(i32Type, 999);
    builder.createStore(newVal, ptr5);
    auto* modifiedVal5 = builder.createLoad(ptr5, "modified_val_5");

    // 返回所有值的和
    auto* temp1 = builder.createAdd(modifiedVal5, val10, "temp1");
    auto* result = builder.createAdd(temp1, val15, "result");
    builder.createRet(result);

    return module;
}

std::unique_ptr<midend::Module> createPeepholeOptimizationCoverageTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("peephole_opt", context.get());
    auto* i32 = context->getInt32Type();

    // Function: i32 peephole_opt(i32 a)
    auto* fty = midend::FunctionType::get(i32, {i32});
    auto* fn = midend::Function::Create(fty, "peephole_opt", module.get());
    auto* argA = fn->getArg(0);
    argA->setName("a");

    auto* entry = midend::BasicBlock::Create(context.get(), "entry", fn);
    midend::IRBuilder b(context.get());
    b.setInsertPoint(entry);

    // Constants
    auto* C0 = midend::ConstantInt::get(i32, 0);
    auto* C1 = midend::ConstantInt::get(i32, 1);
    auto* C2 = midend::ConstantInt::get(i32, 2);
    auto* C3 = midend::ConstantInt::get(i32, 3);
    auto* C4 = midend::ConstantInt::get(i32, 4);
    auto* C5 = midend::ConstantInt::get(i32, 5);
    auto* C7 = midend::ConstantInt::get(i32, 7);
    auto* C8 = midend::ConstantInt::get(i32, 8);  // 2^3 for shift
    auto* C9 = midend::ConstantInt::get(i32, 9);  // decomposable (8 + 1)
    auto* Cn1 = midend::ConstantInt::get(i32, -1);

    // 1. Algebraic identity candidates
    auto* add_zero = b.createAdd(argA, C0, "add_zero");    // a + 0
    auto* sub_zero = b.createSub(argA, C0, "sub_zero");    // a - 0
    auto* sub_self = b.createSub(argA, argA, "sub_self");  // a - a
    auto* mul_one = b.createMul(argA, C1, "mul_one");      // a * 1
    auto* mul_zero = b.createMul(argA, C0, "mul_zero");    // a * 0
    auto* div_one = b.createDiv(argA, C1, "div_one");      // a / 1

    // 2. Strength reduction
    auto* mul_pow2 = b.createMul(argA, C8, "mul_pow2");  // a * 8 -> SLLI
    auto* div_pow2 = b.createDiv(
        argA, C8,
        "div_pow2");  // a / 8 -> SRLI (if treated unsigned or simplified)
    auto* mul_9 = b.createMul(
        argA, C9, "mul_9");  // a * 9 -> (a<<3)+a pattern opportunity

    // 3. Bitwise simplifications
    auto* and_self = b.createAnd(argA, argA, "and_self");
    auto* or_self = b.createOr(argA, argA, "or_self");
    auto* xor_self = b.createXor(argA, argA, "xor_self");

    auto* and_zero = b.createAnd(argA, C0, "and_zero");
    auto* or_zero = b.createOr(argA, C0, "or_zero");
    auto* and_allones = b.createAnd(argA, Cn1, "and_allones");
    auto* or_allones = b.createOr(argA, Cn1, "or_allones");
    auto* xor_allones = b.createXor(argA, Cn1, "xor_allones");  // NOT pattern

    // 4. Reassociation & combination
    // const_c1 = (2 + 3) -> folded to 5
    auto* const_c1 = b.createAdd(C2, C3, "const_c1");
    // chain1 = a + const_c1 (R-type ADD with const in reg, convertible to ADDI)
    auto* chain1 = b.createAdd(argA, const_c1, "chain1");
    // chain2 = chain1 + 7  => should reassociate into single ADDI from a if
    // pass handles
    auto* chain2 = b.createAdd(chain1, C7, "chain2");

    // Copy propagation / MOV elimination candidate: add a,0 single use
    auto* mov_like = b.createAdd(argA, C0, "mov_like");
    auto* use_mov = b.createAdd(mov_like, chain2, "use_mov");

    // Another R-type to I-type candidate: ( (a + 4) + 5 )
    auto* a_plus_4 = b.createAdd(argA, C4, "a_plus_4");  // expect ADDI
    auto* plus_chain =
        b.createAdd(a_plus_4, C5, "a_plus_4_plus_5");  // combine constants

    // Pure constant fold chain
    auto* k1 = b.createMul(C2, C3, "k1");  // 6
    auto* k2 = b.createAdd(k1, C7, "k2");  // 13
    auto* k3 = b.createSub(k2, C1, "k3");  // 12
    auto* k4 = b.createAdd(k3, C0, "k4");  // 12 (tests add zero too)

    // Aggregate all to keep them live
    auto* agg1 = b.createAdd(add_zero, sub_zero, "agg1");
    auto* agg2 = b.createAdd(sub_self, mul_one, "agg2");
    auto* agg3 = b.createAdd(mul_zero, div_one, "agg3");
    auto* agg4 = b.createAdd(mul_pow2, div_pow2, "agg4");
    auto* agg5 = b.createAdd(mul_9, and_self, "agg5");
    auto* agg6 = b.createAdd(or_self, xor_self, "agg6");
    auto* agg7 = b.createAdd(and_zero, or_zero, "agg7");
    auto* agg8 = b.createAdd(and_allones, or_allones, "agg8");
    auto* agg9 = b.createAdd(xor_allones, chain2, "agg9");
    auto* agg10 = b.createAdd(use_mov, plus_chain, "agg10");
    auto* agg11 = b.createAdd(k4, agg1, "agg11");

    // Combine aggressively
    auto* sum1 = b.createAdd(agg2, agg3, "sum1");
    auto* sum2 = b.createAdd(agg4, agg5, "sum2");
    auto* sum3 = b.createAdd(agg6, agg7, "sum3");
    auto* sum4 = b.createAdd(agg8, agg9, "sum4");
    auto* sum5 = b.createAdd(agg10, agg11, "sum5");

    auto* t0 = b.createAdd(sum1, sum2, "t0");
    auto* t1 = b.createAdd(sum3, sum4, "t1");
    auto* t2 = b.createAdd(t0, t1, "t2");
    auto* final_sum = b.createAdd(t2, sum5, "final_sum");

    b.createRet(final_sum);
    return module;
}

}  // namespace testcases

// TestRunner 构造函数实现
TestRunner::TestRunner() {
    // 注册所有测试用例
    testCases_["0_simple_return"] = testcases::createSimpleReturnTest;
    testCases_["1_simple_imm_add"] = testcases::createSimpleImmAddTest;
    testCases_["2_simple_add"] = testcases::createSimpleAddTest;
    testCases_["3_arithmetic_ops"] = testcases::createArithmeticOpsTest;
    testCases_["4_conditional_branch"] = testcases::createConditionalBranchTest;
    testCases_["5_variable_assignment"] =
        testcases::createVariableAssignmentTest;
    testCases_["6_complex_assignment"] = testcases::createComplexAssignmentTest;
    testCases_["7_complex_branch"] = testcases::createComplexBranchTest;
    testCases_["8_complex_function_call"] =
        testcases::createComplexFunctionCallTest;
    testCases_["9_small_register_spill"] =
        testcases::createSmallRegisterSpillTest;
    testCases_["10_big_register_spill"] =
        testcases::createLargeRegisterSpillTest;
    testCases_["11_simple_array_1d"] = testcases::createSimpleArray1DTest;
    testCases_["12_simple_array_2d"] = testcases::createSimpleArray2DTest;
    testCases_["13_complex_memory_array"] =
        testcases::createComplexMemoryArrayTest;
    testCases_["14_comprehensive_binary_ops"] =
        testcases::createComprehensiveBinaryOpsTest;
    // 在 TestRunner 构造函数中添加：
    testCases_["15_comprehensive_unary_ops"] =
        testcases::createComprehensiveUnaryOpsTest;
    testCases_["16_simple_global_constant"] =
        testcases::createSimpleGlobalConstantTest;
    testCases_["17_global_array"] = testcases::createGlobalArrayTest;
    testCases_["18_global_matrix_2d"] = testcases::createGlobalMatrix2DTest;
    testCases_["19_global_3d_array"] = testcases::createGlobal3DArrayTest;
    // testCases_["20_global_init_order"] = testcases::createGlobalInitOrderTest;
    testCases_["21_global_pointer_arithmetic"] =
        testcases::createGlobalPointerArithmeticTest;
    // New comprehensive peephole optimization coverage test
    testCases_["22_peephole_opt_coverage"] =
        testcases::createPeepholeOptimizationCoverageTest;
}

std::unique_ptr<midend::Module> TestRunner::loadTestCase(
    const std::string& testCaseName) {
    auto it = testCases_.find(testCaseName);
    if (it == testCases_.end()) {
        std::cerr << "Error: Unknown test case: " << testCaseName << std::endl;
        return nullptr;
    }

    try {
        // 调用对应的测试用例生成器
        return it->second();
    } catch (const std::exception& e) {
        std::cerr << "Error creating test case '" << testCaseName
                  << "': " << e.what() << std::endl;
        return nullptr;
    }
}

void TestRunner::executeCodeGeneration(const std::string& testCaseName,
                                       std::unique_ptr<midend::Module> module) {
    if (!module) {
        std::cerr << "Error: Invalid module for test case: " << testCaseName
                  << std::endl;
        return;
    }

    std::cout << "=== Running Test Case: " << testCaseName
              << " ===" << std::endl;

    // 打印输入的中端IR
    std::cout << "\n--- Input Midend IR ---" << std::endl;
    std::cout << module->toString() << std::endl;
    for (const auto& bb : *module) {
        std::cout << bb->toString() << std::endl;
        for (const auto& inst : *bb) {
            std::cout << "  " << inst->toString() << std::endl;
        }
    }

    std::cout << "\n--- Input Midend IR (pretty printed) ---" << std::endl;
    std::cout << midend::IRPrinter::toString(module.get()) << std::endl;

    try {
        RISCV64Target target;

        // Phase 1: Instruction Selection
        std::cout << "\n--- Phase 1: Instruction Selection ---" << std::endl;
        auto riscvModule = target.instructionSelectionPass(*module);
        std::cout << "\n[After Instruction Selection]" << std::endl;
        std::cout << riscvModule.toString() << std::endl;

        // Phase 0.5: Value Reuse (optional, pass nullptr AnalysisManager here)
        // std::cout << "\n--- Phase 0.5: Value Reuse (optional) ---" <<
        // std::endl; target.valueReusePass(riscvModule, *module, nullptr);
        // std::cout << "\n[After Value Reuse]" << std::endl;
        // std::cout << riscvModule.toString() << std::endl;

        // Phase 1.5: Initial Frame Index
        std::cout << "\n--- Phase 1.5: Initial Frame Index Creation ---"
                  << std::endl;
        target.initialFrameIndexPass(riscvModule);
        std::cout << "\n[After Initial Frame Index]" << std::endl;
        std::cout << riscvModule.toString() << std::endl;

        // Phase 1.6: Constant Folding
        std::cout << "\n--- Phase 1.6: Constant Folding ---" << std::endl;
        target.constantFoldingPass(riscvModule);
        std::cout << "\n[After Constant Folding]" << std::endl;
        std::cout << riscvModule.toString() << std::endl;

        // Phase 1.7: Basic Block Reordering
        std::cout << "\n--- Phase 1.7: Basic Block Reordering ---" << std::endl;
        target.basicBlockReorderingPass(riscvModule);
        std::cout << "\n[After Basic Block Reordering]" << std::endl;
        std::cout << riscvModule.toString() << std::endl;

        // (Optional experimental) Greedy RA path was here in old runner:
        // target.RAGreedyPass(riscvModule);

        // Phase 2: Register Allocation
        std::cout << "\n--- Phase 2: Register Allocation ---" << std::endl;
        auto& allocatedModule = target.registerAllocationPass(riscvModule);
        std::cout << "\n[After Register Allocation]" << std::endl;
        std::cout << allocatedModule.toString() << std::endl;

        // Phase 3: Frame Index Elimination
        std::cout << "\n--- Phase 3: Frame Index Elimination ---" << std::endl;
        try {
            target.frameIndexEliminationPass(allocatedModule);
            std::cout << "\n[After Frame Index Elimination]" << std::endl;
            std::cout << allocatedModule.toString() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Frame index elimination pass failed: " << e.what()
                      << std::endl;
        }

        // Final assembly (string form)
        std::cout << "\n--- Final Assembly Output (string) ---" << std::endl;
        std::cout << allocatedModule.toString() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during code generation: " << e.what() << std::endl;
    }

    std::cout << "\n=== Test Case " << testCaseName
              << " Completed ===" << std::endl;
    std::cout << std::string(60, '=') << std::endl << std::endl;
}

bool TestRunner::runTestCase(const std::string& testCaseName) {
    auto module = loadTestCase(testCaseName);
    if (!module) {
        return false;
    }

    executeCodeGeneration(testCaseName, std::move(module));
    return true;
}

void TestRunner::runAllTests() {
    auto testCases = getAvailableTestCases();

    if (testCases.empty()) {
        std::cout << "No test cases found in directory: " << testCaseDir
                  << std::endl;
        return;
    }

    std::cout << "Running all RISC-V code generation tests..." << std::endl;
    std::cout << "Found " << testCases.size() << " test case(s)" << std::endl
              << std::endl;

    int successCount = 0;
    int totalCount = testCases.size();

    for (const auto& testCase : testCases) {
        if (runTestCase(testCase)) {
            successCount++;
        }
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Test Summary: " << successCount << "/" << totalCount
              << " tests passed" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

void TestRunner::listTestCases() {
    auto testCases = getAvailableTestCases();

    std::cout << "Available test cases:" << std::endl;
    if (testCases.empty()) {
        std::cout << "  No test cases found in directory: " << testCaseDir
                  << std::endl;
    } else {
        for (size_t i = 0; i < testCases.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << testCases[i] << std::endl;
        }
    }
}

}  // namespace riscv64::test

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [option] [test_case_name]"
              << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help       Show this help message" << std::endl;
    std::cout << "  -l, --list       List all available test cases"
              << std::endl;
    std::cout << "  -a, --all        Run all test cases" << std::endl;
    std::cout << "  [test_case_name] Run a specific test case" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " --list" << std::endl;
    std::cout << "  " << programName << " --all" << std::endl;
    std::cout << "  " << programName << " simple_return" << std::endl;
}

int main(int argc, char* argv[]) {
    riscv64::test::TestRunner runner;

    if (argc == 1) {
        // 默认运行所有测试
        runner.runAllTests();
        return 0;
    }

    std::string option = argv[1];

    if (option == "-h" || option == "--help") {
        printUsage(argv[0]);
        return 0;
    } else if (option == "-l" || option == "--list") {
        runner.listTestCases();
        return 0;
    } else if (option == "-a" || option == "--all") {
        runner.runAllTests();
        return 0;
    } else {
        // 运行指定的测试用例
        std::string testCaseName = option;
        if (!runner.runTestCase(testCaseName)) {
            std::cerr << "Failed to run test case: " << testCaseName
                      << std::endl;
            return 1;
        }
        return 0;
    }
}
