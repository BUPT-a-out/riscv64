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
    auto* noAdjustBB = midend::BasicBlock::Create(context.get(), "no_adjust", func);
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
    auto* isGreaterThan50 = builder.createICmpSGT(finalPhi, fifty_threshold, "is_gt_50");
    builder.createCondBr(isGreaterThan50, adjustBB, noAdjustBB);

    // adjust块: 结果 > 50，减去 10
    builder.setInsertPoint(adjustBB);
    auto* minusTen = midend::ConstantInt::get(i32Type, -10);
    auto* adjustedResult1 = builder.createAdd(finalPhi, minusTen, "adjusted_result1");
    builder.createBr(exitBB);

    // no_adjust块: 结果 <= 50，加上 5
    builder.setInsertPoint(noAdjustBB);
    auto* five = midend::ConstantInt::get(i32Type, 5);
    auto* adjustedResult2 = builder.createAdd(finalPhi, five, "adjusted_result2");
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
    auto module =
        std::make_unique<midend::Module>("complex_function_call", context.get());

    auto* i32Type = context->getInt32Type();

    // 创建辅助函数 1: i32 add_three(i32 a, i32 b, i32 c)
    auto* addThreeFuncType = midend::FunctionType::get(i32Type, {i32Type, i32Type, i32Type});
    auto* addThreeFunc = midend::Function::Create(addThreeFuncType, "add_three", module.get());
    
    auto* addThreeArg1 = addThreeFunc->getArg(0);
    auto* addThreeArg2 = addThreeFunc->getArg(1);
    auto* addThreeArg3 = addThreeFunc->getArg(2);
    addThreeArg1->setName("a");
    addThreeArg2->setName("b");
    addThreeArg3->setName("c");

    auto* addThreeEntry = midend::BasicBlock::Create(context.get(), "entry", addThreeFunc);
    midend::IRBuilder addThreeBuilder(context.get());
    addThreeBuilder.setInsertPoint(addThreeEntry);

    auto* temp1 = addThreeBuilder.createAdd(addThreeArg1, addThreeArg2, "temp1");
    auto* result1 = addThreeBuilder.createAdd(temp1, addThreeArg3, "result");
    addThreeBuilder.createRet(result1);

    // 创建辅助函数 2: i32 multiply_by_two(i32 x)
    auto* multiplyFuncType = midend::FunctionType::get(i32Type, {i32Type});
    auto* multiplyFunc = midend::Function::Create(multiplyFuncType, "multiply_by_two", module.get());
    
    auto* multiplyArg = multiplyFunc->getArg(0);
    multiplyArg->setName("x");

    auto* multiplyEntry = midend::BasicBlock::Create(context.get(), "entry", multiplyFunc);
    midend::IRBuilder multiplyBuilder(context.get());
    multiplyBuilder.setInsertPoint(multiplyEntry);

    auto* two = midend::ConstantInt::get(i32Type, 2);
    auto* result2 = multiplyBuilder.createMul(multiplyArg, two, "result");
    multiplyBuilder.createRet(result2);

    // 创建辅助函数 3: i32 compute_formula(i32 a, i32 b)
    auto* formulaFuncType = midend::FunctionType::get(i32Type, {i32Type, i32Type});
    auto* formulaFunc = midend::Function::Create(formulaFuncType, "compute_formula", module.get());
    
    auto* formulaArg1 = formulaFunc->getArg(0);
    auto* formulaArg2 = formulaFunc->getArg(1);
    formulaArg1->setName("a");
    formulaArg2->setName("b");

    auto* formulaEntry = midend::BasicBlock::Create(context.get(), "entry", formulaFunc);
    midend::IRBuilder formulaBuilder(context.get());
    formulaBuilder.setInsertPoint(formulaEntry);

    // 在 compute_formula 中调用 multiply_by_two
    std::vector<midend::Value*> multiplyArgs = {formulaArg1};
    auto* doubledA = formulaBuilder.createCall(multiplyFunc, multiplyArgs, "doubled_a");
    
    // 计算 doubled_a + b * 3
    auto* three = midend::ConstantInt::get(i32Type, 3);
    auto* bTimesThree = formulaBuilder.createMul(formulaArg2, three, "b_times_3");
    auto* formulaResult = formulaBuilder.createAdd(doubledA, bTimesThree, "formula_result");
    formulaBuilder.createRet(formulaResult);

    // 创建主函数: i32 main(i32 x, i32 y, i32 z)
    auto* mainFuncType = midend::FunctionType::get(i32Type, {i32Type, i32Type, i32Type});
    auto* mainFunc = midend::Function::Create(mainFuncType, "main", module.get());

    auto* mainArg1 = mainFunc->getArg(0);
    auto* mainArg2 = mainFunc->getArg(1);
    auto* mainArg3 = mainFunc->getArg(2);
    mainArg1->setName("x");
    mainArg2->setName("y");
    mainArg3->setName("z");

    auto* mainEntry = midend::BasicBlock::Create(context.get(), "entry", mainFunc);
    auto* condBB = midend::BasicBlock::Create(context.get(), "condition", mainFunc);
    auto* thenBB = midend::BasicBlock::Create(context.get(), "then", mainFunc);
    auto* elseBB = midend::BasicBlock::Create(context.get(), "else", mainFunc);
    auto* mergeBB = midend::BasicBlock::Create(context.get(), "merge", mainFunc);

    midend::IRBuilder mainBuilder(context.get());

    // entry块: 调用 add_three(x, y, z)
    mainBuilder.setInsertPoint(mainEntry);
    std::vector<midend::Value*> addThreeArgs = {mainArg1, mainArg2, mainArg3};
    auto* sumResult = mainBuilder.createCall(addThreeFunc, addThreeArgs, "sum_result");
    mainBuilder.createBr(condBB);

    // condition块: 检查 sum_result > 20
    mainBuilder.setInsertPoint(condBB);
    auto* twenty = midend::ConstantInt::get(i32Type, 20);
    auto* cmp = mainBuilder.createICmpSGT(sumResult, twenty, "cmp");
    mainBuilder.createCondBr(cmp, thenBB, elseBB);

    // then块: 调用 compute_formula(sum_result, x)
    mainBuilder.setInsertPoint(thenBB);
    std::vector<midend::Value*> formulaArgs1 = {sumResult, mainArg1};
    auto* thenResult = mainBuilder.createCall(formulaFunc, formulaArgs1, "then_result");
    mainBuilder.createBr(mergeBB);

    // else块: 调用 multiply_by_two(sum_result) 然后再调用 compute_formula
    mainBuilder.setInsertPoint(elseBB);
    std::vector<midend::Value*> multiplyArgs2 = {sumResult};
    auto* doubledSum = mainBuilder.createCall(multiplyFunc, multiplyArgs2, "doubled_sum");
    
    // 再调用 compute_formula(doubled_sum, y + z)
    auto* yzSum = mainBuilder.createAdd(mainArg2, mainArg3, "yz_sum");
    std::vector<midend::Value*> formulaArgs2 = {doubledSum, yzSum};
    auto* elseResult = mainBuilder.createCall(formulaFunc, formulaArgs2, "else_result");
    mainBuilder.createBr(mergeBB);

    // merge块: 使用 phi 节点合并结果，然后进行最终计算
    mainBuilder.setInsertPoint(mergeBB);
    auto* phi = mainBuilder.createPHI(i32Type, "phi_result");
    phi->addIncoming(thenResult, thenBB);
    phi->addIncoming(elseResult, elseBB);

    // 最终调用: multiply_by_two(phi_result) + add_three(10, 20, 30)
    std::vector<midend::Value*> finalMultiplyArgs = {phi};
    auto* finalDoubled = mainBuilder.createCall(multiplyFunc, finalMultiplyArgs, "final_doubled");

    auto* ten = midend::ConstantInt::get(i32Type, 10);
    auto* thirty = midend::ConstantInt::get(i32Type, 30);
    std::vector<midend::Value*> finalAddArgs = {ten, twenty, thirty};
    auto* constantSum = mainBuilder.createCall(addThreeFunc, finalAddArgs, "constant_sum");

    auto* finalResult = mainBuilder.createAdd(finalDoubled, constantSum, "final_result");
    mainBuilder.createRet(finalResult);

    return module;
}

std::unique_ptr<midend::Module> createLargeRegisterSpillTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module = std::make_unique<midend::Module>("register_spill_test", context.get());

    auto* i32Type = context->getInt32Type();

    // 创建一个会导致寄存器溢出的复杂函数
    // i32 complex_computation(i32 a, i32 b, i32 c, i32 d, i32 e, i32 f, i32 g, i32 h)
    auto* funcType = midend::FunctionType::get(i32Type, {
        i32Type, i32Type, i32Type, i32Type, 
        i32Type, i32Type, i32Type, i32Type
    });
    auto* func = midend::Function::Create(funcType, "complex_computation", module.get());

    // 设置参数名称
    auto* arg1 = func->getArg(0); arg1->setName("a");
    auto* arg2 = func->getArg(1); arg2->setName("b");
    auto* arg3 = func->getArg(2); arg3->setName("c");
    auto* arg4 = func->getArg(3); arg4->setName("d");
    auto* arg5 = func->getArg(4); arg5->setName("e");
    auto* arg6 = func->getArg(5); arg6->setName("f");
    auto* arg7 = func->getArg(6); arg7->setName("g");
    auto* arg8 = func->getArg(7); arg8->setName("h");

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
    auto module = std::make_unique<midend::Module>("register_spill_test", context.get());

    auto* i32Type = context->getInt32Type();

    // 创建一个在RISC-V64上会导致轻微寄存器溢出的函数
    // i32 riscv_spill_test(i32 a, i32 b, i32 c, i32 d, i32 e, i32 f, i32 g, i32 h)
    auto* funcType = midend::FunctionType::get(i32Type, {
        i32Type, i32Type, i32Type, i32Type, 
        i32Type, i32Type, i32Type, i32Type
    });
    auto* func = midend::Function::Create(funcType, "riscv_spill_test", module.get());

    // 设置参数名称
    auto* arg1 = func->getArg(0); arg1->setName("a");
    auto* arg2 = func->getArg(1); arg2->setName("b");
    auto* arg3 = func->getArg(2); arg3->setName("c");
    auto* arg4 = func->getArg(3); arg4->setName("d");
    auto* arg5 = func->getArg(4); arg5->setName("e");
    auto* arg6 = func->getArg(5); arg6->setName("f");
    auto* arg7 = func->getArg(6); arg7->setName("g");
    auto* arg8 = func->getArg(7); arg8->setName("h");

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
    auto module = std::make_unique<midend::Module>("simple_array_1d", context.get());

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
    auto* val0 = midend::ConstantInt::get(i32Type, 10);
    auto* ptr0 = builder.createGEP(arrayType, arrayAlloca, {zero, idx0}, "arr_0_ptr");
    builder.createStore(val0, ptr0);

    // arr[1] = 20
    auto* idx1 = midend::ConstantInt::get(i32Type, 1);
    auto* val1 = midend::ConstantInt::get(i32Type, 20);
    auto* ptr1 = builder.createGEP(arrayType, arrayAlloca, {zero, idx1}, "arr_1_ptr");
    builder.createStore(val1, ptr1);

    // arr[2] = 30
    auto* idx2 = midend::ConstantInt::get(i32Type, 2);
    auto* val2 = midend::ConstantInt::get(i32Type, 30);
    auto* ptr2 = builder.createGEP(arrayType, arrayAlloca, {zero, idx2}, "arr_2_ptr");
    builder.createStore(val2, ptr2);

    // arr[3] = 40
    auto* idx3 = midend::ConstantInt::get(i32Type, 3);
    auto* val3 = midend::ConstantInt::get(i32Type, 40);
    auto* ptr3 = builder.createGEP(arrayType, arrayAlloca, {zero, idx3}, "arr_3_ptr");
    builder.createStore(val3, ptr3);

    // arr[4] = 50
    auto* idx4 = midend::ConstantInt::get(i32Type, 4);
    auto* val4 = midend::ConstantInt::get(i32Type, 50);
    auto* ptr4 = builder.createGEP(arrayType, arrayAlloca, {zero, idx4}, "arr_4_ptr");
    builder.createStore(val4, ptr4);

    // 读取并累加数组元素
    // sum += arr[0]
    auto* currentSum = builder.createLoad(sumAlloca, "current_sum");
    auto* elem0 = builder.createLoad(ptr0, "elem_0");
    auto* newSum = builder.createAdd(currentSum, elem0, "sum_1");
    builder.createStore(newSum, sumAlloca);

    // sum += arr[1]
    currentSum = builder.createLoad(sumAlloca, "current_sum_2");
    auto* elem1 = builder.createLoad(ptr1, "elem_1");
    newSum = builder.createAdd(currentSum, elem1, "sum_2");
    builder.createStore(newSum, sumAlloca);

    // sum += arr[2]
    currentSum = builder.createLoad(sumAlloca, "current_sum_3");
    auto* elem2 = builder.createLoad(ptr2, "elem_2");
    newSum = builder.createAdd(currentSum, elem2, "sum_3");
    builder.createStore(newSum, sumAlloca);

    // sum += arr[3]
    currentSum = builder.createLoad(sumAlloca, "current_sum_4");
    auto* elem3 = builder.createLoad(ptr3, "elem_3");
    newSum = builder.createAdd(currentSum, elem3, "sum_4");
    builder.createStore(newSum, sumAlloca);

    // sum += arr[4]
    currentSum = builder.createLoad(sumAlloca, "current_sum_5");
    auto* elem4 = builder.createLoad(ptr4, "elem_4");
    auto* finalSum = builder.createAdd(currentSum, elem4, "final_sum");

    // 返回总和
    builder.createRet(finalSum);

    return module;
}

std::unique_ptr<midend::Module> createSimpleArray2DTest() {
    static auto context = std::make_unique<midend::Context>();
    auto module = std::make_unique<midend::Module>("simple_array_2d", context.get());

    auto* i32Type = context->getInt32Type();
    // 创建 3x3 的二维数组类型: [3 x [3 x i32]]
    auto* innerArrayType = midend::ArrayType::get(i32Type, 3);
    auto* outerArrayType = midend::ArrayType::get(innerArrayType, 3);
    
    // 创建函数类型: i32 matrix_diagonal_sum()
    auto* funcType = midend::FunctionType::get(i32Type, {});
    auto* func = midend::Function::Create(funcType, "matrix_diagonal_sum", module.get());

    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 分配二维数组: %matrix = alloca [3 x [3 x i32]]
    auto* matrixAlloca = builder.createAlloca(outerArrayType, nullptr, "matrix");
    
    // 分配对角线和变量: %diag_sum = alloca i32
    auto* diagSumAlloca = builder.createAlloca(i32Type, nullptr, "diag_sum");
    
    // 初始化对角线和为0
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
    auto* ptr00 = builder.createGEP(outerArrayType, matrixAlloca, {zero, idx0, idx0}, "matrix_0_0_ptr");
    builder.createStore(val1, ptr00);

    // matrix[0][1] = 2
    auto* ptr01 = builder.createGEP(outerArrayType, matrixAlloca, {zero, idx0, idx1}, "matrix_0_1_ptr");
    builder.createStore(val2, ptr01);

    // matrix[1][1] = 5 (对角线元素)
    auto* ptr11 = builder.createGEP(outerArrayType, matrixAlloca, {zero, idx1, idx1}, "matrix_1_1_ptr");
    builder.createStore(val5, ptr11);

    // matrix[1][0] = 3
    auto* ptr10 = builder.createGEP(outerArrayType, matrixAlloca, {zero, idx1, idx0}, "matrix_1_0_ptr");
    builder.createStore(val3, ptr10);

    // matrix[2][2] = 9 (对角线元素)
    auto* ptr22 = builder.createGEP(outerArrayType, matrixAlloca, {zero, idx2, idx2}, "matrix_2_2_ptr");
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
    auto module = std::make_unique<midend::Module>("complex_memory_array", context.get());

    auto* i32Type = context->getInt32Type();
    auto* arrayType = midend::ArrayType::get(i32Type, 8);  // int arr[8]
    
    // 创建函数类型: i32 complex_array_ops(i32 n)
    auto* funcType = midend::FunctionType::get(i32Type, {i32Type});
    auto* func = midend::Function::Create(funcType, "complex_array_ops", module.get());

    auto* arg = func->getArg(0);
    arg->setName("n");

    // 创建基本块
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);
    auto* initLoopBB = midend::BasicBlock::Create(context.get(), "init_loop", func);
    auto* initCondBB = midend::BasicBlock::Create(context.get(), "init_cond", func);
    auto* processLoopBB = midend::BasicBlock::Create(context.get(), "process_loop", func);
    auto* processCondBB = midend::BasicBlock::Create(context.get(), "process_cond", func);
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
    auto* elemPtr = builder.createGEP(arrayType, arrayAlloca, {zero, currentI}, "elem_ptr");
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
    auto* afterLoopBB = midend::BasicBlock::Create(context.get(), "after_loop", func);
    builder.createCondBr(processCond, afterLoopBB, finalBB);

    builder.setInsertPoint(afterLoopBB);
    currentI = builder.createLoad(iAlloca, "i_in_process");
    
    // 加载当前元素 arr[i]
    auto* currentElemPtr = builder.createGEP(arrayType, arrayAlloca, {zero, currentI}, "current_elem_ptr");
    auto* currentElem = builder.createLoad(currentElemPtr, "current_elem");
    
    // 更新sum: sum += arr[i]
    auto* currentSum = builder.createLoad(sumAlloca, "current_sum");
    auto* newSum = builder.createAdd(currentSum, currentElem, "new_sum");
    builder.createStore(newSum, sumAlloca);
    
    // 更新最大值
    auto* currentMax = builder.createLoad(maxAlloca, "current_max");
    auto* isGreater = builder.createICmpSGT(currentElem, currentMax, "is_greater");
    auto* updateMaxBB = midend::BasicBlock::Create(context.get(), "update_max", func);
    auto* keepMaxBB = midend::BasicBlock::Create(context.get(), "keep_max", func);
    auto* afterMaxBB = midend::BasicBlock::Create(context.get(), "after_max", func);
    
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
    auto* afterEvenOddBB = midend::BasicBlock::Create(context.get(), "after_even_odd", func);
    
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
    auto* afterSwapBB = midend::BasicBlock::Create(context.get(), "after_swap", func);
    
    builder.createCondBr(hasPrevoius, swapBB, noSwapBB);
    
    // 交换逻辑
    builder.setInsertPoint(swapBB);
    auto* prevElemPtr = builder.createGEP(arrayType, arrayAlloca, {zero, iPrevious}, "prev_elem_ptr");
    auto* currentElemForSwap = builder.createLoad(currentElemPtr, "current_for_swap");
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
    auto* intermediate1 = builder.createAdd(finalSum, finalMax, "intermediate1");
    auto* intermediate2 = builder.createAdd(intermediate1, tempVal, "intermediate2");
    auto* result = builder.createSub(intermediate2, arg, "result");

    builder.createRet(result);

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
    testCases_["5_variable_assignment"] = testcases::createVariableAssignmentTest;
    testCases_["6_complex_assignment"] = testcases::createComplexAssignmentTest;
    testCases_["7_complex_branch"] = testcases::createComplexBranchTest;
    testCases_["8_complex_function_call"] = testcases::createComplexFunctionCallTest;
    testCases_["9_small_register_spill"] = testcases::createSmallRegisterSpillTest;
    testCases_["10_big_register_spill"] = testcases::createLargeRegisterSpillTest;
    testCases_["11_simple_array_1d"] = testcases::createSimpleArray1DTest;
    testCases_["12_simple_array_2d"] = testcases::createSimpleArray2DTest;
    testCases_["13_complex_memory_array"] = testcases::createComplexMemoryArrayTest;

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
        // 创建RISC-V目标
        RISCV64Target target;

        // 执行指令选择pass
        std::cout << "\n--- Running Instruction Selection Pass ---"
                  << std::endl;
        auto riscvModule = target.instructionSelectionPass(*module);

        // 打印生成的RISC-V汇编代码
        std::cout
            << "\n--- Generated RISC-V Assembly (with virtual registers) ---"
            << std::endl;
        std::cout << riscvModule.toString() << std::endl;

        // 执行寄存器分配pass
        std::cout << "\n--- Running Register Allocation Pass ---" << std::endl;
        auto& allocatedModule = target.registerAllocationPass(riscvModule);

        // 打印寄存器分配后的代码
        std::cout << "\n--- Final RISC-V Assembly (with physical registers) ---"
                  << std::endl;
        std::cout << allocatedModule.toString() << std::endl;

        // 可选：生成最终的汇编文本
        // auto assembly = target.compileToAssembly(*module);
        // std::cout << "\n--- Final Assembly Output ---" << std::endl;
        // // for (const auto& line : assemblyLines) {
        // //     std::cout << line << std::endl;
        // // }
        // std::cout << assembly << std::endl;

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
