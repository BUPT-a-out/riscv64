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

}  // namespace testcases

// TestRunner 构造函数实现
TestRunner::TestRunner() {
    // 注册所有测试用例
    testCases_["0_simple_return"] = testcases::createSimpleReturnTest;
    testCases_["1_simple_imm_add"] = testcases::createSimpleImmAddTest;
    testCases_["2_simple_add"] = testcases::createSimpleAddTest;
    testCases_["3_arithmetic_ops"] = testcases::createArithmeticOpsTest;
    testCases_["4_conditional_branch"] = testcases::createConditionalBranchTest;
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
        auto allocatedModule = target.registerAllocationPass(riscvModule);

        // 打印寄存器分配后的代码
        std::cout << "\n--- Final RISC-V Assembly (with physical registers) ---"
                  << std::endl;
        std::cout << allocatedModule.toString() << std::endl;

        // 可选：生成最终的汇编文本
        auto assembly = target.compileToAssembly(*module);
        std::cout << "\n--- Final Assembly Output ---" << std::endl;
        // for (const auto& line : assemblyLines) {
        //     std::cout << line << std::endl;
        // }
        std::cout << assembly << std::endl;

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
