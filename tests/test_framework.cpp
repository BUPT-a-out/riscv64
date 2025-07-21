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

class CodeGenTestRunner {
   private:
    // 注册的测试用例映射
    std::map<std::string, TestCaseGenerator> testCases_;

    // 创建简单返回常量的测试用例
    static auto createSimpleReturnTest() -> std::unique_ptr<midend::Module>;

    // 创建简单加法的测试用例
    static auto createSimpleAddTest() -> std::unique_ptr<midend::Module>;

    // 创建多个算术运算的测试用例
    static auto createArithmeticOpsTest() -> std::unique_ptr<midend::Module>;

    // 创建条件分支的测试用例
    static auto createConditionalBranchTest()
        -> std::unique_ptr<midend::Module>;

   public:
    CodeGenTestRunner();
    ~CodeGenTestRunner() = default;

    // 运行单个测试用例
    bool runTest(const std::string& testName);

    // 运行所有测试用例
    void runAllTests();

    // 列出所有可用的测试用例
    void listTestCases() const;

   private:
    // 执行代码生成并打印结果
    void executeCodeGeneration(const std::string& testName,
                               std::unique_ptr<midend::Module> module);
};

// 修复 createSimpleReturnTest 方法

auto CodeGenTestRunner::createSimpleReturnTest()
    -> std::unique_ptr<midend::Module> {
    // 注意：Context 必须比 Module 存活更久
    // 我们需要确保 Context 的生命周期管理正确
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

auto CodeGenTestRunner::createSimpleAddTest()
    -> std::unique_ptr<midend::Module> {
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

auto CodeGenTestRunner::createArithmeticOpsTest()
    -> std::unique_ptr<midend::Module> {
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

auto CodeGenTestRunner::createConditionalBranchTest()
    -> std::unique_ptr<midend::Module> {
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

CodeGenTestRunner::CodeGenTestRunner() {
    // 注册所有测试用例
    testCases_["simple_return"] = createSimpleReturnTest;
    testCases_["simple_add"] = createSimpleAddTest;
    testCases_["arithmetic_ops"] = createArithmeticOpsTest;
    testCases_["conditional_branch"] = createConditionalBranchTest;
}

void CodeGenTestRunner::executeCodeGeneration(
    const std::string& testName, std::unique_ptr<midend::Module> module) {
    if (!module) {
        std::cerr << "Error: Invalid module for test case: " << testName
                  << std::endl;
        return;
    }

    constexpr int SEPARATOR_LENGTH = 60;

    std::cout << "=== Running Test Case: " << testName << " ===" << std::endl;

    // 打印输入的中端IR
    std::cout << "\n--- Input Midend IR ---" << std::endl;
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

        // 执行寄存器分配pass（如果实现了的话）
        try {
            std::cout << "\n--- Running Register Allocation Pass ---"
                      << std::endl;
            auto& allocatedModule = target.registerAllocationPass(riscvModule);

            // 打印寄存器分配后的代码
            std::cout
                << "\n--- Final RISC-V Assembly (with physical registers) ---"
                << std::endl;
            std::cout << allocatedModule.toString() << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Register allocation not implemented or failed: "
                      << e.what() << std::endl;
        }

        // 生成最终的汇编文本（如果实现了的话）
        try {
            auto assembly = target.compileToAssembly(*module);
            std::cout << "\n--- Final Assembly Output ---" << std::endl;
            std::cout << assembly << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Assembly generation not implemented or failed: "
                      << e.what() << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error during code generation: " << e.what() << std::endl;
    }

    std::cout << "\n=== Test Case " << testName
              << " Completed ===" << std::endl;
    std::cout << std::string(SEPARATOR_LENGTH, '=') << std::endl << std::endl;
}

bool CodeGenTestRunner::runTest(const std::string& testName) {
    auto it = testCases_.find(testName);
    if (it == testCases_.end()) {
        std::cerr << "Error: Unknown test case: " << testName << std::endl;
        return false;
    }

    try {
        auto module = it->second();
        executeCodeGeneration(testName, std::move(module));
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error creating test module: " << e.what() << std::endl;
        return false;
    }
}

void CodeGenTestRunner::runAllTests() {
    std::cout << "Running all RISC-V code generation tests..." << std::endl;
    std::cout << "Found " << testCases_.size() << " test case(s)" << std::endl
              << std::endl;

    int successCount = 0;
    const int totalCount = static_cast<int>(testCases_.size());

    for (const auto& [testName, generator] : testCases_) {
        if (runTest(testName)) {
            successCount++;
        }
    }

    constexpr int SEPARATOR_LENGTH = 60;
    std::cout << "\n" << std::string(SEPARATOR_LENGTH, '=') << std::endl;
    std::cout << "Test Summary: " << successCount << "/" << totalCount
              << " tests passed" << std::endl;
    std::cout << std::string(SEPARATOR_LENGTH, '=') << std::endl;
}

void CodeGenTestRunner::listTestCases() const {
    std::cout << "Available test cases:" << std::endl;
    if (testCases_.empty()) {
        std::cout << "  No test cases available" << std::endl;
    } else {
        int index = 1;
        for (const auto& [testName, generator] : testCases_) {
            std::cout << "  " << index << ". " << testName << std::endl;
            index++;
        }
    }
}

}  // namespace riscv64::test

void printUsage(const char* programName) {
    std::cout << "RISC-V Code Generation Test Runner" << std::endl;
    std::cout << "Usage: " << programName << " [option] [test_case_name]"
              << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help       Show this help message" << std::endl;
    std::cout << "  -l, --list       List all available test cases"
              << std::endl;
    std::cout << "  -a, --all        Run all test cases (default)" << std::endl;
    std::cout << "  [test_case_name] Run a specific test case" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " --list" << std::endl;
    std::cout << "  " << programName << " --all" << std::endl;
    std::cout << "  " << programName << " simple_return" << std::endl;
    std::cout << "  " << programName << " simple_add" << std::endl;
}

auto main(int argc, char* argv[]) -> int {
    riscv64::test::CodeGenTestRunner runner;

    if (argc == 1) {
        // 默认运行所有测试
        runner.runAllTests();
        return 0;
    }

    const std::string option = argv[1];

    if (option == "-h" || option == "--help") {
        printUsage(argv[0]);
        return 0;
    }

    if (option == "-l" || option == "--list") {
        runner.listTestCases();
        return 0;
    }

    if (option == "-a" || option == "--all") {
        runner.runAllTests();
        return 0;
    }

    // 运行指定的测试用例
    if (!runner.runTest(option)) {
        std::cerr << "Failed to run test case: " << option << std::endl;
        return 1;
    }

    return 0;
}
