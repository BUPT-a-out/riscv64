#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "IR/BasicBlock.h"
#include "IR/Constant.h"
#include "IR/Function.h"
#include "IR/IRBuilder.h"
#include "IR/Module.h"
#include "IR/Type.h"
#include "Instructions/All.h"
#include "Target.h"

namespace riscv64::test {

class CodeGenTester {
   private:
    auto createTestModule(const std::string& testName)
        -> std::unique_ptr<midend::Module>;

   public:
    CodeGenTester() = default;
    ~CodeGenTester() = default;

    // 运行单个测试用例
    void runTest(const std::string& testName);

    // 运行所有测试用例
    void runAllTests();

    // 创建简单的返回整数常量的函数
    auto createReturnConstTest() -> std::unique_ptr<midend::Module>;

    // 创建简单的加法函数
    auto createAddTest() -> std::unique_ptr<midend::Module>;
};

auto CodeGenTester::createReturnConstTest() -> std::unique_ptr<midend::Module> {
    // 创建上下文和模块
    auto context = std::make_unique<midend::Context>();
    auto module =
        std::make_unique<midend::Module>("test_return_const", context.get());

    // 创建函数类型: i32 main()
    auto* i32Type = context->getInt32Type();
    auto* funcType = midend::FunctionType::get(i32Type, {});

    // 创建函数
    auto* func = midend::Function::Create(funcType, "main", module.get());

    // 创建基本块
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);

    // 创建 IRBuilder
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 创建返回指令: ret i32 42
    constexpr int RETURN_VALUE = 42;
    auto* retVal = midend::ConstantInt::get(i32Type, RETURN_VALUE);
    builder.createRet(retVal);

    return module;
}

auto CodeGenTester::createAddTest() -> std::unique_ptr<midend::Module> {
    // 创建上下文和模块
    auto context = std::make_unique<midend::Context>();
    auto module = std::make_unique<midend::Module>("test_add", context.get());

    // 创建函数类型: i32 add(i32, i32)
    auto* i32Type = context->getInt32Type();
    auto* funcType = midend::FunctionType::get(i32Type, {i32Type, i32Type});

    // 创建函数
    auto* func = midend::Function::Create(funcType, "add", module.get());

    // 获取参数
    auto* arg1 = func->getArg(0);
    auto* arg2 = func->getArg(1);
    arg1->setName("a");
    arg2->setName("b");

    // 创建基本块
    auto* entry = midend::BasicBlock::Create(context.get(), "entry", func);

    // 创建 IRBuilder
    midend::IRBuilder builder(context.get());
    builder.setInsertPoint(entry);

    // 创建加法指令: %result = add i32 %a, %b
    auto* result = builder.createAdd(arg1, arg2, "result");

    // 创建返回指令: ret i32 %result
    builder.createRet(result);

    return module;
}

auto CodeGenTester::createTestModule(const std::string& testName)
    -> std::unique_ptr<midend::Module> {
    if (testName == "return_const") {
        return createReturnConstTest();
    } else if (testName == "add") {
        return createAddTest();
    } else {
        std::cerr << "Unknown test case: " << testName << std::endl;
        return nullptr;
    }
}

void CodeGenTester::runTest(const std::string& testName) {
    std::cout << "=== Running test: " << testName << " ===" << std::endl;

    // 创建测试模块
    auto module = createTestModule(testName);
    if (!module) {
        std::cerr << "Failed to create test module: " << testName << std::endl;
        return;
    }

    std::cout << "\n--- Input IR Module ---" << std::endl;
    std::cout << module->toString() << std::endl;

    try {
        // 创建 RISC-V 目标
        RISCV64Target target;

        // 指令选择 pass
        std::cout << "\n--- Running Instruction Selection ---" << std::endl;
        auto riscvModule = target.instructionSelectionPass(*module);

        std::cout << "\n--- Generated RISC-V Assembly ---" << std::endl;
        std::cout << riscvModule.toString() << std::endl;

        // 如果有寄存器分配 pass，也可以运行
        // auto allocatedModule = target.registerAllocationPass(riscvModule);
        // std::cout << "\n--- After Register Allocation ---" << std::endl;
        // std::cout << allocatedModule.toString() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error during code generation: " << e.what() << std::endl;
    }

    std::cout << "\n=== Test " << testName << " completed ===" << std::endl
              << std::endl;
}

void CodeGenTester::runAllTests() {
    std::vector<std::string> testCases = {"return_const", "add"};

    std::cout << "Running all RISC-V code generation tests..." << std::endl
              << std::endl;

    for (const auto& testCase : testCases) {
        runTest(testCase);
    }

    std::cout << "All tests completed!" << std::endl;
}

}  // namespace riscv64::test

int main(int argc, char* argv[]) {
    riscv64::test::CodeGenTester tester;

    if (argc > 1) {
        // 运行指定的测试用例
        std::string testName = argv[1];
        tester.runTest(testName);
    } else {
        // 运行所有测试用例
        tester.runAllTests();
    }

    return 0;
}
