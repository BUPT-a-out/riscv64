#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "IR/Module.h"
#include "IR/IRPrinter.h"
#include "Instructions/All.h"
#include "Target.h"

namespace riscv64::test {

class TestRunner {
   public:
    TestRunner() = default;
    ~TestRunner() = default;

    // 运行指定的测试用例
    bool runTestCase(const std::string& testCaseName);

    // 运行所有测试用例
    void runAllTests();

    // 列出所有可用的测试用例
    void listTestCases();

   private:
    // 从文件加载测试用例并解析为中端IR Module
    std::unique_ptr<midend::Module> loadTestCase(
        const std::string& testCaseName);

    // 执行代码生成并打印结果
    void executeCodeGeneration(const std::string& testCaseName,
                               std::unique_ptr<midend::Module> module);

    // 获取测试用例的完整路径
    std::string getTestCasePath(const std::string& testCaseName);

    // 获取所有测试用例的列表
    std::vector<std::string> getAvailableTestCases();

    // 测试用例目录
    const std::string testCaseDir = "testcases/";
};

std::string TestRunner::getTestCasePath(const std::string& testCaseName) {
    return testCaseDir + testCaseName + ".cpp";
}

std::vector<std::string> TestRunner::getAvailableTestCases() {
    std::vector<std::string> testCases;

    try {
        for (const auto& entry :
             std::filesystem::directory_iterator(testCaseDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".cpp") {
                std::string filename = entry.path().stem().string();
                testCases.push_back(filename);
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error reading test case directory: " << e.what()
                  << std::endl;
    }

    return testCases;
}

std::unique_ptr<midend::Module> TestRunner::loadTestCase(
    const std::string& testCaseName) {
    std::string filePath = getTestCasePath(testCaseName);

    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open test case file: " << filePath
                  << std::endl;
        return nullptr;
    }

    // 这里应该调用测试用例文件中定义的函数来创建Module
    // 由于每个测试用例都是一个独立的.cpp文件，包含createModule函数
    // 在实际实现中，我们需要动态加载这些测试用例
    // 这里先返回nullptr，具体的加载逻辑需要根据实际的测试用例格式来实现
    std::cerr << "TODO: Implement dynamic loading of test case: "
              << testCaseName << std::endl;
    return nullptr;
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
