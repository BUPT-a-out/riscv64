# RISC-V Backend Code Generation Test Makefile

# 编译器设置
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
INCLUDES = -I../include -I../../midend/include
LIBS = -L../../../build/modules/midend -L../../../build/modules/riscv64 -lmidend -lriscv64

# 目标文件
TARGETS = test_framework test_runner test_codegen

# 源文件
SOURCES = test_framework.cpp test_runner.cpp test_codegen.cpp

.PHONY: all clean run-all run-simple list help

# 默认目标
all: $(TARGETS)

# 编译各个测试程序
test_framework: test_framework.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

test_runner: test_runner.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

test_codegen: test_codegen.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LIBS)

# 运行测试
run-all: test_framework
	@echo "=== Running all tests with test_framework ==="
	./test_framework --all

run-simple: test_framework
	@echo "=== Running simple tests ==="
	./test_framework simple_return
	./test_framework simple_add

run-complex: test_framework
	@echo "=== Running complex tests ==="
	./test_framework arithmetic_ops
	./test_framework conditional_branch

run-original: test_codegen
	@echo "=== Running original tests ==="
	./test_codegen

# 列出可用的测试用例
list: test_framework
	./test_framework --list

# 清理
clean:
	rm -f $(TARGETS)
	rm -f *.o

# 帮助
help:
	@echo "Available targets:"
	@echo "  all          - Build all test programs"
	@echo "  test_framework - Build the main test framework"
	@echo "  test_runner  - Build the generic test runner"
	@echo "  test_codegen - Build the original test program"
	@echo "  run-all      - Run all tests"
	@echo "  run-simple   - Run simple tests only"
	@echo "  run-complex  - Run complex tests only"
	@echo "  run-original - Run original tests"
	@echo "  list         - List available test cases"
	@echo "  clean        - Remove built executables"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Build all programs"
	@echo "  make run-all           # Build and run all tests"
	@echo "  make test_framework    # Build only the main test framework"
	@echo "  ./test_framework simple_return  # Run specific test"
