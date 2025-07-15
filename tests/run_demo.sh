#!/bin/bash

# RISC-V Backend Test Runner Script using XMake
set -e

echo "=================================="
echo "RISC-V Backend Code Generation Test"
echo "Using XMake Build System"
echo "=================================="
echo

cd "$(dirname "$0")/../../.."

# 显示帮助信息
show_help() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  all              Run all tests"
    echo "  simple           Run simple tests only"
    echo "  complex          Run complex tests only"
    echo "  list             List available test cases"
    echo "  <testcase>       Run specific test case"
    echo "  help             Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 all                    # Run all tests"
    echo "  $0 simple                # Run simple tests"
    echo "  $0 simple_return         # Run specific test"
    echo "  $0 list                  # List all test cases"
}

# 构建测试框架
build_tests() {
    echo "Building RISC-V test framework..."
    xmake build riscv64_test_framework
    echo
}

case "${1:-all}" in
    "help"|"-h"|"--help")
        show_help
        ;;
    "list")
        build_tests
        echo "--- Available Test Cases ---"
        xmake run riscv64_test_framework --list
        ;;
    "all")
        build_tests
        echo "--- Running All Tests ---"
        xmake run riscv64_test_framework --all
        ;;
    "simple")
        build_tests
        echo "--- Running Simple Tests ---"
        xmake run riscv64_test_framework simple_return
        xmake run riscv64_test_framework simple_add
        ;;
    "complex")
        build_tests
        echo "--- Running Complex Tests ---"
        xmake run riscv64_test_framework arithmetic_ops
        xmake run riscv64_test_framework conditional_branch
        ;;
    *)
        build_tests
        echo "--- Running Test: $1 ---"
        xmake run riscv64_test_framework "$1"
        ;;
esac

echo
echo "=== Test completed ==="