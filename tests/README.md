# RISC-V 后端代码生成测试框架

这个目录包含了用于测试 RISC-V 后端代码生成的工具和测试用例。

## 目录结构

```
tests/
├── test_framework.cpp      # 主要的测试运行器（推荐）
├── test_runner.cpp         # 通用测试运行器
├── test_codegen.cpp        # 原有的测试文件
├── testcases/              # 测试用例源文件目录（可选）
├── xmake.lua              # XMake 构建配置
├── run_tests.sh           # 便捷的测试脚本
└── README.md              # 本文件
```

## 构建和运行

### 使用 XMake（推荐）

#### 1. 构建测试框架

```bash
# 从项目根目录
xmake build riscv64_test_framework

# 或构建所有 RISC-V 测试
xmake build riscv64_test_framework riscv64_test_runner riscv64_codegen_test
```

#### 2. 运行测试

```bash
# 运行所有测试
xmake run riscv64_test_framework

# 运行特定测试
xmake run riscv64_test_framework simple_return

# 列出所有测试用例
xmake run riscv64_test_framework --list

# 显示帮助
xmake run riscv64_test_framework --help
```

#### 3. 使用 XMake 任务

```bash
# 运行所有 RISC-V 测试
xmake test-riscv64

# 运行特定测试用例
xmake test-riscv64 -v simple_return

# 列出所有测试用例
xmake test-riscv64-list

# 只运行简单测试
xmake test-riscv64-simple

# 只运行复杂测试
xmake test-riscv64-complex
```

### 使用便捷脚本

```bash
# 进入测试目录
cd modules/riscv64/tests

# 运行所有测试
./run_tests.sh all

# 运行简单测试
./run_tests.sh simple

# 运行特定测试
./run_tests.sh simple_return

# 列出测试用例
./run_tests.sh list

# 显示帮助
./run_tests.sh help
```

## 测试用例说明

当前包含以下测试用例：

1. **simple_return**: 简单的返回常量函数
   ```c
   i32 main() { return 42; }
   ```

2. **simple_add**: 简单的加法函数
   ```c
   i32 add(i32 a, i32 b) { return a + b; }
   ```

3. **arithmetic_ops**: 多个算术运算
   ```c
   i32 arithmetic(i32 a, i32 b) { return (a + b) * (a - b); }
   ```

4. **conditional_branch**: 条件分支和 PHI 节点
   ```c
   i32 max(i32 a, i32 b) { return a > b ? a : b; }
   ```

## 输出格式

每个测试用例的输出包含以下几个部分：

1. **Input Midend IR**: 输入的中端 IR 表示
2. **Generated RISC-V Assembly (with virtual registers)**: 指令选择后生成的 RISC-V 汇编
3. **Final RISC-V Assembly (with physical registers)**: 寄存器分配后的最终汇编（如果实现）
4. **Final Assembly Output**: 最终的汇编文本输出（如果实现）

## 开发和调试

### 添加新的测试用例

1. 在 [`test_framework.cpp`](test_framework.cpp) 中添加新的测试生成器
2. 在 `CodeGenTestRunner` 构造函数中注册新测试
3. 重新构建并运行

### 调试技巧

```bash
# 运行单个测试进行调试
xmake run riscv64_test_framework simple_return

# 使用调试模式构建
xmake config --mode=debug
xmake build riscv64_test_framework

# 使用 GDB 调试
gdb $(xmake show -t riscv64_test_framework)
```

## 集成到 CI/CD

可以将这些测试集成到持续集成中：

```bash
#!/bin/bash
# CI 脚本示例
set -e

echo "Building RISC-V backend tests..."
xmake build riscv64_test_framework

echo "Running all tests..."
xmake run riscv64_test_framework --all

echo "All tests passed!"
```

## 注意事项

1. 确保 `midend` 和 `riscv64` 模块已正确构建
2. 测试框架会优雅地处理未实现的功能
3. 使用 XMake 可以自动处理依赖关系和链接
4. 测试用例的复杂性逐步递增，便于调试