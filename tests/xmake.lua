add_requires("gtest 1.14.0", {configs = {shared = false, main = true}, system = false})

-- 主要的测试框架（推荐使用）
target("riscv64_test_framework")
    set_kind("binary")
    set_languages("c++17")
    set_default(false)
    
    add_packages("gtest")
    
    add_files("test_framework.cpp")
    
    add_deps("riscv64", "midend")
    
    add_cxxflags("-DGTEST_HAS_PTHREAD=1")
    add_defines("GTEST_LINKED_AS_SHARED_LIBRARY=0")
    
    set_warnings("all")
    add_cxxflags("-Wall", "-Wextra")
    
    if is_mode("debug") then
        add_cxxflags("-g", "-O0")
        set_symbols("debug")
        set_optimize("none")
    elseif is_mode("release") then
        add_cxxflags("-O3", "-DNDEBUG")
        set_symbols("hidden")
        set_optimize("fastest")
    end
    
    add_ldflags("-pthread")
    
    after_build(function (target)
        print("RISC-V test framework built successfully.")
        print("Run with: xmake run riscv64_test_framework")
        print("Or: xmake run riscv64_test_framework simple_return")
    end)

-- 通用测试运行器
target("riscv64_test_runner")
    set_kind("binary")
    set_languages("c++17")
    set_default(false)
    
    add_packages("gtest")
    
    add_files("test_runner.cpp")
    
    add_deps("riscv64", "midend")
    
    add_cxxflags("-DGTEST_HAS_PTHREAD=1")
    add_defines("GTEST_LINKED_AS_SHARED_LIBRARY=0")
    
    set_warnings("all")
    add_cxxflags("-Wall", "-Wextra")
    
    if is_mode("debug") then
        add_cxxflags("-g", "-O0")
        set_symbols("debug")
        set_optimize("none")
    elseif is_mode("release") then
        add_cxxflags("-O3", "-DNDEBUG")
        set_symbols("hidden")
        set_optimize("fastest")
    end
    
    add_ldflags("-pthread")

-- 原有的代码生成测试
target("riscv64_codegen_test")
    set_kind("binary")
    set_languages("c++17")
    set_default(false)
    
    add_packages("gtest")
    
    add_files("test_codegen.cpp")
    
    add_deps("riscv64", "midend")
    
    add_cxxflags("-DGTEST_HAS_PTHREAD=1")
    add_defines("GTEST_LINKED_AS_SHARED_LIBRARY=0")
    
    set_warnings("all")
    add_cxxflags("-Wall", "-Wextra")
    
    if is_mode("debug") then
        add_cxxflags("-g", "-O0")
        set_symbols("debug")
        set_optimize("none")
    elseif is_mode("release") then
        add_cxxflags("-O3", "-DNDEBUG")
        set_symbols("hidden")
        set_optimize("fastest")
    end
    
    add_ldflags("-pthread")

-- 定义测试任务
task("test-riscv64")
    set_menu {
        usage = "xmake test-riscv64 [testcase]",
        description = "Run RISC-V backend tests",
        options = {
            {"testcase", "v", "string", "Specific test case to run"}
        }
    }
    on_run(function (option)
        import("core.project.project")
        import("core.base.task")
        
        -- 构建测试框架
        task.run("build", {}, "riscv64_test_framework")
        
        local testcase = option.testcase
        if testcase then
            -- 运行特定测试用例
            cprint("${color.info}Running specific test case: %s", testcase)
            os.exec("xmake run riscv64_test_framework %s", testcase)
        else
            -- 运行所有测试用例
            cprint("${color.info}Running all RISC-V backend tests...")
            os.exec("xmake run riscv64_test_framework --all")
        end
    end)

task("test-riscv64-list")
    set_menu {
        usage = "xmake test-riscv64-list",
        description = "List all available RISC-V test cases"
    }
    on_run(function ()
        import("core.base.task")
        
        task.run("build", {}, "riscv64_test_framework")
        os.exec("xmake run riscv64_test_framework --list")
    end)

task("test-riscv64-simple")
    set_menu {
        usage = "xmake test-riscv64-simple",
        description = "Run simple RISC-V tests only"
    }
    on_run(function ()
        import("core.base.task")
        
        task.run("build", {}, "riscv64_test_framework")
        cprint("${color.info}Running simple RISC-V tests...")
        os.exec("xmake run riscv64_test_framework simple_return")
        os.exec("xmake run riscv64_test_framework simple_add")
    end)

task("test-riscv64-complex")
    set_menu {
        usage = "xmake test-riscv64-complex",
        description = "Run complex RISC-V tests only"
    }
    on_run(function ()
        import("core.base.task")
        
        task.run("build", {}, "riscv64_test_framework")
        cprint("${color.info}Running complex RISC-V tests...")
        os.exec("xmake run riscv64_test_framework arithmetic_ops")
        os.exec("xmake run riscv64_test_framework conditional_branch")
    end)