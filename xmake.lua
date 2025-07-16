add_rules("plugin.compile_commands.autoupdate", {outputdir = "."})

package("midend")
    set_description("Compiler middle-end")
    set_urls("https://github.com/BUPT-a-out/midend.git")

    on_install("linux", "macosx", "windows", function (package)
        import("package.tools.xmake").install(package)
    end)
package_end()


if os.isfile(path.join(path.directory(os.scriptdir()), "..", "xmake.lua")) then
    add_deps("midend")
else
    add_requires("midend main", {
        system = false,
        verify = false
    })
    add_packages("midend")
end

target("riscv64")
    set_kind("static")
    set_languages("c++17")
    
    add_files("src/*.cpp")
    
    add_includedirs("include", {public = true})
    
    add_headerfiles("include/(*.h)")
    add_headerfiles("include/**/*.h")
    
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

-- 包含测试
if os.isdir(path.join(os.scriptdir(), "tests")) then
    includes("tests/xmake.lua")
end

-- 添加便捷的测试任务
task("test")
    set_menu {
        usage = "xmake test",
        description = "Run RISC-V backend tests",
        options = {}
    }
    on_run(function ()
        import("core.project.project")
        import("core.base.task")
        
        task.run("build", { target = "riscv64_test_framework" })
        os.exec("xmake run riscv64_test_framework")
    end)
