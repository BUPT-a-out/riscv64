local midend_dir = path.join(path.directory(os.scriptdir()), "midend")

if os.isdir(midend_dir) then
    add_deps("midend")
else
    add_requires("midend main", {
        alias   = "midend",
        urls    = "https://github.com/BUPT-a-out/midend.git",
        verify  = false
    })
    add_packages("midend")
end

target("riscv64")
    set_kind("static")
    set_languages("c++17")
    
    add_files("src/*.cpp")
    
    add_includedirs("include", {public = true})
    
    add_headerfiles("include/(**.h)")
    
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

if os.isdir(path.join(os.scriptdir(), "tests")) then
    includes("tests/xmake.lua")
end