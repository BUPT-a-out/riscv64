cmake_minimum_required(VERSION 3.20)
project(riscv64 VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
endif()


# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Collect source files
file(GLOB_RECURSE SOURCES "src/*.cpp")

# Create library
add_library(riscv64
    ${SOURCES}
)

target_include_directories(riscv64 PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set library properties
set_target_properties(riscv64 PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)


install(TARGETS riscv64
    EXPORT riscv64Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT riscv64Targets
    FILE riscv64Targets.cmake
    NAMESPACE riscv64::
    DESTINATION lib/cmake/riscv64
)

# Generate config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/riscv64Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/riscv64Config.cmake"
    INSTALL_DESTINATION lib/cmake/riscv64
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/riscv64ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/riscv64Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/riscv64ConfigVersion.cmake"
    DESTINATION lib/cmake/riscv64
)

# Print configuration summary
message(STATUS "riscv64 Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build testing: ${BUILD_TESTING}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")