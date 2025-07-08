cmake_minimum_required(VERSION 3.20)
project(midend VERSION 1.0.0 LANGUAGES CXX)

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
add_library(backend
    ${SOURCES}
)

target_include_directories(midend PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set library properties
set_target_properties(midend PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)



install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT midendTargets
    FILE midendTargets.cmake
    NAMESPACE midend::
    DESTINATION lib/cmake/backend
)

# Generate config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/backendConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/backendConfig.cmake"
    INSTALL_DESTINATION lib/cmake/backend
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/backendConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/backendConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/backendConfigVersion.cmake"
    DESTINATION lib/cmake/backend
)

# Print configuration summary
message(STATUS "midend Configuration Summary:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build testing: ${BUILD_TESTING}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")