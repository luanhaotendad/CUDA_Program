cmake_minimum_required(VERSION 3.21)  # 若≥3.24 可自动用 CUDA native 架构（见下方）

set(CMAKE_CUDA_ARCHITECTURES native)
# ===== 项目信息 =====
project(MyApp
  VERSION 0.1.0
  LANGUAGES CXX CUDA
)

# ===== 语言标准 =====
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# 缺省构建类型（单配置生成器如 Makefiles/Ninja 下生效）
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release;RelWithDebInfo;MinSizeRel")
endif()

# ===== 第三方库：SFML 3.x =====
# 典型项目至少用到 Graphics / Window / System
set(SFML_COMPONENTS Graphics Window System)
set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 3 REQUIRED COMPONENTS ${SFML_COMPONENTS})
# 说明：SFML 3 的组件名首字母需大写，链接目标名为 SFML::<Module>（例如 SFML::Graphics）。这在官方迁移文档与配置脚本中均有说明。

# ===== CUDA 运行库（推荐显式找到 cudart）=====
find_package(CUDAToolkit REQUIRED)

# ===== 源文件收集 =====
# 为了少改 CMake，这里用 GLOB 自动收集；如更追求可控性，改成手工列出源即可。
file(GLOB_RECURSE CPP_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE CU_SOURCES  CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cu")

add_executable(${PROJECT_NAME}
  ${CPP_SOURCES}
  ${CU_SOURCES}
)

# 头文件目录
target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# 链接依赖
target_link_libraries(${PROJECT_NAME}
  PRIVATE
    SFML::Graphics
    SFML::Window
    SFML::System
    CUDA::cudart
)

# CUDA 目标属性：允许跨翻译单元的设备链接
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# CUDA 架构：CMake >= 3.24 可直接用 native 自动探测
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES native)
else()
  message(STATUS "建议手动设置 -DCMAKE_CUDA_ARCHITECTURES=xx（例如 86），否则使用编译器默认。")
endif()

# 常用警告等级
if(MSVC)
  target_compile_options(${PROJECT_NAME} PRIVATE /W4 /permissive-)
else()
  target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# 在 IDE 中按目录树分组显示源码
source_group(TREE "${CMAKE_SOURCE_DIR}/src" FILES ${CPP_SOURCES} ${CU_SOURCES})

