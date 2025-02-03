cmake_minimum_required(VERSION 3.14)

project(mesh-slicer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(LIBIGL_DIR ${CMAKE_SOURCE_DIR}/depends/libigl)

set(LIBIGL_WITH_CGAL ON CACHE INTERNAL "Enable CGAL with libIGL")
set(LIBIGL_WITH_COMISO OFF CACHE INTERNAL "Disable COMISO with libIGL")
set(LIBIGL_WITH_TETGEN OFF CACHE INTERNAL "Disable Tetgen with libIGL")
set(LIBIGL_USE_STATIC_LIBRARY OFF CACHE INTERNAL "Disable static lib")
#set(CGAL_WITH_GMPXX OFF CACHE INTERNAL "Disable CGAL with GMPXX") # Compile issue?

include(FetchContent)

FetchContent_Declare(
    libigl
    GIT_REPOSITORY "https://github.com/libigl/libigl"
    SOURCE_DIR ${LIBIGL_DIR}
    GIT_TAG v2.2.0
)

FetchContent_MakeAvailable(libigl)

set(SOURCES 
    src/main.cpp
    src/unproject_on_skeleton.cpp
    src/fitsurface.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(${PROJECT_NAME} igl::core igl::opengl_glfw igl::opengl_glfw_imgui CGAL::CGAL)
