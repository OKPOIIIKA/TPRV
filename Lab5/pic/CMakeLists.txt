cmake_minimum_required(VERSION 3.10)
project(pic)

find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(pic main.cpp kernel.cl)

target_include_directories(pic PRIVATE ${OpenCL_INCLUDE_DIRS})
target_link_libraries(pic PRIVATE ${OpenCL_LIBRARIES} ${OpenCV_LIBS})