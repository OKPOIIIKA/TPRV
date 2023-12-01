cmake_minimum_required(VERSION 3.10)

project(MPI_Password_Cracker)

find_package(MPI REQUIRED)

add_executable(password_cracker lab3.cpp)

target_link_libraries(password_cracker MPI::MPI_CXX ssl crypto)
