cmake_minimum_required(VERSION 2.8.12)

if (CMAKE_VERSION VERSION_LESS 3.0)
  PROJECT(age_and_gender CXX)
else()
  cmake_policy(SET CMP0048 NEW)
  PROJECT(age_and_gender VERSION "1.0.0" LANGUAGES CXX)
endif()

set(CMAKE_VERBOSE_MAKEFILE OFF)

include_directories("libs/dlib")
add_subdirectory(libs/pybind11)

pybind11_add_module(age_and_gender src/main.cpp libs/dlib/dlib/all/source.cpp)

target_link_libraries(age_and_gender PRIVATE pthread)
target_link_libraries(age_and_gender PRIVATE X11)
target_link_libraries(age_and_gender PRIVATE jpeg)
target_link_libraries(age_and_gender PRIVATE png)

add_definitions(-DDLIB_JPEG_SUPPORT -DDLIB_PNG_SUPPORT)
