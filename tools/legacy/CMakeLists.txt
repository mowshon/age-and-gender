cmake_minimum_required(VERSION 3.16)

project(age_and_gender_legacy_oracle LANGUAGES CXX)

find_package(Threads REQUIRED)
string(TOUPPER "${CMAKE_BUILD_TYPE}" ORACLE_BUILD_TYPE_UPPER)
set(ORACLE_CONFIG_FLAGS "${CMAKE_CXX_FLAGS_${ORACLE_BUILD_TYPE_UPPER}}")

add_executable(
  age_and_gender_legacy_oracle
  oracle.cpp
  ../../libs/dlib/dlib/all/source.cpp
)

target_include_directories(
  age_and_gender_legacy_oracle
  PRIVATE ../../libs/dlib
)
target_compile_features(age_and_gender_legacy_oracle PRIVATE cxx_std_14)
target_compile_definitions(
  age_and_gender_legacy_oracle
  PRIVATE
    DLIB_NO_GUI_SUPPORT
    ORACLE_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
)

if(MSVC)
  target_compile_options(age_and_gender_legacy_oracle PRIVATE /O2 /fp:precise)
  target_compile_definitions(
    age_and_gender_legacy_oracle
    PRIVATE ORACLE_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${ORACLE_CONFIG_FLAGS} /O2 /fp:precise"
  )
else()
  target_compile_options(age_and_gender_legacy_oracle PRIVATE -O2 -fno-fast-math)
  target_compile_definitions(
    age_and_gender_legacy_oracle
    PRIVATE ORACLE_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${ORACLE_CONFIG_FLAGS} -O2 -fno-fast-math"
  )
endif()

target_link_libraries(age_and_gender_legacy_oracle PRIVATE Threads::Threads)
