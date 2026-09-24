cmake_minimum_required(VERSION 3.16)

project(age_and_gender_dlib_exporter LANGUAGES CXX)

find_package(Threads REQUIRED)

add_executable(
  age_and_gender_export_dlib
  export_dlib.cpp
  ../vendor/dlib/dlib/all/source.cpp
)

add_executable(
  age_and_gender_probe_dlib
  probe_dlib.cpp
  ../vendor/dlib/dlib/all/source.cpp
)

add_executable(
  age_and_gender_export_shape_predictor
  export_shape_predictor.cpp
  ../vendor/dlib/dlib/all/source.cpp
)

add_executable(
  age_and_gender_probe_shape_predictor
  probe_shape_predictor.cpp
  ../vendor/dlib/dlib/all/source.cpp
)

target_include_directories(age_and_gender_export_dlib PRIVATE ../vendor/dlib)
target_include_directories(age_and_gender_probe_dlib PRIVATE ../vendor/dlib)
target_include_directories(age_and_gender_export_shape_predictor PRIVATE ../vendor/dlib)
target_include_directories(age_and_gender_probe_shape_predictor PRIVATE ../vendor/dlib)
target_compile_features(age_and_gender_export_dlib PRIVATE cxx_std_14)
target_compile_features(age_and_gender_probe_dlib PRIVATE cxx_std_14)
target_compile_features(age_and_gender_export_shape_predictor PRIVATE cxx_std_14)
target_compile_features(age_and_gender_probe_shape_predictor PRIVATE cxx_std_14)
target_compile_definitions(age_and_gender_export_dlib PRIVATE DLIB_NO_GUI_SUPPORT)
target_compile_definitions(age_and_gender_probe_dlib PRIVATE DLIB_NO_GUI_SUPPORT)
target_compile_definitions(age_and_gender_export_shape_predictor PRIVATE DLIB_NO_GUI_SUPPORT)
target_compile_definitions(age_and_gender_probe_shape_predictor PRIVATE DLIB_NO_GUI_SUPPORT)

if(MSVC)
  target_compile_options(age_and_gender_export_dlib PRIVATE /O2 /fp:precise)
  target_compile_options(age_and_gender_probe_dlib PRIVATE /O2 /fp:precise)
  target_compile_options(age_and_gender_export_shape_predictor PRIVATE /O2 /fp:precise)
  target_compile_options(age_and_gender_probe_shape_predictor PRIVATE /O2 /fp:precise)
else()
  target_compile_options(age_and_gender_export_dlib PRIVATE -O2 -fno-fast-math)
  target_compile_options(age_and_gender_probe_dlib PRIVATE -O2 -fno-fast-math)
  target_compile_options(age_and_gender_export_shape_predictor PRIVATE -O2 -fno-fast-math)
  target_compile_options(age_and_gender_probe_shape_predictor PRIVATE -O2 -fno-fast-math)
endif()

target_link_libraries(age_and_gender_export_dlib PRIVATE Threads::Threads)
target_link_libraries(age_and_gender_probe_dlib PRIVATE Threads::Threads)
target_link_libraries(age_and_gender_export_shape_predictor PRIVATE Threads::Threads)
target_link_libraries(age_and_gender_probe_shape_predictor PRIVATE Threads::Threads)
