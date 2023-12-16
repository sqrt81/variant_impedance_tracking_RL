cmake_minimum_required(VERSION 3.16)

project(
    mujoco_parallel
)

enable_language(C)
enable_language(CXX)
set( CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(FindOrFetch)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  pybind11
  LIBRARY_NAME
  pybind11
  GIT_REPO
  https://github.com/pybind/pybind11
  GIT_TAG
  5b0a6fc2017fcc176545afe3e09c9f9885283242 # v2.10.4
  TARGETS
  pybind11::pybind11_headers
  EXCLUDE_FROM_ALL
)

add_subdirectory(parallel_sim)

add_custom_target( parallel_sim_copy_resources ALL  
    VERBATIM
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/python ${CMAKE_BINARY_DIR}/parallel_sim
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/converted_models ${CMAKE_BINARY_DIR}/models
    DEPENDS _mj_parallel
)



