cmake_minimum_required(VERSION 3.16)

if (WIN32 OR APPLE)
    message( FATAL_ERROR "Only linux system is supported" )
endif ()

# INTERPROCEDURAL_OPTIMIZATION is enforced when enabled.
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
# Default to GLVND if available.
set(CMAKE_POLICY_DEFAULT_CMP0072 NEW)

enable_language(C)
enable_language(CXX)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -ggdb")

set( PARALLEL_COMPILE_OPTIONS
    "${AVX_COMPILE_OPTIONS}"
    "${EXTRA_COMPILE_OPTIONS}"
    -fPIC
)
set( PARALLEL_COMPILE_OPTIONS
    "${EXTRA_LINK_OPTIONS}"
)

include_directories(~/.mujoco/mujoco235/include)
link_directories(~/.mujoco/mujoco235/lib)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG 5b0a6fc2017fcc176545afe3e09c9f9885283242 # v2.10.4
)

# exclude from main cmake project
FetchContent_GetProperties( pybind11 )
if ( NOT pybind11_POPULATED )
    FetchContent_Populate( pybind11 )
    add_subdirectory( ${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR}
        EXCLUDE_FROM_ALL )
endif ()

find_package( Threads REQUIRED )

add_library( mj_parallel_sim SHARED
    parallel_sim.cpp
    param_loader.cpp
    a1_gym_env.cpp
)
target_compile_options( mj_parallel_sim PRIVATE ${PARALLEL_COMPILE_OPTIONS} )
target_link_libraries( mj_parallel_sim
    PRIVATE
    glfw3
    X11
    dl
    Threads::Threads
    mujoco
)

pybind11_add_module(_mj_parallel py_interface.cpp)
target_compile_options( _mj_parallel PRIVATE ${PARALLEL_COMPILE_OPTIONS} )
#set_target_properties( _mj_parallel
#    PROPERTIES
#    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#)
target_link_libraries( _mj_parallel
    PRIVATE
    mj_parallel_sim
)

add_executable( eval_success_rate
    eval_success_rate.cpp
)
target_link_libraries( eval_success_rate
    PRIVATE
    glfw3
    X11
    dl
    mj_parallel_sim
    Threads::Threads
    mujoco
)


add_executable( test_a1_env
    test_a1_env.cpp
)
target_link_libraries( test_a1_env
    PRIVATE
    glfw3
    X11
    dl
    mj_parallel_sim
    Threads::Threads
    mujoco
)

add_executable( test_parallel_env
    test_parallel_env.cpp
)
target_link_libraries( test_parallel_env
    PRIVATE
    mj_parallel_sim
)
