cmake_minimum_required(VERSION 3.15)
project(Pyre VERSION 0.1.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add your source files
add_executable(pyre
    src/pyre/linear.cpp
    src/pyre/main.cpp
)

# Include directories
target_include_directories(pyre
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Fetch and link Ember
include(FetchContent)
FetchContent_Declare(
    ember
    GIT_REPOSITORY https://github.com/austinagii/ember.git
    GIT_TAG v1.1.0
)
FetchContent_MakeAvailable(ember)
target_link_libraries(pyre PRIVATE ember)
