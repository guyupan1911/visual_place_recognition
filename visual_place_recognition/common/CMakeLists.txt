cmake_minimum_required(VERSION 3.16)
project(common)

set(SOURCE_FILES dump.cc converter.cc)

add_library(common SHARED STATIC ${SOURCE_FILES})
target_link_libraries(common ${OpenCV_LIBS} gflags proto_gen_cxx)
