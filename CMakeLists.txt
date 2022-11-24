cmake_minimum_required(VERSION 3.5)
project(visual_place_recognition)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_definitions("$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")

set(OpenCV_DIR "/home/yuxuanhuang/projects/thirdParty/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# message("OpenCV version: ${OpenCV_VERSION}")

set(DBoW3_DIR "/home/yuxuanhuang/projects/thirdParty/DBow3/build")
find_package(DBoW3 REQUIRED)
include_directories(DBoW3_INCLUDE_DIRS)
message("DBoW3 include ${DBoW3_INCLUDE_DIRS}")
# message("DBoW3 libs ${DBoW3_LIBS}")

find_package(gflags REQUIRED)

find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARIES})

find_package(matplotlib_cpp)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# add_library(visual_place_recognition SHARED
#     src/frame.cc
#     src/io.cc
#     src/ORBextractor.cc
#     src/kdtree/KDTree.cpp
#     src/bowMapper.cc
#     src/bowLocalizer.cc
#     src/vladMapper.cc
#     src/vlad/Vocabulary.cc
#     src/featureMatcher.cc
#     src/tracker/tracking.cc
#     src/msg/addressbook.pb.cc
#     src/msg/cvMat.pb.cc
#     src/msg/frame.pb.cc)
# target_link_libraries(visual_place_recognition
#     ${OpenCV_LIBS}
#     ${DBoW3_LIBS}
#     gflags
#     ${Boost_LIBRARIES}
#     ${matplotlib_cpp_LIBRARIES}
#     Python3::Python
#     Python3::Module
#     ${Protobuf_LIBRARIES})

# add_executable(hello_world src/hello_world.cc)
# add_executable(main src/main.cc
#     src/frame.cc)
# link_directories(${OpenCV_LIBRARY_DIRS})
# target_link_libraries(main ${OpenCV_LIBS} ${DBoW3_LIBS})

# add_executable(train_voc src/train_voc.cc) 
# target_link_libraries(train_voc visual_place_recognition ${Boost_LIBRARIES})

# add_executable(query src/query.cc
#     src/frame.cc src/io.cc src/ORBextractor.cc)
# link_directories(${OpenCV_LIBRARY_DIRS})
# target_link_libraries(query ${OpenCV_LIBS} ${DBoW3_LIBS})

# add_executable(test_vlad src/test_vlad.cc)
# target_link_libraries(test_vlad visual_place_recognition ${Boost_LIBRARIES})

# add_executable(test_IO src/test_io.cc)
# target_link_libraries(test_IO visual_place_recognition ${Boost_LIBRARIES})

# add_executable(test_Config src/test_Config.cc)
# target_link_libraries(test_Config visual_place_recognition ${Boost_LIBRARIES})

# add_executable(test_matplot src/test_matplotlib.cc)
# target_link_libraries(test_matplot visual_place_recognition)

# add_executable(test_proto_write src/test_proto_write.cc)
# target_link_libraries(test_proto_write visual_place_recognition)

# add_executable(test_proto_read src/test_proto_read.cc)
# target_link_libraries(test_proto_read visual_place_recognition)

# add_executable(test_cvMat_proto src/test_cvMat_proto.cc)
# target_link_libraries(test_cvMat_proto visual_place_recognition)

set(VPR_INSTALL_BIN_DIR ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/include
    )

file(GLOB_RECURSE ALL_PROTOS "visual_place_recognition/*.proto")
set(ALL_PROTO_SRCS)
set(ALL_PROTO_HDRS)
foreach(ABS_FIL ${ALL_PROTOS})
    file(RELATIVE_PATH REL_FIL ${PROJECT_SOURCE_DIR} ${ABS_FIL})
    get_filename_component(DIR ${REL_FIL} DIRECTORY)
    get_filename_component(FIL_WE ${REL_FIL} NAME_WE)

    list(APPEND ALL_PROTO_SRCS "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.cc")
    list(APPEND ALL_PROTO_HDRS "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.h")

    add_custom_command(
    OUTPUT "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.cc"
            "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.h"
    COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
    ARGS --cpp_out  ${PROJECT_BINARY_DIR} -I
        ${PROJECT_SOURCE_DIR} ${ABS_FIL}
    DEPENDS ${ABS_FIL}
    COMMENT "Running C++ protocol buffer compiler on ${ABS_FIL}"
    VERBATIM
    )
endforeach()

add_library(proto_gen_cxx ${ALL_PROTO_SRCS} ${ALL_PROTO_HDRS})
target_link_libraries(proto_gen_cxx protobuf)

add_subdirectory(visual_place_recognition)