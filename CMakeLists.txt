cmake_minimum_required(VERSION 3.5)
project(visual_place_recognition)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -o0 -Wall -g2 -ggdb")

set(OpenCV_DIR "/home/yuxuanhuang/projects/thirdParty/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# message("OpenCV version: ${OpenCV_VERSION}")

set(DBoW3_DIR "/home/yuxuanhuang/projects/thirdParty/DBow3/build")
find_package(DBoW3 REQUIRED)
include_directories(DBoW3_INCLUDE_DIRS)
message("DBoW3 include ${DBoW3_INCLUDE_DIRS}")
# message("DBoW3 libs ${DBoW3_LIBS}")

add_library(visual_place_recognition STATIC
    src/frame.cc
    src/io.cc
    src/ORBextractor.cc
    src/kdtree/KDTree.cpp
    src/bowMapper.cc
    src/bowLocalizer.cc
    src/vladMapper.cc
    src/vlad/Vocabulary.cc
    src/featureMatcher.cc)
target_link_libraries(visual_place_recognition ${OpenCV_LIBS} ${DBoW3_LIBS})

# add_executable(hello_world src/hello_world.cc)
# add_executable(main src/main.cc
#     src/frame.cc)
# link_directories(${OpenCV_LIBRARY_DIRS})
# target_link_libraries(main ${OpenCV_LIBS} ${DBoW3_LIBS})

add_executable(train_voc src/train_voc.cc) 
target_link_libraries(train_voc visual_place_recognition)

# add_executable(query src/query.cc
#     src/frame.cc src/io.cc src/ORBextractor.cc)
# link_directories(${OpenCV_LIBRARY_DIRS})
# target_link_libraries(query ${OpenCV_LIBS} ${DBoW3_LIBS})

add_executable(test_vlad src/test_vlad.cc)
target_link_libraries(test_vlad visual_place_recognition)

add_executable(test_IO src/test_io.cc)
target_link_libraries(test_IO visual_place_recognition)

add_executable(test_Config src/test_Config.cc)
target_link_libraries(test_Config visual_place_recognition)