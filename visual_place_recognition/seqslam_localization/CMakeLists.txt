cmake_minimum_required(VERSION 3.16)

set(SOURCE_FILES
    seqslam_localization.h seqslam_localization.cc
    )

add_library(seqslam_localization SHARED STATIC ${SOURCE_FILES})
target_link_libraries(seqslam_localization common)
add_dependencies(seqslam_localization common)

add_executable(seqslam_localization_main main.cc)
target_link_libraries(seqslam_localization_main seqslam_localization ${OpenCV_LIBS})
install(TARGETS seqslam_localization_main DESTINATION ${VPR_INSTALL_BIN_DIR})

