cmake_minimum_required(VERSION 3.16)

set(SOURCE_FILES
    seqslam_localization.cc
)

add_library(seqslam_localization SHARED STATIC ${SOURCE_FILES})
target_link_libraries(seqslam_localization common)
add_dependencies(seqslam_localization common)

add_executable(localization_main localization_main.cc)
target_link_libraries(localization_main seqslam_localization ${OpenCV_LIBS})
install(TARGETS localization_main DESTINATION ${VPR_INSTALL_BIN_DIR})

add_executable(mapping_main mapping_main.cc)
target_link_libraries(mapping_main seqslam_localization ${OpenCV_LIBS})
install(TARGETS mapping_main DESTINATION ${VPR_INSTALL_BIN_DIR})