/*
 -------------------------------------------------------------------------------
 * Frame implementation file
 *
 * Copyright (C) 2022 AutoX, Inc.
 * Author: Guyu Pan (yuxuanhuang@autox.ai)
 -------------------------------------------------------------------------------
 */

#ifndef SRC_IO_H_
#define SRC_IO_H_

#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

#include "opencv2/opencv.hpp"

std::unordered_map<uint64_t, cv::Vec3d> readGNSSPoses(std::string path2Dir);

std::map<uint64_t, std::pair<std::string, cv::Vec3d>> readImages(
    std::string path2Dir);

std::map<uint64_t, std::pair<std::string, cv::Vec3d>> readImagesAndGNSS(
    std::string path2Dir);

#endif  // SRC_IO_H_
