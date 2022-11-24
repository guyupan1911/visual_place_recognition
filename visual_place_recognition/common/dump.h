#pragma once

#include <vector>
#include <string>
#include <fstream>

#include <gflags/gflags.h>
#include "opencv4/opencv2/opencv.hpp"

#include "visual_place_recognition/common/frame.h"
#include "visual_place_recognition/common/preprocessing.h"

DECLARE_string(autox_train_images_dir);
DECLARE_string(autox_test_images_dir);
DECLARE_string(seasons_train_images_dir);
DECLARE_string(seasons_test_images_dir);

void dumpAutoX(const std::string &path, std::vector<Frame> &vFrames);

void dump4Seasons(const std::string &path, std::vector<Frame> &vFrames);
