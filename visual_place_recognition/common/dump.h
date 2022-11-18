#pragma once

#include <vector>
#include <string>
#include <fstream>

#include <gflags/gflags.h>
#include "opencv4/opencv2/opencv.hpp"

#include "visual_place_recognition/common/frame.h"

DECLARE_string(autox_train_images_dir);
DECLARE_string(autox_test_images_dir);

void dumpAutoX(const std::string &path, std::vector<Frame> &vFrames);