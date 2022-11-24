#pragma once

#include "opencv4/opencv2/opencv.hpp"

cv::Mat patchNormalization(const cv::Mat &resized_image);

cv::Mat normalization(const cv::Mat &resized_image);

