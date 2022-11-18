#pragma once

#include <string>

#include "opencv4/opencv2/opencv.hpp"

class Frame {
 public:
  Frame(const uint64_t &id,
        const cv::Mat &raw_image,
        const cv::Vec3d &gnss_pose) : 
        id_(id), raw_image_(raw_image), gnss_pose_(gnss_pose) {}

 public:
  uint64_t id_;
  cv::Mat raw_image_;
  cv::Vec3d gnss_pose_;
};