#pragma once

#include <string>

#include "opencv4/opencv2/opencv.hpp"

class Frame {
 public:
  Frame(const std::string &path, const uint64_t &id,
        const cv::Vec3d &gnss_pose) : 
        url_to_raw_image_(path) , id_(id), gnss_pose_(gnss_pose) {}

  Frame(const std::string &path, const uint64_t &id,
        const cv::Mat &resized_image, const cv::Vec3d &gnss_pose) : 
        url_to_raw_image_(path) , id_(id), resized_image_(resized_image),
        gnss_pose_(gnss_pose) {}

 public:
  std::string url_to_raw_image_;
  uint64_t id_;
  cv::Mat resized_image_;
  cv::Vec3d gnss_pose_;
};