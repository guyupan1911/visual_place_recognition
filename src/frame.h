/*
 -------------------------------------------------------------------------------
 * Frame header file
 *
 * Copyright (C) 2022 AutoX, Inc.
 * Author: Guyu Pan (yuxuanhuang@autox.ai)
 -------------------------------------------------------------------------------
 */

#ifndef VISUAL_PLACE_RECOGNITION_FRAME_H_
#define VISUAL_PLACE_RECOGNITION_FRAME_H_
#include <stdint.h>

#include <string>
#include <vector>

#include "opencv4/opencv2/opencv.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "ORBextractor.h"

class Frame{
 public:
  Frame(uint64_t frame_id, const std::string& path2im, cv::Vec3d pose,
        std::string type, int feature_nums);

  cv::Mat getImage() const {
            return image_.clone();
        }

  cv::Mat getDes() const {
            return descriptors_.clone();
        }

  uint64_t getId() const {
            return id_;
        }

  std::vector<cv::KeyPoint> getKpts() const {
            return keypoints_;
        }

  cv::Vec3d getPose() const {
            return pose_;
        }

 private:
  uint64_t id_;
  cv::Mat image_, mask_;
  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;
  cv::Vec3d pose_;
};

#endif  // VISUAL_PLACE_RECOGNITION_FRAME_H_
