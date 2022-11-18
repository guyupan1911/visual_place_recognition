#pragma once

#include <vector>

#include "opencv4/opencv2/opencv.hpp"

#include "visual_place_recognition/common/frame.h"

class SeqSlamLocalization{
 public:
  void mapping(const std::vector<Frame> &frames);
  void localize(const std::vector<Frame> &frames);
  void localize(const Frame &frame);
 		
 private:
  void resize();
  void cvtColor();
  void patchNormalization();
  void calculate_difference_matrix();
  void enhanced_difference_matrix();
  void match();

 private:
  cv::Mat difference_matrix_;
  cv::Mat enhanced_difference_matrix_;
};