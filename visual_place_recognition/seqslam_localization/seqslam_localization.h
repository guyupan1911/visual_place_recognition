#pragma once

#include <vector>
#include <string>

#include "opencv4/opencv2/opencv.hpp"

#include "visual_place_recognition/common/frame.h"
#include "visual_place_recognition/common/proto/frame.pb.h"

class SeqSlamLocalization{
 public:

  void mapping(const std::vector<Frame> &frames, const std::string &filename);
  
  void fetchMap(const std::string &url_to_map_1, const std::string &url_to_map_2);
  
  void localize(const std::vector<Frame> &frames);
  void localize(const Frame &frame);

  void run();

 private:
  void resize();
  void cvtColor();
  void patchNormalization();
  void calculate_difference_matrix();
  void enhanced_difference_matrix();
  cv::Mat calculateTrajectories(int length);
  void findMatchByTrajectory(int testId);
  void findMatchByDifferenceMatrix(int testId);
  void drawMatch(int trainId, int testId);


 private:
  cv::Mat difference_matrix_;
  cv::Mat enhanced_difference_matrix_;
  cv::Mat D_, DD_;
  int success = 0;
  int failure = 0;

  std::vector<Frame> vFrames_[2];
};