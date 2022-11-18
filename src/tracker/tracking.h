#pragma once

#include <vector>
#include <string>
#include <fstream>

#include "opencv4/opencv2/opencv.hpp"

#include "../frame.h"
#include "../featureMatcher.h"

class Tracking {
public:
  Tracking();
  void track(const std::string &path_to_image, const cv::Vec3d pose,
    uint64_t frame_id);

  void save(const std::string filename);

private:
  std::shared_ptr<Frame> last_frame_, current_frame_;
  std::vector<std::shared_ptr<Frame>> vFrames_;
  std::vector<std::shared_ptr<MapPoint>> vMapPoints_;
  std::shared_ptr<FeatureMatcher> feature_matcher_;
};