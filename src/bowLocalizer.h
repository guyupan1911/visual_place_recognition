#ifndef BOW_LOCALIZER_H_
#define BOW_LOCALIZER_H_

#include <unordered_map>
#include <string>
#include <fstream>

#include "DBoW3/DBoW3.h"
#include "opencv4/opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "io.h"
#include "frame.h"
#include "featureMatcher.h"

class BowLocalizer {
 public:
  BowLocalizer();
  void load();
  void relocalize(const MetaData& data);

 private:
  void readEntry2Frame(std::string filename);

 private:
  std::unordered_map<DBoW3::EntryId, uint64_t> databaseID_to_FrameID_;
  DBoW3::Database* pDataBase_;
  DataIO trainData_;
  cv::Mat relocalize_result_;
  cv::Mat result_bar_;

  cv::VideoWriter video_writer_;
};

#endif