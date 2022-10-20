#ifndef VLAD_MAPPER_H_
#define VLAD_MAPPER_H_

#include <vector>
#include <unordered_map>

#include "opencv4/opencv2/opencv.hpp"

#include "vlad/Vocabulary.h"
#include "io.h"
#include "frame.h"
#include "featureMatcher.h"

class VladMapper {
 public:
  VladMapper();
  void mapping(const std::vector<MetaData>& mapping_data);

  void relocalize(const MetaData& data);

 private:
  void saveEntry2Frame(std::string filename);


 private:
  std::unordered_map<unsigned int, uint64_t> entryId_to_frameID;
  vlad::DataBase* pdb_;
  DataIO trainData_;
  DataIO testData_;

  cv::Mat relocalize_result_;
  cv::VideoWriter video_writer_;
};

#endif