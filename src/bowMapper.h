#ifndef BOW_MAPPER_H_
#define BOW_MAPPER_H_

#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <utility>

#include "DBoW3/DBoW3.h"
#include "opencv4/opencv2/opencv.hpp"

#include "io.h"
#include "frame.h"



class BowMapper {
 public:
  void mapping(const std::vector<MetaData>& mapping_data);
  void save();

 private:
  void trainVocabulary(const std::vector<MetaData>& mapping_data);
  void createDataBase(const std::vector<MetaData>& mapping_data);
  void saveEntry2Frame(std::string filename);

 private:
  std::vector<Frame> vFrames_;
  std::unordered_map<DBoW3::EntryId, uint64_t> databaseID_to_FrameID_;
};


#endif  // BowMapper