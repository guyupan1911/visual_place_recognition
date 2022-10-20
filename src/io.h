/*
 -------------------------------------------------------------------------------
 * Frame implementation file
 *
 * Copyright (C) 2022 AutoX, Inc.
 * Author: Guyu Pan (yuxuanhuang@autox.ai)
 -------------------------------------------------------------------------------
 */

#ifndef SRC_IO_H_
#define SRC_IO_H_

#include <map>
#include <unordered_map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <utility>

#include "opencv2/opencv.hpp"

#include "kdtree/KDTree.hpp"

std::unordered_map<uint64_t, cv::Vec3d> readGNSSPoses(std::string path2Dir);

std::map<uint64_t, std::pair<std::string, cv::Vec3d>> readImages(
    std::string path2Dir);

std::map<uint64_t, std::pair<std::string, cv::Vec3d>> readImagesAndGNSS(
    std::string path2Dir);

struct MetaData {
  MetaData() {}
  MetaData(uint64_t id, std::string path, cv::Vec4d pose):
    image_id(id), path_to_image(path), gnss_pose(pose) {}

  uint64_t image_id;
  std::string path_to_image;
  cv::Vec4d gnss_pose;
};

class DataIO {
 public:
  DataIO() {}
  explicit DataIO(const std::string& path_to_dir);

  DataIO(const DataIO& data): id_image_(data.id_image_), id_gnss_(data.id_gnss_)
    , all_data_(data.all_data_), pKDtree_(data.pKDtree_) {}

  DataIO& operator=(const DataIO& data) {
    id_image_ = data.id_image_;
    id_gnss_ = data.id_gnss_;
    all_data_ = data.all_data_;
    pKDtree_ = data.pKDtree_;

    return *this;
  }

  MetaData find_closest(const cv::Vec4d& pose);
  MetaData findByFrameId(uint64_t frame_id);
  
  std::vector<MetaData> get_add_data() {
    return all_data_;
  }

  void ReadAutoX(const std::string &path_to_dir, int start, int end, int step);
  void Clear();

 private:
  void read(const std::string& path_to_dir);
  void readGnss(const std::string& path_to_dir);
  void readImage(const std::string& path_to_dir);
  void create_kdtree();

 private:
  std::unordered_map<uint64_t, cv::Vec4d> id_gnss_;
  std::unordered_map<uint64_t, std::string> id_image_;

  std::vector<MetaData> all_data_;
  KDTree* pKDtree_;
};

#endif  // SRC_IO_H_
