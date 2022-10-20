/*
 -------------------------------------------------------------------------------
 * Frame implementation file
 *
 * Copyright (C) 2022 AutoX, Inc.
 * Author: Guyu Pan (yuxuanhuang@autox.ai)
 -------------------------------------------------------------------------------
 */

#include "io.h"

std::unordered_map<uint64_t, cv::Vec3d> GNSS_ret;
std::map<uint64_t, std::pair<std::string, cv::Vec3d>> image_ret;

std::unordered_map<uint64_t, cv::Vec3d> readGNSSPoses(std::string path2Dir) {
  std::unordered_map<uint64_t, cv::Vec3d> ret;
  std::string path2GNSS = path2Dir + "/GNSSPoses.txt";
  std::ifstream fin(path2GNSS.c_str());
  std::string line;

  while (getline(fin, line)) {
    if (line[0] == '#')
      continue;

    std::stringstream ss(line);

    std::vector<std::string> vWords;
    std::string word;
    while (getline(ss, word, ','))
      vWords.push_back(word);

    uint64_t frame_id;
    double tx, ty, tz;

    frame_id = stoull(vWords[0]);
    tx = stod(vWords[1]);
    ty = stod(vWords[2]);
    tz = stod(vWords[3]);

    // cout << "frame_id: " << frame_id << " tx: " << tx << " ty: " << ty
    // << " tz: " << tz << endl;
    cv::Vec3d pose(tx, ty, tz);
    ret[frame_id] = pose;
  }
  return ret;
}

std::map<uint64_t, std::pair<std::string, cv::Vec3d>> readImages(
    std::string path2Dir) {
  std::map<uint64_t, std::pair<std::string, cv::Vec3d>> ret;
  std::string path2Time = path2Dir + "/times.txt";
  std::ifstream fin(path2Time.c_str());

  std::string line;
  int i = 0;
  while (getline(fin, line)) {
    std::stringstream ss(line);
    uint64_t frame_id;
    ss >> frame_id;
    // if(i++ % 5 != 0)
    //     continue;
    // std::cout << frame_id << std::endl;
    std::string path2image =
    path2Dir + "/undistorted_images/cam0/" + std::to_string(frame_id) + ".png";

    if (GNSS_ret.find(frame_id) == GNSS_ret.end())
      continue;

    cv::Vec3d pose = GNSS_ret.find(frame_id)->second;
    ret.insert(make_pair(frame_id, make_pair(path2image, pose)));

    // cv::Mat image = cv::imread(path2image, cv::IMREAD_GRAYSCALE);
    // cv::imshow("1", image);
    // cv::waitKey(1);
  }

  std::cout << "total " << ret.size() << " images" << std::endl;
  return ret;
}

std::map<uint64_t, std::pair<std::string, cv::Vec3d>> readImagesAndGNSS(
    std::string path2Dir) {
    GNSS_ret = readGNSSPoses(path2Dir);
    image_ret = readImages(path2Dir);

    return image_ret;
}

DataIO::DataIO(const std::string& path_to_dir) {
  read(path_to_dir);
}

void DataIO::read(const std::string& path_to_dir) {
  readGnss(path_to_dir);
  readImage(path_to_dir);
  create_kdtree();
}

void DataIO::readGnss(const std::string& path_to_dir) {
  std::string path2GNSS = path_to_dir + "/GNSSPoses_small.txt";
  std::ifstream fin(path2GNSS.c_str());
  std::string line;

  while (getline(fin, line)) {
    if (line[0] == '#')
      continue;

    std::stringstream ss(line);

    std::vector<std::string> vWords;
    std::string word;
    while (getline(ss, word, ','))
      vWords.push_back(word);

    uint64_t frame_id;
    double tx, ty, tz, yaw;

    frame_id = stoull(vWords[0]);
    tx = stod(vWords[1]);
    ty = stod(vWords[2]);
    tz = stod(vWords[3]);
    yaw = stod(vWords[4]);

    // cout << "frame_id: " << frame_id << " tx: " << tx << " ty: " << ty
    // << " tz: " << tz << endl;
    cv::Vec4d pose(tx, ty, tz, yaw);
    id_gnss_.insert(std::make_pair(frame_id, pose));
  }
  std::cout << "gnss poses size: " << id_gnss_.size() << std::endl;
}

void DataIO::readImage(const std::string& path_to_dir) {
  std::string path2Time = path_to_dir + "/times.txt";
  std::ifstream fin(path2Time.c_str());

  std::string line;
  int i = 0;
  while (getline(fin, line)) {
    std::stringstream ss(line);
    uint64_t frame_id;
    ss >> frame_id;
    // if(i++ % 5 != 0)
    //     continue;
    // std::cout << frame_id << std::endl;
    std::string path2image =
    path_to_dir + "/undistorted_images/cam0/" +
    std::to_string(frame_id) + ".png";

    if (id_gnss_.find(frame_id) != id_gnss_.end()) {
      cv::Vec4d pose = id_gnss_.find(frame_id)->second;
      id_image_.insert(make_pair(frame_id, path2image));
      all_data_.push_back(MetaData{frame_id, path2image, pose});
      // cv::Mat image = cv::imread(path2image, cv::IMREAD_GRAYSCALE);
      // cv::imshow("1", image);
      // cv::waitKey(5);
    }
  }

  std::cout << "total data " << all_data_.size() << " images" << std::endl;
}

void DataIO::create_kdtree() {
  pointVec all_poses;
  for (auto& elem : all_data_) {
    point_t point{elem.gnss_pose[0], elem.gnss_pose[1],
      elem.gnss_pose[2]};
    all_poses.push_back(point);
  }
  pKDtree_ = new KDTree(all_poses);
}

MetaData DataIO::find_closest(const cv::Vec4d& pose) {
  point_t pt{pose[0], pose[1], pose[2]};
  unsigned int nearest_index = pKDtree_->nearest_index(pt);
  return all_data_[nearest_index];
}

MetaData DataIO::findByFrameId(uint64_t frame_id) {
  MetaData ret;
  ret.image_id = frame_id;
  ret.gnss_pose = id_gnss_[frame_id];
  ret.path_to_image = id_image_[frame_id];
  return ret;
}

void DataIO::Clear() {
  id_gnss_.clear();
  id_image_.clear();
  all_data_.clear();
}

void DataIO::ReadAutoX(const std::string &path_to_dir, int start, int end,
  int step) {
  Clear();
  // read 
  for (int i = start; i < end; ++i) {
    if (i % step == 0) {
      std::string path_to_image = path_to_dir + "/" + std::to_string(i) + ".png";
      all_data_.push_back(MetaData{i, path_to_image, cv::Vec4d()});
    }
  }
}