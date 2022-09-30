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
