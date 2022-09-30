#include <stdint.h>

#include <map>
#include <string>
#include <iostream>

#include "DBoW3/DBoW3.h"

#include "frame.h"
#include "io.h"


void train(int image_step, std::string feature_type, int feature_nums) {
  // read images
  std::map<uint64_t, std::pair<std::string, cv::Vec3d>> train_images =
    readImagesAndGNSS("../dataset/train/recording_2021-02-25_13-39-06_images");

  std::map<int64_t, Frame> train_frames;  // map image id to frame

  std::vector<cv::Mat> vDes_train;

  int i = 0;
  for (auto& elem : train_images) {
    if (i% image_step != 0)
      continue;
    Frame frame(elem.first, elem.second.first, elem.second.second,
                feature_type, feature_nums);
    train_frames.insert(std::make_pair(elem.first, frame));
    vDes_train.push_back(frame.getDes());
  }

  const int k = 10;
  const int L = 4;
  const DBoW3::WeightingType weight = DBoW3::TF_IDF;
  const DBoW3::ScoringType score = DBoW3::L1_NORM;

  DBoW3::Vocabulary voc(k, L, weight, score);

  std::cout << "Creating a small " << k << "^" << L << " vocabulary..."
  << std::endl;
  voc.create(vDes_train);
  std::cout << "... done!" << std::endl;

  std::cout << std::endl << "Saving vocabulary..." << std::endl;
  std::string path_to_voc = feature_type + ".yml.gz";
  voc.save(path_to_voc);
  std::cout << "Done" << std::endl;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: ./train_voc image_step feature_type feature_nums"
    << std::endl;
  }

  int image_step = std::stoi(argv[1]);
  int feature_nums = std::stoi(argv[3]);
  std::string feature_type = argv[2];

  train(image_step, feature_type, feature_nums);
}
