#include "Vocabulary.h"

namespace vlad {
const int CODEBOOK_SIZE = 32;

void Vocabulary::create() {
   std::map<uint64_t, std::pair<std::string, cv::Vec3d>> train_images =
    readImagesAndGNSS("../dataset/train/recording_2021-02-25_13-39-06_images");

  std::map<int64_t, Frame> train_frames;  // map image id to frame

  std::vector<cv::Mat> vDes_train;

  int i = 0;
  for (auto& elem : train_images) {
    if (i% 20 != 0)
      continue;
    Frame frame(elem.first, elem.second.first, elem.second.second,
                "ORB2", 2000);
    train_frames.insert(std::make_pair(elem.first, frame));
    vDes_train.push_back(frame.getDes());
  }

  const int k = 32;
  const int L = 1;
  const DBoW3::WeightingType weight = DBoW3::TF_IDF;
  const DBoW3::ScoringType score = DBoW3::L1_NORM;

  pvoc_ = new DBoW3::Vocabulary(k, L, weight, score);

  std::cout << "Creating a small " << k << "^" << L << " vocabulary..."
  << std::endl;
  pvoc_->create(vDes_train);
  std::cout << "... done!" << std::endl;

  for (int i = 0; i < CODEBOOK_SIZE; i++) {
    cv::Mat cen = voc.getWord(i);
    vWords_.push_back(cen);
  }
}

}  // namespace vlad
