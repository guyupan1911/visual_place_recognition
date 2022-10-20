#ifndef VLAD_VOCABULARY_H_
#define VLAD_VOCABULARY_H_

#include <vector>
#include <string>

#include "opencv4/opencv2/opencv.hpp"
#include "DBoW3/DBoW3.h"

namespace vlad {

const int clusterCount = 256;

class Vocabulary {
 public:
  void create(const std::vector<cv::Mat> &descriptors);

  void save();
  void load();

 private:
  cv::Mat labels_;
  std::vector<cv::Mat> centers_;
};





class DataBase {
 public:
  explicit DataBase(std::string voc_filename);
  unsigned int add(const cv::Mat &descriptor);
  std::vector<unsigned int> query(const cv::Mat &descriptor, int nums);

 private:
  cv::Mat calculate_vlad_vector(const cv::Mat& descriptor);
  double euclidean_distance(cv::Mat baseImg, cv::Mat targetImg);
  int hammingDistance(cv::Mat baseImg, cv::Mat targetImg);

  struct dist {
    int dis;
    int index;
  };

  struct distHamming {
    int dis;
    int index;
  };

  static bool vecCmp(const dist &a, const dist &b) {
        return a.dis < b.dis;
  }

  static bool vecCmpHamming(const distHamming &a, const distHamming &b) {
      return a.dis < b.dis;
  }

 private:
  DBoW3::Vocabulary* pVoc_;
  std::vector<cv::Mat> vlad_vectors_;
};

}  // namespace vlad

#endif  //  VLAD_VOCABULARY_H_
