#ifndef VLAD_VOCABULARY_H_
#define VLAD_VOCABULARY_H_

#include <vector>

#include "opencv4/opencv2/opencv.hpp"
#include "DBoW3/DBoW3.h"

#include "../io.h"
#include "../frame.h"

namespace vlad {

class Vocabulary {
 public:
  void create();

  void save();
  void load();

 private:
  DBoW3::Vocabulary* pvoc_;
};

class Database {
 public:

 private:
  Vocabulary* voc_;
};

}  // namespace vlad

#endif  //  VLAD_VOCABULARY_H_
