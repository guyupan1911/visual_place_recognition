#include <string>

#include "opencv4/opencv2/opencv.hpp"

#include "vladMapper.h"
#include "io.h"
#include "frame.h"

// void train() {
//   // read images
//   std::string path_to_dir_train =
//     "../dataset/train/recording_2021-02-25_13-39-06_images";
//   DataIO train_data(path_to_dir_train);

//   std::string path_to_dir_test =
//     "../dataset/test/recording_2020-12-22_12-04-35_images";
//   DataIO test_data(path_to_dir_test);

//   VladMapper mapper;
//   mapper.mapping(train_data.get_all_data());

//   for (const auto& elem : test_data.get_all_data()) {
//     mapper.relocalize(elem);
//   }
// }

void train_autox() {
  // read images
  DataIO train_data;
  train_data.ReadAutoX(FLAGS_autox_train_images_dir);

  DataIO test_data;
  test_data.ReadAutoX(FLAGS_autox_test_images_dir);

  VladMapper mapper;
  mapper.mapping(train_data.get_all_data());

  for (const auto& elem : test_data.get_all_data()) {
    mapper.relocalize(elem);
  }
}


int main() {
  train_autox();
}
