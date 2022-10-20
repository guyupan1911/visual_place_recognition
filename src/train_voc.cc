#include <stdint.h>

#include <map>
#include <string>
#include <iostream>

#include "DBoW3/DBoW3.h"

#include "frame.h"
#include "io.h"
#include "bowMapper.h"
#include "bowLocalizer.h"


void train() {
  // read images
  std::string path_to_dir_train =
    "../dataset/train/recording_2021-02-25_13-39-06_images";
  DataIO train_data(path_to_dir_train);

  std::string path_to_dir_test =
    "../dataset/test/recording_2020-12-22_12-04-35_images";
  DataIO test_data(path_to_dir_test);

  BowMapper mapper;
  mapper.mapping(train_data.get_add_data());

  BowLocalizer localizer;

  for (auto& elem : test_data.get_add_data()) {
    localizer.relocalize(elem);
  }
}

void trainAutoX() {
  std::string path_to_dir_train =
  "/home/yuxuanhuang/projects/mapping-experimental/data/bag/2022-08-25/top_view";
  DataIO train_data;
  train_data.ReadAutoX(path_to_dir_train, 2600, 5000, 2);

  BowMapper mapper;
  mapper.mapping(train_data.get_add_data());

  // std::string path_to_dir_test =
  // "/home/yuxuanhuang/projects/mapping-experimental/data/bag/2022-08-25/top_view";
  // DataIO test_data;
  // test_data.ReadAutoX(path_to_dir_test, 2600, 5000, 3);

  // BowLocalizer localizer;

  // for (auto& elem : test_data.get_add_data()) {
  //   localizer.relocalize(elem);
  // }
}

int main(int argc, char** argv) {
  train();
  // trainAutoX();
}
