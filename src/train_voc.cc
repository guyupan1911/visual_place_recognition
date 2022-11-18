#include <stdint.h>

#include <map>
#include <string>
#include <iostream>

#include "DBoW3/DBoW3.h"

#include "frame.h"
#include "io.h"
#include "bowMapper.h"
#include "bowLocalizer.h"
#include "utils.h"
#include "matplotlibcpp.h"
#include "tracker/tracking.h"
#include "msg/frame.pb.h"

namespace plt = matplotlibcpp;
slam::cvMat convertTocvMat(const cv::Mat &mat) {
  slam::cvMat serializableMat;

  serializableMat.set_rows(mat.rows);
  serializableMat.set_cols(mat.cols);
  serializableMat.set_elt_type(mat.type());
  serializableMat.set_elt_size((int)mat.elemSize());

  size_t dataSize = mat.rows * mat.cols * mat.elemSize();
  serializableMat.set_mat_data(mat.data, dataSize);

  return serializableMat;
}

slam::Frame convertToFrame(const Frame& frame, const std::shared_ptr<DBoW3::Vocabulary> &voc) {
  slam::Frame serialized_frame;
  serialized_frame.set_id(frame.getId());
  serialized_frame.set_path_to_image(frame.path_to_image);
  
  slam::Pose* serialized_pose = new slam::Pose();
  serialized_pose->set_x(frame.getPose()[0]);
  serialized_pose->set_y(frame.getPose()[1]);
  serialized_pose->set_z(frame.getPose()[2]);
  serialized_frame.set_allocated_pose(serialized_pose);

  slam::cvMat serialized_Mat = convertTocvMat(frame.getDes());
  serialized_frame.mutable_desc()->CopyFrom(serialized_Mat);

  DBoW3::BowVector bow_vector;
  voc->transform(frame.getDes(), bow_vector);
  google::protobuf::Map<uint32_t, double>* serialized_bow_vector =
    serialized_frame.mutable_bow_vector(); 
  for(const auto& elem : bow_vector) {
    google::protobuf::MapPair<uint32_t, double> pair(static_cast<uint32_t>(elem.first), double(elem.second));
    serialized_bow_vector->insert(pair);
  }

  return serialized_frame;
}

void train() {
  // read images
  std::string path_to_dir_train =
    "../dataset/train/recording_2021-02-25_13-39-06_images";
  DataIO train_data(path_to_dir_train);

  std::string path_to_dir_test =
    "../dataset/test/recording_2020-12-22_12-04-35_images";
  DataIO test_data(path_to_dir_test);

  BowMapper mapper;
  mapper.mapping(train_data.get_all_data());

  BowLocalizer localizer;

  for (auto& elem : test_data.get_all_data()) {
    localizer.relocalize(elem);
  }
}

void trainAutoX() {
  DataIO train_data;
  train_data.ReadAutoX(FLAGS_autox_train_images_dir);

  BowMapper mapper;
  mapper.mapping(train_data.get_all_data());

  DataIO test_data;
  test_data.ReadAutoX(FLAGS_autox_test_images_dir);




  BowLocalizer localizer;

  for (auto& elem : test_data.get_all_data()) {
    localizer.relocalize(elem);
  }
}

void test_tracking() {
  DataIO train_data;
  train_data.ReadAutoX(FLAGS_autox_train_images_dir);

  Tracking tracker;

  for (auto& elem : train_data.get_all_data()) {
    tracker.track(elem.path_to_image, elem.gnss_pose, elem.image_id);
  }

}

void preprocess() {
  // read images
  DataIO train_data;
  train_data.ReadAutoX(FLAGS_autox_train_images_dir);

  DataIO test_data;
  test_data.ReadAutoX(FLAGS_autox_test_images_dir);


  BowMapper mapper;
  mapper.mapping(train_data.get_all_data());

  // calculate bow vector and save to file
  std::string path_to_voc = "voc.yml.gz";
  std::shared_ptr<DBoW3::Vocabulary> voc(new DBoW3::Vocabulary(path_to_voc));

  std::ofstream output_train("train_data_proto.bin", std::ios::out | std::ios::binary);

  slam::Frames train_frames;
  for (const auto& elem : train_data.get_all_data()) {
    cv::Vec3d pose{elem.gnss_pose[0], elem.gnss_pose[1], elem.gnss_pose[2]};
    Frame frame(elem.image_id, elem.path_to_image, pose, "ORB2", 2000);
    slam::Frame* serailized_frame = train_frames.add_elems();
    serailized_frame->CopyFrom(convertToFrame(frame, voc));
  }
  std::cout << "train data size: " << train_data.get_all_data().size() << std::endl;

  if(train_frames.SerializeToOstream(&output_train))
    std::cout << "encode train frames" << std::endl;

  std::ofstream output_test("test_data_proto.bin", std::ios::out | std::ios::binary);

  slam::Frames test_frames;
  for (const auto& elem : test_data.get_all_data()) {
    cv::Vec3d pose{elem.gnss_pose[0], elem.gnss_pose[1], elem.gnss_pose[2]};
    Frame frame(elem.image_id, elem.path_to_image, pose, "ORB2", 2000);
    slam::Frame* serailized_frame = test_frames.add_elems();
    serailized_frame->CopyFrom(convertToFrame(frame, voc));
  }
  std::cout << "test data size: " << test_data.get_all_data().size() << std::endl;

  if(test_frames.SerializeToOstream(&output_test))
    std::cout << "encode test frames" << std::endl;
}

void postprocess() {

}

int main(int argc, char** argv) {
  // train();
  // trainAutoX();
  // test_tracking();
  preprocess();
}
