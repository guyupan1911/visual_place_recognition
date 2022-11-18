/*
 -------------------------------------------------------------------------------
 * Frame header file
 *
 * Copyright (C) 2022 AutoX, Inc.
 * Author: Guyu Pan (yuxuanhuang@autox.ai)
 -------------------------------------------------------------------------------
 */

#ifndef VISUAL_PLACE_RECOGNITION_FRAME_H_
#define VISUAL_PLACE_RECOGNITION_FRAME_H_
#include <stdint.h>

#include <string>
#include <vector>

#include <gflags/gflags.h>
#include "opencv4/opencv2/opencv.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "ORBextractor.h"

DECLARE_int32(image_row);
DECLARE_int32(image_col);

class MapPoint;

class Frame {
 public:
  Frame(uint64_t frame_id, const std::string& path2im, cv::Vec3d pose,
        std::string type, int feature_nums);

  cv::Mat getImage() const {
            return image_.clone();
        }

  cv::Mat getDes() const {
            return descriptors_.clone();
        }

  uint64_t getId() const {
            return id_;
        }

  std::vector<cv::KeyPoint> getKpts() const {
            return keypoints_;
        }

  cv::Vec3d getPose() const {
            return pose_;
        }

	void updateConnections();

 public:
	std::string path_to_image;
  uint64_t id_;
  cv::Mat image_, mask_;
  std::vector<cv::KeyPoint> keypoints_;
	std::vector<std::shared_ptr<MapPoint>> map_points_;
	std::map<std::shared_ptr<Frame>, int> connections_;
  cv::Mat descriptors_;
  cv::Vec3d pose_;
};

class MapPoint {
 public:
	MapPoint();

	void add_Observations(const std::shared_ptr<Frame> F) {
		observations_and_depth_[F] = -1;
	}

	std::map<std::shared_ptr<Frame>, double> getObservations() {
		return observations_and_depth_;
	}
	
 public:
	std::map<std::shared_ptr<Frame>, double> observations_and_depth_;
	static int nextId;
	int id;
};

#endif  // VISUAL_PLACE_RECOGNITION_FRAME_H_
