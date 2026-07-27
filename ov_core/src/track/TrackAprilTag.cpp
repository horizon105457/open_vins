/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "TrackAprilTag.h"

#include <opencv2/imgproc.hpp>

#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "utils/opencv_lambda_body.h"

using namespace ov_core;

TrackAprilTag::TrackAprilTag(
    std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras,
    int max_tag_features,
    const std::string &tag_family)
    : TrackBase(cameras, 0, 0, false, HistogramMethod::NONE),
      max_tag_features_(max_tag_features) {
#if ENABLE_APRILTAG_TAGS
  detector_ = apriltag_detector_create();
  if (detector_ == nullptr) {
    PRINT_ERROR(RED "[ERROR]: failed to create apriltag detector\n" RESET);
    std::exit(EXIT_FAILURE);
  }

  if (tag_family == "36h11") {
    family_ = tag36h11_create();
  } else {
    PRINT_ERROR(RED "[ERROR]: unsupported tag family: %s\n" RESET, tag_family.c_str());
    std::exit(EXIT_FAILURE);
  }

  apriltag_detector_add_family(detector_, family_);

  detector_->nthreads = 1;
  detector_->quad_decimate = 4.0f;
  detector_->refine_edges = 0;
  detector_->decode_sharpening = 0.0;
#else
  PRINT_ERROR(RED "[ERROR]: you have not compiled with apriltag tag support!!!\n" RESET);
  std::exit(EXIT_FAILURE);
#endif
}

TrackAprilTag::~TrackAprilTag() {
#if ENABLE_APRILTAG_TAGS
  if (detector_ != nullptr) {
    apriltag_detector_destroy(detector_);
    detector_ = nullptr;
  }
  if (family_ != nullptr) {
    tag36h11_destroy(family_);
    family_ = nullptr;
  }
#endif
}

void TrackAprilTag::feed_new_camera(const CameraData &message) {

  // Error check that we have all the data
  if (message.sensor_ids.empty() ||
      message.sensor_ids.size() != message.images.size() ||
      message.images.size() != message.masks.size()) {
    PRINT_ERROR(RED "[ERROR]: MESSAGE DATA SIZES DO NOT MATCH OR EMPTY!!!\n" RESET);
    PRINT_ERROR(RED "[ERROR]:   - message.sensor_ids.size() = %zu\n" RESET, message.sensor_ids.size());
    PRINT_ERROR(RED "[ERROR]:   - message.images.size() = %zu\n" RESET, message.images.size());
    PRINT_ERROR(RED "[ERROR]:   - message.masks.size() = %zu\n" RESET, message.masks.size());
    std::exit(EXIT_FAILURE);
  }

#if ENABLE_APRILTAG_TAGS
  // Clear previous candidates
  {
    std::lock_guard<std::mutex> lck(mtx_candidates_);
    pnp_candidates_.clear();
  }

  // Process each camera image
  for (size_t i = 0; i < message.images.size(); i++) {
    size_t cam_id = message.sensor_ids.at(i);

    // Start timing
    rT1 = boost::posix_time::microsec_clock::local_time();

    // Lock this data feed for this camera
    std::lock_guard<std::mutex> lck(mtx_feeds.at(cam_id));

    // Histogram equalize
    cv::Mat img;
    if (histogram_method == HistogramMethod::HISTOGRAM) {
      cv::equalizeHist(message.images.at(i), img);
    } else if (histogram_method == HistogramMethod::CLAHE) {
      double eq_clip_limit = 10.0;
      cv::Size eq_win_size = cv::Size(8, 8);
      cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(eq_clip_limit, eq_win_size);
      clahe->apply(message.images.at(i), img);
    } else {
      img = message.images.at(i);
    }

    // Perform detection
    std::vector<ov_core::TagDetection> detections;
    perform_detection(img, cam_id, detections);
    rT2 = boost::posix_time::microsec_clock::local_time();

    // Filter and insert corners into database
    std::vector<size_t> ids_new;
    std::vector<cv::KeyPoint> pts_new;

    for (size_t j = 0; j < detections.size(); j++) {
      if (!pass_coarse_filter(detections.at(j)))
        continue;

      insert_corners_to_database(detections.at(j), message.timestamp, cam_id,
                                 max_tag_features_);

      for (int n = 0; n < 4; n++) {
        cv::KeyPoint kpt;
        kpt.pt = detections.at(j).corners.at(n);
        size_t tmp_id = (size_t)detections.at(j).id + n * (size_t)max_tag_features_;
        ids_new.push_back(tmp_id);
        pts_new.push_back(kpt);
      }
    }
    rT3 = boost::posix_time::microsec_clock::local_time();

    // Store candidates that passed coarse filter
    {
      std::lock_guard<std::mutex> lck(mtx_candidates_);
      for (size_t j = 0; j < detections.size(); j++) {
        if (pass_coarse_filter(detections.at(j))) {
          pnp_candidates_.push_back(detections.at(j));
        }
      }
    }

    // Move forward in time
    {
      std::lock_guard<std::mutex> lckv(mtx_last_vars);
      img_last[cam_id] = img;
      img_mask_last[cam_id] = message.masks.at(i);
      ids_last[cam_id] = ids_new;
      pts_last[cam_id] = pts_new;
    }

    // Timing information
    PRINT_ALL("[TIME-APRILTAG]: %.4f seconds for detection\n",
              (rT2 - rT1).total_microseconds() * 1e-6);
    PRINT_ALL("[TIME-APRILTAG]: %.4f seconds for feature DB update (%d features)\n",
              (rT3 - rT2).total_microseconds() * 1e-6, (int)ids_new.size());
    PRINT_ALL("[TIME-APRILTAG]: %.4f seconds for total\n",
              (rT3 - rT1).total_microseconds() * 1e-6);
  }
#else
  PRINT_ERROR(RED "[ERROR]: you have not compiled with apriltag tag support!!!\n" RESET);
  std::exit(EXIT_FAILURE);
#endif
}

void TrackAprilTag::perform_detection(const cv::Mat &img, size_t cam_id,
                                       std::vector<ov_core::TagDetection> &detections) {
#if ENABLE_APRILTAG_TAGS
  detections.clear();

  // Create image_u8_t header pointing to cv::Mat data
  image_u8_t im = {img.cols, img.rows, img.cols, img.data};

  // Run the detector
  zarray_t *detections_raw = apriltag_detector_detect(detector_, &im);

  // Convert to our TagDetection structs
  int num_detections = zarray_size(detections_raw);
  for (int i = 0; i < num_detections; i++) {
    apriltag_detection_t *det;
    zarray_get(detections_raw, i, &det);

    TagDetection tag_det;
    tag_det.id = det->id;
    tag_det.hamming = det->hamming;
    tag_det.decision_margin = (double)det->decision_margin;
    tag_det.cam_id = cam_id;

    // Extract the 4 corners (counter-clockwise)
    tag_det.corners.resize(4);
    for (int n = 0; n < 4; n++) {
      tag_det.corners.at(n).x = (float)det->p[n][0];
      tag_det.corners.at(n).y = (float)det->p[n][1];
    }

    // Compute max_edge_px: max corner-to-corner distance
    double max_edge = 0.0;
    for (int n = 0; n < 4; n++) {
      int next = (n + 1) % 4;
      double dx = det->p[n][0] - det->p[next][0];
      double dy = det->p[n][1] - det->p[next][1];
      double edge = std::sqrt(dx * dx + dy * dy);
      if (edge > max_edge)
        max_edge = edge;
    }
    tag_det.max_edge_px = max_edge;

    detections.push_back(tag_det);

    apriltag_detection_destroy(det);
  }

  apriltag_detections_destroy(detections_raw);
#endif
}

bool TrackAprilTag::pass_coarse_filter(const ov_core::TagDetection &det,
                                        int min_edge_px, int max_hamming) {
  return (det.max_edge_px >= (double)min_edge_px && det.hamming <= max_hamming);
}

void TrackAprilTag::insert_corners_to_database(
    const ov_core::TagDetection &det, double timestamp, size_t cam_id,
    int max_tag_features) {

  for (int n = 0; n < 4; n++) {
    size_t featid = (size_t)det.id + n * (size_t)max_tag_features;
    cv::Point2f npt = camera_calib.at(cam_id)->undistort_cv(det.corners.at(n));
    database->update_feature(featid, timestamp, cam_id,
                             det.corners.at(n).x, det.corners.at(n).y,
                             npt.x, npt.y);
  }
}

std::vector<ov_core::TagDetection> TrackAprilTag::get_pnp_candidates() const {
  std::lock_guard<std::mutex> lck(mtx_candidates_);
  return pnp_candidates_;
}

