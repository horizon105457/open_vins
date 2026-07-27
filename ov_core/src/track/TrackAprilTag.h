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

#ifndef OV_CORE_TRACK_APRILTAG_H
#define OV_CORE_TRACK_APRILTAG_H

#if ENABLE_APRILTAG_TAGS
#include <apriltag.h>
#include <tag36h11.h>
#endif

#include "TrackBase.h"

namespace ov_core {

/**
 * @brief Tracking of AprilTags using the AprilTag C library.
 *
 * This class wraps the AprilTag 3 C library for visual fiducial detection.
 * It performs a single detection pass per frame, applies a coarse filter
 * (minimum edge length and maximum hamming distance), and inserts the
 * four tag corners into the FeatureDatabase with encoded feature IDs.
 *
 * Detections that pass the coarse filter are stored as candidates for
 * subsequent PnP-based absolute pose updates (by UpdaterTag).
 */
class TrackAprilTag : public TrackBase {

public:
  /**
   * @brief Public constructor with configuration variables
   * @param cameras camera calibration object which has all camera intrinsics in it
   * @param max_tag_features stride for featurd ID encoding (tag_id + n * max_tag_features)
   * @param tag_family AprilTag family name (e.g. "36h11", "25h9", "16h5")
   */
  explicit TrackAprilTag(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras,
                         int max_tag_features = 4096,
                         const std::string &tag_family = "36h11");

  ~TrackAprilTag();

  /**
   * @brief Process a new image
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_new_camera(const CameraData &message) override;

  /**
   * @brief Get candidates that passed the coarse filter (for UpdaterTag)
   * @return Thread-safe copy of the pnp_candidates_ vector
   */
  std::vector<ov_core::TagDetection> get_pnp_candidates() const;

protected:
  /**
   * @brief Single C library detection call per frame
   * @param img input grayscale image
   * @param cam_id camera id that produced this image
   * @param detections output vector of TagDetection
   */
  void perform_detection(const cv::Mat &img, size_t cam_id,
                         std::vector<ov_core::TagDetection> &detections);

  /**
   * @brief Coarse filter: edge_px >= min_edge_px AND hamming <= max_hamming
   * @param det a TagDetection to check
   * @param min_edge_px minimum edge length in pixels
   * @param max_hamming maximum allowed hamming distance
   * @return true if the detection passes
   */
  bool pass_coarse_filter(const ov_core::TagDetection &det,
                          int min_edge_px = 20, int max_hamming = 2);

  /**
   * @brief Insert 4 corners into FeatureDatabase with fatid encoding
   * @param det a TagDetection whose corners to insert
   * @param timestamp image timestamp
   * @param cam_id camera id
   * @param max_tag_features stride for featid encoding
   */
  void insert_corners_to_database(const ov_core::TagDetection &det,
                                  double timestamp, size_t cam_id,
                                  int max_tag_features);

  // AprilTag C library handles
#if ENABLE_APRILTAG_TAGS
  apriltag_detector_t *detector_ = nullptr;
  apriltag_family_t *family_ = nullptr;
#endif

  /// Stride for feature ID encoding
  int max_tag_features_;

  /// Mutex for pnp_candidates_ access
  mutable std::mutex mtx_candidates_;

  /// Candidates that passed coarse filter (for UpdaterTag PnP)
  std::vector<ov_core::TagDetection> pnp_candidates_;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_APRILTAG_H */
