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

#ifndef OV_MSCKF_UPDATER_TAG_H
#define OV_MSCKF_UPDATER_TAG_H

#include <map>
#include <memory>
#include <vector>

#include <Eigen/Eigen>

namespace ov_core {
struct TagDetection;
class CamBase;
} // namespace ov_core

namespace ov_msckf {

struct TagEntry;

class State;
class Propagator;

/**
 * @brief AprilTag-based absolute pose EKF updater.
 *
 * Processes coarse-filtered AprilTag detection candidates from TrackAprilTag
 * through a fine filter and PnP-based absolute pose update for imu_pose.
 * Implements tag-enhanced absolute odometry as described in the
 * tag-absodom spec:
 *   - §4.3 fine filter: decision_margin + tag database lookup
 *   - §4.4 PnP: solve pose from 4 corners
 *   - §4.5 chain information: propagate I_world_tag + I_pnp → I_world_imu
 *   - §4.6 whitening + FEJ + EKFUpdate
 *   - §4.7 tag information accumulation
 *
 * Single-threaded pipeline: detection → coarse filter → FeatureDatabase →
 * fine filter → PnP → EKF.
 */
class UpdaterTag {

public:
  struct Options {
    double chi2_multipler;
    double min_decision_margin;
    double sigma_pix;
    double info_accum_weight;
    double tag_pos_publish_threshold;
    Options()
        : chi2_multipler(1.0), min_decision_margin(0.5), sigma_pix(1.0),
          info_accum_weight(1.0), tag_pos_publish_threshold(0.01) {}
  };

  UpdaterTag(std::shared_ptr<State> state, std::shared_ptr<Propagator> propagator,
             const Options &options = Options{});

  /**
   * @brief Process coarse-filtered candidates from TrackAprilTag.
   * @param candidates Already passed coarse filter (edge_px, hamming)
   * @param t_tag Image timestamp (must match state clone time)
   * @param tag_db Known tag database (world pose + prior info)
   */
  void update(const std::vector<ov_core::TagDetection> &candidates, double t_tag,
              const std::map<int, TagEntry> &tag_db);

protected:
  bool pass_fine_filter(const ov_core::TagDetection &det,
                        const std::map<int, TagEntry> &tag_db);

  bool solve_pnp(const ov_core::TagDetection &det,
                 const std::shared_ptr<ov_core::CamBase> &camera,
                 double tag_size, Eigen::Matrix<double, 7, 1> &T_tag_cam,
                 Eigen::Matrix<double, 6, 6> &I_pnp);

  Eigen::Matrix<double, 6, 6>
  chain_information(const Eigen::Matrix<double, 6, 6> &I_world_tag,
                    const Eigen::Matrix<double, 6, 6> &I_pnp_body,
                    const Eigen::Matrix<double, 7, 1> &T_world_tag,
                    const Eigen::Matrix<double, 7, 1> &T_tag_body);

  void whiten_and_update(const Eigen::Matrix<double, 7, 1> &z,
                          const Eigen::Matrix<double, 6, 6> &I_world_imu,
                          const Eigen::Matrix<double, 6, 1> &r, double t_tag);

  std::shared_ptr<State> state_;
  std::shared_ptr<Propagator> propagator_;
  Options options_;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_TAG_H
