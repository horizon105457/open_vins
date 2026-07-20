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

#include "UpdaterTag.h"

#include "core/VioManagerOptions.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "types/IMU.h"
#include "types/PoseJPL.h"
#include "types/Type.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

#include <Eigen/Eigen>
#include <boost/math/distributions/chi_squared.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

UpdaterTag::UpdaterTag(std::shared_ptr<State> state,
                       std::shared_ptr<Propagator> propagator,
                       const Options &options)
    : state_(std::move(state)), propagator_(std::move(propagator)),
      options_(options) {}

void UpdaterTag::update(const std::vector<ov_core::TagDetection> &candidates,
                        double t_tag,
                        const std::map<int, TagEntry> &tag_db) {

  if (candidates.empty())
    return;

  for (const auto &candidate : candidates) {

    if (!pass_fine_filter(candidate, tag_db))
      continue;

    size_t cam_id = candidate.cam_id;
    auto it_cam = state_->_cam_intrinsics_cameras.find(cam_id);
    if (it_cam == state_->_cam_intrinsics_cameras.end())
      continue;
    auto camera = it_cam->second;

    Eigen::Matrix<double, 7, 1> T_tag_cam;
    Eigen::Matrix<double, 6, 6> I_pnp_cam;

    int tag_id = candidate.id;
    auto it_tag = tag_db.find(tag_id);
    if (it_tag == tag_db.end())
      continue;
    const auto &tag_entry = it_tag->second;

    if (!solve_pnp(candidate, camera, tag_entry.size, T_tag_cam, I_pnp_cam))
      continue;

    auto it_calib = state_->_calib_IMUtoCAM.find(cam_id);
    if (it_calib == state_->_calib_IMUtoCAM.end())
      continue;
    auto &calib = it_calib->second;

    Eigen::Matrix3d R_ItoC = calib->Rot();
    Eigen::Vector3d p_IinC = calib->pos();
    Eigen::Matrix3d R_CtoI = R_ItoC.transpose();
    Eigen::Vector3d p_CinI = -R_CtoI * p_IinC;

    Eigen::Matrix3d R_tag_cam =
        ov_core::quat_2_Rot(T_tag_cam.head<4>());
    Eigen::Vector3d p_tag_in_cam = T_tag_cam.tail<3>();

    Eigen::Matrix3d R_tag_body = R_tag_cam * R_CtoI;
    Eigen::Vector3d p_tag_in_body = p_tag_in_cam + R_tag_cam * p_CinI;

    Eigen::Matrix<double, 7, 1> T_tag_body;
    T_tag_body.head<4>() = ov_core::rot_2_quat(R_tag_body);
    T_tag_body.tail<3>() = p_tag_in_body;

    Eigen::Matrix<double, 6, 6> R_block = Eigen::Matrix<double, 6, 6>::Zero();
    R_block.block<3, 3>(0, 0) = R_CtoI;
    R_block.block<3, 3>(3, 3) = R_CtoI;
    Eigen::Matrix<double, 6, 6> I_pnp_body =
        R_block * I_pnp_cam * R_block.transpose();

    Eigen::Matrix<double, 6, 6> I_world_imu =
        chain_information(tag_entry.info.asDiagonal(), I_pnp_body,
                          tag_entry.pose, T_tag_body);

    Eigen::Matrix3d R_world_tag =
        ov_core::quat_2_Rot(tag_entry.pose.tail<4>());
    Eigen::Vector3d p_world_tag = tag_entry.pose.head<3>();
    Eigen::Matrix<double, 4, 1> q_world_tag = tag_entry.pose.tail<4>();
    Eigen::Matrix<double, 4, 1> q_tag_body = T_tag_body.head<4>();
    Eigen::Matrix<double, 4, 1> q_z =
        ov_core::quat_multiply(q_world_tag, q_tag_body);
    Eigen::Vector3d p_z =
        p_world_tag + R_world_tag * p_tag_in_body;

    Eigen::Matrix<double, 7, 1> z;
    z.head<4>() = q_z;
    z.tail<3>() = p_z;

    Eigen::Matrix3d R_est =
        (state_->_options.do_fej) ? state_->_imu->Rot_fej()
                                  : state_->_imu->Rot();
    Eigen::Vector3d p_est =
        (state_->_options.do_fej) ? state_->_imu->pos_fej()
                                  : state_->_imu->pos();

    Eigen::Matrix<double, 6, 1> r;
    r.head<3>() = -ov_core::log_so3(R_est.transpose() *
                                     ov_core::quat_2_Rot(z.head<4>()));
    r.tail<3>() = z.tail<3>() - p_est;

    whiten_and_update(z, I_world_imu, r, t_tag);
  }
}

bool UpdaterTag::pass_fine_filter(const ov_core::TagDetection &det,
                                   const std::map<int, TagEntry> &tag_db) {
  if (det.decision_margin < options_.min_decision_margin)
    return false;
  if (tag_db.find(det.id) == tag_db.end())
    return false;
  return true;
}

bool UpdaterTag::solve_pnp(const ov_core::TagDetection &det,
                            const std::shared_ptr<ov_core::CamBase> &camera,
                            double tag_size,
                            Eigen::Matrix<double, 7, 1> &T_tag_cam,
                            Eigen::Matrix<double, 6, 6> &I_pnp) {

  cv::Matx33d K = camera->get_K();

  double s = tag_size / 2.0;
  std::vector<cv::Point3d> obj_pts = {
      {-s, -s, 0.0}, {s, -s, 0.0}, {s, s, 0.0}, {-s, s, 0.0}};

  std::vector<cv::Point2d> img_pts;
  for (const auto &c : det.corners) {
    img_pts.emplace_back(c.x, c.y);
  }

  cv::Mat rvec, tvec;
  bool success = cv::solvePnP(obj_pts, img_pts, K, cv::noArray(), rvec, tvec,
                               false, cv::SOLVEPNP_IPPE_SQUARE);
  if (!success)
    return false;

  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);

  Eigen::Matrix3d R;
  R << R_cv.at<double>(0, 0), R_cv.at<double>(0, 1), R_cv.at<double>(0, 2),
      R_cv.at<double>(1, 0), R_cv.at<double>(1, 1), R_cv.at<double>(1, 2),
      R_cv.at<double>(2, 0), R_cv.at<double>(2, 1), R_cv.at<double>(2, 2);

  Eigen::Vector3d t;
  t << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);

  T_tag_cam.head<4>() = ov_core::rot_2_quat(R);
  T_tag_cam.tail<3>() = t;

  double fx = K(0, 0);
  double fy = K(1, 1);

  Eigen::Matrix<double, 8, 6> J = Eigen::Matrix<double, 8, 6>::Zero();

  for (int i = 0; i < 4; i++) {
    Eigen::Vector3d p_tag;
    p_tag << obj_pts[i].x, obj_pts[i].y, obj_pts[i].z;

    Eigen::Vector3d X_cam = R * p_tag + t;
    double X = X_cam(0);
    double Y = X_cam(1);
    double Z = X_cam(2);

    if (Z < 1e-6)
      return false;

    double invZ = 1.0 / Z;
    double invZ2 = invZ * invZ;

    Eigen::Matrix<double, 2, 3> K_proj;
    K_proj << fx * invZ, 0.0, -fx * X * invZ2, 0.0, fy * invZ,
        -fy * Y * invZ2;

    Eigen::Matrix3d p_skew = ov_core::skew_x(p_tag);

    J.block<2, 3>(2 * i, 0) = -K_proj * R * p_skew;
    J.block<2, 3>(2 * i, 3) = K_proj;
  }

  Eigen::Matrix<double, 6, 6> JTJ = J.transpose() * J;

  Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(
      JTJ, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix<double, 6, 1> lambda = svd.singularValues();
  for (int i = 0; i < 6; i++) {
    if (lambda(i) < 1e-6)
      lambda(i) = 0.0;
  }

  double sigma2 = options_.sigma_pix * options_.sigma_pix;
  I_pnp =
      svd.matrixV() * lambda.asDiagonal() * svd.matrixV().transpose() / sigma2;

  return true;
}

Eigen::Matrix<double, 6, 6>
UpdaterTag::chain_information(const Eigen::Matrix<double, 6, 6> &I_world_tag,
                               const Eigen::Matrix<double, 6, 6> &I_pnp_body,
                               const Eigen::Matrix<double, 7, 1> &T_world_tag,
                               const Eigen::Matrix<double, 7, 1> &T_tag_body) {

  Eigen::Matrix3d R_world_tag =
      ov_core::quat_2_Rot(T_world_tag.tail<4>());
  Eigen::Vector3d p_tag_imu = T_tag_body.tail<3>();

  Eigen::Matrix<double, 6, 12> J_chain =
      Eigen::Matrix<double, 6, 12>::Zero();

  J_chain.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  J_chain.block<3, 3>(0, 6) = R_world_tag;

  J_chain.block<3, 3>(3, 0) =
      -ov_core::skew_x(R_world_tag * p_tag_imu);
  J_chain.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  J_chain.block<3, 3>(3, 9) = R_world_tag;

  Eigen::Matrix<double, 12, 12> I_blk =
      Eigen::Matrix<double, 12, 12>::Zero();
  I_blk.block<6, 6>(0, 0) = I_world_tag;
  I_blk.block<6, 6>(6, 6) = I_pnp_body;

  Eigen::Matrix<double, 6, 6> I_world_imu =
      J_chain * I_blk * J_chain.transpose();

  return I_world_imu;
}

void UpdaterTag::whiten_and_update(
    const Eigen::Matrix<double, 7, 1> &z,
    const Eigen::Matrix<double, 6, 6> &I_world_imu,
    const Eigen::Matrix<double, 6, 1> &r, double t_tag) {

  Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(I_world_imu);

  Eigen::Matrix<double, 6, 6> R_sqrt_inv;

  if (llt.info() == Eigen::Success) {
    R_sqrt_inv = llt.matrixU();
  } else {
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(
        I_world_imu, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix<double, 6, 1> s = svd.singularValues();
    for (int i = 0; i < 6; i++) {
      if (s(i) < 1e-8)
        s(i) = 0.0;
      else
        s(i) = std::sqrt(s(i));
    }
    R_sqrt_inv = s.asDiagonal() * svd.matrixV().transpose();
  }

  Eigen::Matrix<double, 6, 6> H_whitened =
      R_sqrt_inv * Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Matrix<double, 6, 1> r_whitened = R_sqrt_inv * r;

  Eigen::MatrixXd P_pos =
      StateHelper::get_marginal_covariance(state_, {state_->_imu->p()});
  double pos_trace = P_pos.trace();
  double mul = (pos_trace > options_.tag_pos_publish_threshold * 100.0)
                   ? 100.0
                   : options_.chi2_multipler;

  double chi2 =
      r_whitened.transpose() * r_whitened;

  static std::map<int, double> chi_squared_table;
  if (chi_squared_table.empty()) {
    for (int i = 1; i < 1000; i++) {
      boost::math::chi_squared chi_squared_dist(i);
      chi_squared_table[i] =
          boost::math::quantile(chi_squared_dist, 0.95);
    }
  }

  double chi2_check = chi_squared_table[6];

  if (chi2 > mul * chi2_check)
    return;

  std::vector<std::shared_ptr<Type>> H_order;
  H_order.push_back(state_->_imu->pose());

  StateHelper::EKFUpdate(state_, H_order, H_whitened, r_whitened,
                          Eigen::Matrix<double, 6, 6>::Identity());

  state_->_timestamp = t_tag;
}
