#include <glim/mapping/sub_map.hpp>

#include <fstream>
#include <Eigen/Eigen>
#include <spdlog/spdlog.h>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>

#include <gtsam_points/types/point_cloud_cpu.hpp>
#include <gtsam_points/util/covariance_estimation.hpp>

namespace glim {

void SubMap::drop_frame_points() {
  for (auto& frame : frames) {
    frame = frame->clone_wo_points();
  }

  for (auto& frame : odom_frames) {
    frame = frame->clone_wo_points();
  }
}

void SubMap::save(const std::string& path) const {
  boost::filesystem::create_directories(path);
  std::ofstream ofs(path + "/data.txt");
  ofs << "id: " << id << std::endl;
  ofs << "T_world_origin: " << std::endl << T_world_origin.matrix() << std::endl;
  ofs << "T_origin_endpoint_L: " << std::endl << T_origin_endpoint_L.matrix() << std::endl;
  ofs << "T_origin_endpoint_R: " << std::endl << T_origin_endpoint_R.matrix() << std::endl;

  if (!frames.empty()) {
    ofs << "T_lidar_imu: " << std::endl << frames.back()->T_lidar_imu.matrix() << std::endl;
    ofs << "imu_bias: " << frames.back()->imu_bias.transpose() << std::endl;
    ofs << "frame_id: " << static_cast<int>(frames.back()->frame_id) << std::endl;
  }

  ofs << "num_frames: " << frames.size() << std::endl;

  for (int i = 0; i < frames.size(); i++) {
    ofs << "frame_" << i << std::endl;
    ofs << "id: " << frames[i]->id << std::endl;
    ofs << "stamp: " << boost::format("%.9f") % frames[i]->stamp << std::endl;
    ofs << "T_odom_lidar: " << std::endl << odom_frames[i]->T_world_lidar.matrix() << std::endl;
    ofs << "T_world_lidar: " << std::endl << frames[i]->T_world_lidar.matrix() << std::endl;
    ofs << "v_world_imu: " << frames[i]->v_world_imu.transpose() << std::endl;
  }

  std::ofstream ofs_imu_rate(path + "/imu_rate.txt");
  for (const auto& frame : frames) {
    if (!frame->imu_rate_trajectory.size()) {
      continue;
    }

    for (int i = 0; i < frame->imu_rate_trajectory.cols(); i++) {
      const auto& data = frame->imu_rate_trajectory.col(i);
      ofs_imu_rate << boost::format("%.9f %.6f %.6f %.6f %.6f %.6f %.6f %.6f") % data[0] % data[1] % data[2] % data[3] % data[4] % data[5] % data[6] % data[7] << std::endl;
    }
  }

  frame->save_compact(path);
}

namespace {
template <int ROWS, int COLS>
Eigen::Matrix<double, ROWS, COLS> read_matrix(std::ifstream& ifs) {
  Eigen::Matrix<double, ROWS, COLS> m;
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      ifs >> m(i, j);
    }
  }
  return m;
}
}  // namespace

SubMap::Ptr SubMap::load(const std::string& path) {
  std::ifstream ifs(path + "/data.txt");
  if (!ifs) {
    spdlog::error("failed to open {}/data.txt", path);
    return nullptr;
  }

  SubMap::Ptr submap(new SubMap);

  std::string token;
  ifs >> token >> submap->id;

  ifs >> token;
  submap->T_world_origin.matrix() = read_matrix<4, 4>(ifs);
  ifs >> token;
  submap->T_origin_endpoint_L.matrix() = read_matrix<4, 4>(ifs);
  ifs >> token;
  submap->T_origin_endpoint_R.matrix() = read_matrix<4, 4>(ifs);

  ifs >> token;
  Eigen::Isometry3d T_lidar_imu(read_matrix<4, 4>(ifs));
  ifs >> token;
  Eigen::Matrix<double, 6, 1> imu_bias = read_matrix<6, 1>(ifs);

  int frame_id;
  ifs >> token >> frame_id;

  int num_frames;
  ifs >> token >> num_frames;

  for (int i = 0; i < num_frames; i++) {
    int id;
    double stamp;
    ifs >> token >> token >> id;
    ifs >> token >> stamp;

    ifs >> token;
    Eigen::Isometry3d T_odom_lidar(read_matrix<4, 4>(ifs));

    ifs >> token;
    Eigen::Isometry3d T_world_lidar(read_matrix<4, 4>(ifs));

    ifs >> token;
    Eigen::Vector3d v_world_imu = read_matrix<3, 1>(ifs);

    EstimationFrame::Ptr frame(new EstimationFrame);
    frame->id = id;
    frame->stamp = stamp;
    frame->T_lidar_imu = T_lidar_imu;
    frame->T_world_lidar = T_world_lidar;
    frame->T_world_imu = T_world_lidar * T_lidar_imu;

    frame->v_world_imu = v_world_imu;
    frame->imu_bias = imu_bias;
    frame->frame_id = static_cast<FrameID>(frame_id);

    EstimationFrame::Ptr odom_frame(new EstimationFrame);
    *odom_frame = *frame;
    odom_frame->T_world_lidar = T_odom_lidar;
    odom_frame->T_world_imu = T_odom_lidar * T_lidar_imu;

    submap->frames.push_back(frame);
    submap->odom_frames.push_back(odom_frame);
  }

  auto frame = gtsam_points::PointCloudCPU::load(path);

  if (!frame) {
    spdlog::error("failed to load frame from {}", path);
    return nullptr;
  }

  if (frame->size()) {
    const auto valid_point = [](const Eigen::Vector4d& p) { return std::abs(p.w() - 1.0) < 1e-3 && p.head<3>().allFinite(); };
    const auto valid_cov = [](const Eigen::Matrix4d& cov) {
      if (!cov.allFinite() || !cov.rightCols<1>().isZero(1e-6) || !cov.bottomRows<1>().isZero(1e-6)) {
        return false;
      }

      if (!(cov - cov.transpose()).isZero(1e-6)) {
        return false;
      }

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig;
      eig.computeDirect(cov.topLeftCorner<3, 3>());
      if (eig.eigenvalues()[0] < 1e-6 || eig.eigenvalues()[2] > 1e6) {
        return false;
      }

      return true;
    };

    const bool points_corrupted = !valid_point(frame->points[0]) || !valid_point(frame->points[frame->size() - 1]);
    bool cov_corrupted = !valid_cov(frame->covs[0]) || !valid_cov(frame->covs[frame->size() - 1]) || !frame->has_covs();

    if (points_corrupted) {
      spdlog::warn("corrupted points detected in {}", path);
      size_t nonhomo_count = 0;
      for (size_t i = 0; i < frame->size(); i++) {
        nonhomo_count += std::abs(frame->points[i].w() - 1.0) > 1e-3;
        frame->points[i].w() = 1.0;
      }

      size_t nonfinite_count = 0;
      auto remove_loc = std::remove_if(frame->points_storage.begin(), frame->points_storage.end(), [](const Eigen::Vector4d& p) { return !p.allFinite(); });
      if (remove_loc != frame->points_storage.end()) {
        nonfinite_count = std::distance(remove_loc, frame->points_storage.end());
        frame->points_storage.erase(remove_loc, frame->points_storage.end());
        frame->num_points = frame->points_storage.size();
        cov_corrupted = true;
      }

      spdlog::warn("nonhomo_count={} nonfinite_count={}", nonhomo_count, nonfinite_count);
    }

    if (cov_corrupted) {
      spdlog::warn("corrupted covariances detected in {}", path);
      spdlog::warn("recomputing covariances for submap_{}", submap->id);
      frame->add_covs(gtsam_points::estimate_covariances(frame->points, frame->size()));
    }
  } else {
    spdlog::warn("no points in {}", path);
  }

  submap->frame = frame;

  return submap;
}

// Save a merged point cloud as a single binary PCD file.
// Writes x, y, z (and intensity if available) as 32-bit floats.
bool save_points_pcd(const std::string& path, const gtsam_points::PointCloud& points) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    return false;
  }

  const size_t num_points = points.size();
  const bool has_intensities = points.has_intensities();

  ofs << "# .PCD v0.7 - Point Cloud Data file format\n";
  ofs << "VERSION 0.7\n";
  if (has_intensities) {
    ofs << "FIELDS x y z intensity\n";
    ofs << "SIZE 4 4 4 4\n";
    ofs << "TYPE F F F F\n";
    ofs << "COUNT 1 1 1 1\n";
  } else {
    ofs << "FIELDS x y z\n";
    ofs << "SIZE 4 4 4\n";
    ofs << "TYPE F F F\n";
    ofs << "COUNT 1 1 1\n";
  }
  ofs << "WIDTH " << num_points << "\n";
  ofs << "HEIGHT 1\n";
  ofs << "VIEWPOINT 0 0 0 1 0 0 0\n";
  ofs << "POINTS " << num_points << "\n";
  ofs << "DATA binary\n";

  const int stride = has_intensities ? 4 : 3;
  std::vector<float> buffer(num_points * stride);
  for (size_t i = 0; i < num_points; i++) {
    buffer[i * stride + 0] = static_cast<float>(points.points[i].x());
    buffer[i * stride + 1] = static_cast<float>(points.points[i].y());
    buffer[i * stride + 2] = static_cast<float>(points.points[i].z());
    if (has_intensities) {
      buffer[i * stride + 3] = static_cast<float>(points.intensities[i]);
    }
  }
  ofs.write(reinterpret_cast<const char*>(buffer.data()), sizeof(float) * buffer.size());

  return ofs.good();
}

// Merge the submap points transformed into the world frame (intensities are kept only when every submap has them)
gtsam_points::PointCloud::Ptr merge_submap_points(const std::vector<SubMap::Ptr>& submaps) {
  auto merged = std::make_shared<gtsam_points::PointCloudCPU>();

  size_t total_points = 0;
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
      continue;
    }
    total_points += submap->frame->size();
  }

  std::vector<Eigen::Vector4d> points;
  points.reserve(total_points);

  bool export_intensities = true;
  for (const auto& submap : submaps) {
    if (!submap || !submap->frame || !submap->frame->has_intensities()) {
      export_intensities = false;
      break;
    }
  }

  std::vector<double> intensities;
  if (export_intensities) {
    intensities.reserve(total_points);
  }

  for (const auto& submap : submaps) {
    if (!submap || !submap->frame) {
      continue;
    }

    for (int i = 0; i < submap->frame->size(); i++) {
      const Eigen::Vector4d point = submap->T_world_origin * submap->frame->points[i];
      points.push_back(point);

      if (export_intensities) {
        intensities.push_back(submap->frame->intensities[i]);
      }
    }
  }

  if (!points.empty()) {
    merged->add_points(points);
    if (export_intensities) {
      merged->add_intensities(intensities);
    }
  }

  return merged;
}

}  // namespace glim
