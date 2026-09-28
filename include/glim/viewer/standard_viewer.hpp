#pragma once

#include <mutex>
#include <chrono>
#include <atomic>
#include <thread>
#include <memory>
#include <vector>
#include <unordered_map>
#include <optional>
#include <boost/weak_ptr.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <gtsam_points/util/gtsam_migration.hpp>
#include <glim/util/extension_module.hpp>
#include <gtsam_points/util/runnning_statistics.hpp>

namespace spdlog {
class logger;
}

namespace gtsam {
class NonlinearFactor;
}

namespace glk {
class Drawable;
class PointCloudBuffer;
}

namespace guik {
class ShaderSetting;
}

namespace glim {

class TrajectoryManager;
struct EstimationFrame;

struct SubMapMemoryStats;
struct FactorMemoryStats;

class StandardViewer : public ExtensionModule {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StandardViewer();
  virtual ~StandardViewer();

  virtual bool ok() const override;

  void invoke(const std::function<void()>& task);

private:
  Eigen::Isometry3f resolve_pose(const std::shared_ptr<const EstimationFrame>& frame);

  void set_callbacks();
  void viewer_loop();

  bool drawable_filter(const std::string& name);
  void drawable_selection();
  void update_axes();
  void apply_camera_mode();
  void apply_color(guik::ShaderSetting& shader_setting, int group, int id) const;
  void update_colors(int group);
  void sync_settings(bool save);

private:
  std::atomic_bool viewer_started;
  std::atomic_bool request_to_terminate;
  std::atomic_bool kill_switch;
  std::thread thread;


  bool enable_partial_rendering;
  int partial_rendering_budget;

  bool track;
  bool show_current_coord;
  bool show_current_points;
  int camera_mode;

  bool show_odometry_scans;
  bool show_odometry_keyframes;
  bool show_odometry_factors;

  bool show_submaps;
  bool show_submap_frames;
  bool show_factors;

  // Trajectories of each stage, drawn uncorrected in the world frame to compare drift directly
  bool show_odom_traj;
  bool show_submap_traj;
  bool show_global_traj;
  std::vector<Eigen::Vector3f> odom_traj;
  std::vector<Eigen::Vector3f> submap_traj;

  bool show_odometry_status;
  int last_id;
  int last_num_points;
  std::pair<double, double> last_point_stamps;
  std::chrono::steady_clock::time_point last_frame_time;
  double last_frame_stamp;
  double frontend_hz;  // Wall-clock rate of new odometry frames (EMA)
  double sensor_hz;    // Rate of the frame stamps (EMA)
  Eigen::Vector3d last_imu_vel;
  Eigen::Matrix<double, 6, 1> last_imu_bias;
  double last_median_distance;
  std::vector<double> last_voxel_resolutions;

  using FactorLine = std::tuple<Eigen::Vector3f, Eigen::Vector3f, Eigen::Vector4f, Eigen::Vector4f>;
  using FactorLineGetter = std::function<std::optional<FactorLine>(const gtsam::NonlinearFactor*)>;
  std::vector<std::pair<gtsam_points::weak_ptr<gtsam::NonlinearFactor>, FactorLineGetter>> odometry_factor_lines;
  std::unordered_map<std::uint64_t, Eigen::Isometry3f> odometry_poses;

  bool show_mapping_tools;
  float min_overlap;

  bool show_memory_stats;
  int submap_memstats_count;
  std::vector<SubMapMemoryStats> submap_memstats;

  int global_factor_stats_count;
  std::vector<FactorMemoryStats> global_factor_memstats;

  size_t total_gl_bytes;

  float point_size;

  // Coordinate axes markers, one style per group of drawables
  enum AxesType { SUBMAP_AXES, WINDOW_AXES, CURRENT_AXES };
  struct Axes {
    std::string name;                // UI label and config key
    std::string prefix;              // Name prefix of the drawables using this style
    Eigen::Vector2f length_radius;   // [m]
    std::shared_ptr<const glk::Drawable> drawable;
  };
  std::vector<Axes> axes;

  // Point cloud coloring, one style per group of drawables
  enum ColorMode { FLAT, HEIGHT, INTENSITY, NORMAL, INDEX };
  enum ColorGroup { CURRENT_COLOR, WINDOW_COLOR, LOCAL_MAP_COLOR, SUBMAP_COLOR, PANEL_COLOR };
  struct ColorStyle {
    std::string name;                   // UI label and settings key
    std::vector<std::string> prefixes;  // Name prefixes of the drawables using this style (*coord* drawables are skipped)
    bool panel;                         // Drawables live in the submap panel instead of the main viewer
    int mode;                           // ColorMode
    Eigen::Vector4f color;              // FLAT color
  };
  std::vector<ColorStyle> colors;

  float traj_width;     // [px]
  float factors_width;  // [px]
  bool point_size_metric;
  bool point_shape_circle;

  int z_range_mode;
  Eigen::Vector2f z_range;
  Eigen::Vector2f auto_z_range;
  double last_submap_z;
  float points_alpha;
  double factors_alpha;

  bool auto_intensity_range;
  Eigen::Vector2f intensity_range;
  gtsam_points::RunningStatistics<double> intensity_dist;

  std::unique_ptr<TrajectoryManager> trajectory;
  std::vector<Eigen::Isometry3f> submap_keyframes;

  std::vector<std::pair<int, int>> global_between_factors;

  std::mutex invoke_queue_mutex;
  std::vector<std::function<void()>> invoke_queue;

  // Logging
  std::shared_ptr<spdlog::logger> logger;
};
}  // namespace glim