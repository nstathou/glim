#pragma once

#include <any>
#include <atomic>
#include <memory>
#include <random>
#include <thread>
#include <gtsam_points/util/gtsam_migration.hpp>
#include <glim/util/concurrent_vector.hpp>
#include <glim/mapping/global_mapping_base.hpp>

namespace gtsam {
class Values;
class NonlinearFactor;
class NonlinearFactorGraph;
}  // namespace gtsam

namespace gtsam_points {
class ISAM2Ext;
class StreamTempBufferRoundRobin;

class GaussianVoxelMap;
class NearestNeighborSearch;
}  // namespace gtsam_points

namespace glim {

class IMUIntegration;

/**
 * @brief GlobalMappingPoseGraph parameters.
 */
struct GlobalMappingPoseGraphParams {
public:
  GlobalMappingPoseGraphParams();
  ~GlobalMappingPoseGraphParams();

public:
  bool enable_optimization;

  std::string registration_type;

  double min_travel_dist;
  double max_neighbor_dist;
  double min_inliear_fraction;

  int subsample_target;
  double subsample_rate;
  double gicp_max_correspondence_dist;
  double vgicp_voxel_resolution;

  double odom_factor_stddev;
  double odom_rot_stddev;           ///< Odometry factor rotation stddev [rad] (default: odom_factor_stddev)
  double odom_rot_stddev_per_m;     ///< Added rotation stddev per meter traveled between submaps [rad/m]
  double odom_trans_stddev;         ///< Odometry factor translation stddev [m] (default: odom_factor_stddev)
  double odom_trans_stddev_per_m;   ///< Added translation stddev per meter traveled between submaps [m/m]
  double odom_rot_stddev_per_rad;   ///< Added rotation stddev per radian rotated between submaps [rad/rad]
  double odom_trans_stddev_per_rad; ///< Added translation stddev per radian rotated between submaps [m/rad]
  double loop_factor_stddev;
  double loop_rot_stddev;  ///< Rotation stddev [rad] of the loop's best constrained rotation direction (with loop_use_registration_hessian)
  bool loop_use_registration_hessian;  ///< Shape the loop information with the registration Hessian (strongest direction = loop_factor_stddev)
  std::string loop_factor_robust_type;  ///< HUBER or CAUCHY
  double loop_factor_robust_width;
  int loop_batch_max_iterations;    ///< Batch LM over the whole pose graph when new loops arrive (0 = iSAM2 only)

  int loop_candidate_buffer_size;
  int loop_candidate_eval_per_thread;

  bool use_isam2_dogleg;
  double isam2_relinearize_skip;
  double isam2_relinearize_thresh;

  double init_pose_damping_scale;

  // Dense VGICP refinement run on save, initialized with the pose graph estimate
  int refine_max_iterations;  ///< 0 = disabled
  int refine_rounds;          ///< Rounds of overlapping-pair search + LM, stops when no new pairs appear
  double refine_voxel_resolution;
  int refine_voxelmap_levels;
  double refine_voxelmap_scaling_factor;
  double refine_randomsampling_rate;
  double refine_min_overlap;
  double refine_max_distance;
  double refine_odom_information_scale;  ///< Odometry factor information multiplier during refinement

  bool save_merged_pcd;

  int num_threads;
};

/// @brief Submap target
struct SubMapTarget {
  using Ptr = std::shared_ptr<SubMapTarget>;
  using ConstPtr = std::shared_ptr<const SubMapTarget>;

  SubMap::ConstPtr submap;
  gtsam_points::PointCloud::ConstPtr subsampled;
  std::shared_ptr<gtsam_points::NearestNeighborSearch> tree;
  std::shared_ptr<gtsam_points::GaussianVoxelMap> voxels;
  double travel_dist;
};

/// @brief Loop candidate
struct LoopCandidate {
  SubMapTarget::ConstPtr target;
  SubMapTarget::ConstPtr source;
  Eigen::Isometry3d init_T_target_source;
};

/**
 * @brief Global mapping with the old conventional pose graph optimization.
 * @note  We recommend using GlobalMapping instead of this class if accuracy matters.
 */
class GlobalMappingPoseGraph : public GlobalMappingBase {
public:
  GlobalMappingPoseGraph(const GlobalMappingPoseGraphParams& params = GlobalMappingPoseGraphParams());
  virtual ~GlobalMappingPoseGraph();

  virtual void insert_submap(const SubMap::Ptr& submap) override;

  virtual void optimize() override;

  virtual void save(const std::string& path) override;
  virtual gtsam_points::PointCloud::Ptr export_points() override;

private:
  void insert_submap(int current, const SubMap::Ptr& submap);

  std::shared_ptr<gtsam::NonlinearFactorGraph> create_odometry_factors(int current) const;
  void find_loop_candidates(int current);
  std::shared_ptr<gtsam::NonlinearFactorGraph> collect_detected_loops();

  void update_submaps();
  void optimize_batch();
  std::unique_ptr<gtsam::Values> refine();

  void loop_detection_task();

private:
  using Params = GlobalMappingPoseGraphParams;
  Params params;

  std::mt19937 mt;

  std::atomic_bool kill_switch;
  std::atomic_int loop_backlog;  ///< Loop candidates queued or being evaluated
  std::thread loop_detection_thread;
  ConcurrentVector<LoopCandidate> loop_candidates;
  ConcurrentVector<gtsam_points::shared_ptr<gtsam::NonlinearFactor>> detected_loops;

  std::vector<SubMap::Ptr> submaps;
  std::vector<SubMapTarget::Ptr> submap_targets;

  std::unique_ptr<gtsam::Values> new_values;
  std::unique_ptr<gtsam::NonlinearFactorGraph> new_factors;

  std::unique_ptr<gtsam_points::ISAM2Ext> isam2;

  std::shared_ptr<void> tbb_task_arena;
};
}  // namespace glim
