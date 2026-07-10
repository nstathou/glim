# GLIM configuration files

## How configuration is resolved

`config.json` is the **entry point**. Its `"global"` section maps logical config
names to the actual files in this directory:

```json
"global": {
  "config_path": "",                              // filled in at runtime
  "config_odometry": "config_odometry_cpu.json",  // logical name -> file
  ...
}
```

At startup, `glim_ros` reads the ROS parameter `config_path` (default: `"config"`,
resolved relative to the installed `glim` package share directory if not absolute)
and loads `<config_path>/config.json` into the `GlobalConfig` singleton. Each module
then asks for its config by logical name, e.g. `GlobalConfig::get_config_path("config_odometry")`,
which returns `<config_path>/<filename listed in config.json>`.

To run with a custom config set, copy this whole directory and point glim at it:

```bash
ros2 run glim_ros glim_rosbag <bag> --ros-args -p config_path:=/absolute/path/to/my_config
```

All files may contain `//` and `/* */` comments.

## File overview

| File | Loaded via | Configures |
|---|---|---|
| `config.json` | entry point | Which file each module loads (module composition) |
| `config_ros.json` | `config_ros` | ROS I/O: topics, QoS, TF frame IDs, time offsets, extension modules to load |
| `config_logging.json` | `config_logging` | Log level and log directory |
| `config_sensors.json` | `config_sensors` | Sensor properties: `T_lidar_imu` extrinsics, IMU noise/bias parameters, point field names |
| `config_preprocess.json` | `config_preprocess` | Point cloud preprocessing: downsampling, distance filtering, outlier removal |
| `config_odometry_*.json` | `config_odometry` | Odometry estimation frontend |
| `config_sub_mapping_*.json` | `config_sub_mapping` | Local mapping (submap construction) |
| `config_global_mapping_*.json` | `config_global_mapping` | Global mapping backend (loop closure, graph optimization) |
| `config_viewer.json` | `config_viewer` | Standard viewer settings |

## Composing config.json

Modules with several `*_cpu` / `*_gpu` / other variants are swapped by changing the
filename in `config.json` — nothing else needs to change:

| Logical name | Variants |
|---|---|
| `config_odometry` | `config_odometry_cpu.json` (GICP, multi-thread CPU), `config_odometry_gpu.json` (VGICP, needs CUDA build), `config_odometry_ct.json` (continuous-time ICP, no IMU) |
| `config_sub_mapping` | `config_sub_mapping_cpu.json`, `config_sub_mapping_gpu.json`, `config_sub_mapping_passthrough.json` (no submap optimization, minimal latency) |
| `config_global_mapping` | `config_global_mapping_cpu.json`, `config_global_mapping_gpu.json`, `config_global_mapping_pose_graph.json` (lightweight pose-graph-only backend) |

Example — CPU odometry with a pose-graph-only backend:

```json
"global": {
  "config_odometry": "config_odometry_cpu.json",
  "config_sub_mapping": "config_sub_mapping_passthrough.json",
  "config_global_mapping": "config_global_mapping_pose_graph.json",
  ...
}
```

Notes:

- GPU variants require glim to be built with `-DBUILD_WITH_CUDA=ON`.
- Extension modules (e.g. `libimu_validator.so` from `glim_ext`) are listed in
  `config_ros.json` under `glim_ros/extension_modules`. Their own configs are
  resolved separately via the `glim_ext` package share directory (`config_ext`),
  not through this folder.
- On every run, the effective configuration is dumped alongside the map output
  (e.g. `/tmp/dump/config/`), which is useful to see exactly what was used.
