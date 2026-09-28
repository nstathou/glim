#include <glim/viewer/standard_viewer.hpp>

#include <fstream>
#include <filesystem>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <gtsam_points/config.hpp>
#include <gtsam_points/types/point_cloud_cpu.hpp>

#include <glim/odometry/callbacks.hpp>
#include <glim/odometry/estimation_frame.hpp>
#include <glim/mapping/callbacks.hpp>
#include <glim/util/config.hpp>
#include <glim/util/logging.hpp>
#include <glim/util/trajectory_manager.hpp>

#include <glk/colormap.hpp>
#include <glk/lines.hpp>
#include <glk/thin_lines.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/spdlog_sink.hpp>
#include <guik/viewer/light_viewer.hpp>

#include <glim/viewer/standard_viewer_mem.hpp>

namespace glim {

bool StandardViewer::drawable_filter(const std::string& name) {
  const auto starts_with = [](const std::string& name, const std::string& pattern) {
    if (name.size() < pattern.size()) {
      return false;
    }

    return std::equal(pattern.begin(), pattern.end(), name.begin());
  };

  if (!show_current_coord && name == "current_coord") {
    return false;
  }

  if (!show_current_points && name == "current_frame") {
    return false;
  }

  if (!show_odometry_scans && starts_with(name, "frame_")) {
    return false;
  }

  if (!show_odometry_keyframes && starts_with(name, "odometry_keyframe_")) {
    return false;
  }

  if (!show_odometry_factors && starts_with(name, "odometry_factors")) {
    return false;
  }

  if (!show_submap_frames && starts_with(name, "submap_coord_")) {
    return false;
  }

  if (!show_submaps && starts_with(name, "submap_") && !starts_with(name, "submap_coord_")) {
    return false;
  }

  if (!show_factors && starts_with(name, "factors")) {
    return false;
  }

  if ((!show_odom_traj && name == "traj_odom") || (!show_submap_traj && name == "traj_submap") || (!show_global_traj && name == "traj_global")) {
    return false;
  }

  return true;
}

namespace {
void set_line_width(const glk::Drawable::ConstPtr& drawable, float width) {
  auto lines = std::dynamic_pointer_cast<const glk::ThinLines>(drawable);
  if (lines) {
    std::const_pointer_cast<glk::ThinLines>(lines)->set_line_width(width);  // !!
  }
}
}  // namespace

void StandardViewer::apply_camera_mode() {
  auto viewer = guik::LightViewer::instance();
  switch (camera_mode) {
    default:
    case 0:
      viewer->use_orbit_camera_control();
      break;
    case 1:
      viewer->use_sensor_view_camera_control(Eigen::Translation3f(-0.05f, 0.0f, 0.0f) * Eigen::Isometry3f::Identity(), 1e-3, 1e-3);
      break;
    case 2:
      viewer->use_sensor_view_camera_control();
      break;
    case 3:
      viewer->use_topdown_camera_control();
      break;
  }
}

// Load (save=false) or save (save=true) the UI settings in ~/.config/glim/standard_viewer.json.
// Loaded values override config_viewer.json. Delete the file to go back to the config defaults.
void StandardViewer::sync_settings(bool save) {
  const char* home = std::getenv("HOME");
  const std::filesystem::path path = std::filesystem::path(home ? home : ".") / ".config/glim/standard_viewer.json";

  nlohmann::json json;
  if (!save) {
    std::ifstream ifs(path);
    if (!ifs) {
      return;
    }
    json = nlohmann::json::parse(ifs, nullptr, false);
    if (json.is_discarded()) {
      logger->warn("failed to parse {}", path.string());
      return;
    }
  }

  try {
    const auto field = [&](const std::string& name, auto& value) {
      if (save) {
        json[name] = value;
      } else if (json.contains(name)) {
        json.at(name).get_to(value);
      }
    };
    const auto field2 = [&](const std::string& name, Eigen::Vector2f& value) {
      std::array<float, 2> v = {value[0], value[1]};
      field(name, v);
      value << v[0], v[1];
    };

    field("track", track);
    field("show_current_coord", show_current_coord);
    field("show_current_points", show_current_points);
    field("show_odometry_scans", show_odometry_scans);
    field("show_odometry_keyframes", show_odometry_keyframes);
    field("show_odometry_factors", show_odometry_factors);
    field("show_submaps", show_submaps);
    field("show_submap_frames", show_submap_frames);
    field("show_factors", show_factors);
    field("show_odom_traj", show_odom_traj);
    field("show_submap_traj", show_submap_traj);
    field("show_global_traj", show_global_traj);
    field("camera_mode", camera_mode);
    for (auto& style : colors) {
      field(style.name + "_color_mode", style.mode);
      std::array<float, 3> rgb = {style.color[0], style.color[1], style.color[2]};
      field(style.name + "_color", rgb);
      style.color << rgb[0], rgb[1], rgb[2], 1.0f;
    }
    field("z_range_mode", z_range_mode);
    field2("z_range", z_range);
    field("auto_intensity_range", auto_intensity_range);
    field2("intensity_range", intensity_range);
    field("point_size", point_size);
    field("points_alpha", points_alpha);
    field("traj_width", traj_width);
    field("factors_width", factors_width);
    for (auto& a : axes) {
      field2(a.name, a.length_radius);
    }
  } catch (const nlohmann::json::exception& e) {
    logger->warn("failed to {} {}: {}", save ? "save" : "load", path.string(), e.what());
    return;
  }

  if (save) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    std::ofstream ofs(path);
    ofs << json.dump(2) << std::endl;
    if (!ofs) {
      logger->warn("failed to save {}", path.string());
      return;
    }
    logger->info("viewer settings saved to {}", path.string());
  }
}

void StandardViewer::apply_color(guik::ShaderSetting& shader_setting, int group, int id) const {
  const auto& style = colors[group];
  switch (style.mode) {
    default:
    case FLAT:
      shader_setting.set_color_mode(guik::ColorMode::FLAT_COLOR).set_color(style.color);
      break;
    case HEIGHT:
      shader_setting.set_color_mode(guik::ColorMode::RAINBOW);
      break;
    case INTENSITY:
      shader_setting.set_color_mode(guik::ColorMode::VERTEX_COLORMAP);
      break;
    case NORMAL:
      shader_setting.set_color_mode(guik::ColorMode::VERTEX_COLOR);
      break;
    case INDEX:
      shader_setting.set_color_mode(guik::ColorMode::FLAT_COLOR).set_color(glk::colormap_categoricalf(glk::COLORMAP::TURBO, id, 16));
      break;
  }

  if (group == SUBMAP_COLOR) {
    shader_setting.set_alpha(points_alpha);
  }
}

void StandardViewer::update_colors(int group) {
  const auto& style = colors[group];
  auto viewer = guik::LightViewer::instance();
  auto context = style.panel ? viewer->sub_viewer("submap").get() : viewer;

  for (auto& [name, drawable] : context->get_drawables()) {
    if (name.find("coord") != std::string::npos) {
      continue;
    }
    for (const auto& prefix : style.prefixes) {
      if (name.rfind(prefix, 0) == 0) {
        apply_color(*drawable.first, group, std::atoi(name.substr(name.rfind('_') + 1).c_str()));
      }
    }
  }
}

void StandardViewer::update_axes() {
  // glk::Lines width is in world units, so the axes get a real thickness (unlike the thin-line coordinate_system primitive)
  const auto make_axes = [](const Eigen::Vector2f& length_radius) {
    const float l = length_radius[0];
    const std::vector<Eigen::Vector3f> vertices = {
      Eigen::Vector3f::Zero(),
      Eigen::Vector3f(l, 0.0f, 0.0f),
      Eigen::Vector3f::Zero(),
      Eigen::Vector3f(0.0f, l, 0.0f),
      Eigen::Vector3f::Zero(),
      Eigen::Vector3f(0.0f, 0.0f, l)};
    const std::vector<Eigen::Vector4f> colors = {
      Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f),
      Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f),
      Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f),
      Eigen::Vector4f(0.0f, 1.0f, 0.0f, 1.0f),
      Eigen::Vector4f(0.0f, 0.0f, 1.0f, 1.0f),
      Eigen::Vector4f(0.0f, 0.0f, 1.0f, 1.0f)};
    return std::make_shared<glk::Lines>(2.0f * length_radius[1], vertices, colors);
  };

  for (auto& a : axes) {
    a.drawable = make_axes(a.length_radius);
  }

  for (auto& [name, drawable] : guik::LightViewer::instance()->get_drawables()) {
    for (const auto& a : axes) {
      if (name.rfind(a.prefix, 0) == 0) {
        drawable.second = a.drawable;
      }
    }
  }
}

void StandardViewer::drawable_selection() {
  auto viewer = guik::LightViewer::instance();

  ImGui::SetWindowPos("images", {1800, 60}, ImGuiCond_FirstUseEver);
  ImGui::SetWindowPos("logging", {1800, 950}, ImGuiCond_FirstUseEver);

  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.5));
  ImGui::Begin("selection", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
  ImGui::PopStyleColor();

  if (ImGui::Checkbox("track", &track)) {
    if (track) {
      guik::LightViewer::instance()->reset_center();
    }
  }
  ImGui::SameLine();
  bool show_current = show_current_coord || show_current_points;
  if (ImGui::Checkbox("current", &show_current)) {
    show_current_coord = show_current_points = show_current;
  }
  ImGui::SameLine();
  ImGui::Checkbox("coord", &show_current_coord);
  ImGui::SameLine();
  ImGui::Checkbox("points", &show_current_points);
  ImGui::SameLine();
  if (ImGui::Button("Save view")) {
    sync_settings(true);
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Save the viewer settings (also done on exit)");

  ImGui::Separator();
  bool show_odometry = show_odometry_scans || show_odometry_keyframes || show_odometry_factors;
  if (ImGui::Checkbox("odometry", &show_odometry)) {
    show_odometry_scans = show_odometry_keyframes = show_odometry;
    show_odometry_factors &= show_odometry;
  }

  ImGui::SameLine();
  if (ImGui::Button("Status")) {
    show_odometry_status = true;
  }

  ImGui::Checkbox("window scans##odom", &show_odometry_scans);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Scans (points + axes) in the odometry smoother window");
  ImGui::SameLine();
  ImGui::Checkbox("local map##odom", &show_odometry_keyframes);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Snapshot of the odometry scan matching target map (refreshed every 50 frames)");
  ImGui::SameLine();
  ImGui::Checkbox("smoother factors##odom", &show_odometry_factors);
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Constraints in the odometry smoother (scan matching, IMU)");

  ImGui::Separator();
  bool show_mapping = show_submaps || show_submap_frames || show_factors;
  if (ImGui::Checkbox("mapping", &show_mapping)) {
    show_submaps = show_submap_frames = show_factors = show_mapping;
  }

  ImGui::SameLine();
  if (ImGui::Button("Tools")) {
    show_mapping_tools = true;
  }

  ImGui::SameLine();
  if (ImGui::Button("Mem stats")) {
    show_memory_stats = true;
  }

  ImGui::SameLine();
  if (ImGui::Button("Log")) {
    viewer->register_ui_callback("logging", guik::create_logger_ui(glim::get_ringbuffer_sink(), 0.5));
  }

  ImGui::Checkbox("submap points", &show_submaps);
  ImGui::SameLine();
  ImGui::Checkbox("submap frames", &show_submap_frames);
  ImGui::SameLine();
  ImGui::Checkbox("factors", &show_factors);

  ImGui::Separator();
  ImGui::Text("traj");
  ImGui::SameLine();
  ImGui::Checkbox("odom##traj", &show_odom_traj);
  ImGui::SameLine();
  ImGui::Checkbox("submap##traj", &show_submap_traj);
  ImGui::SameLine();
  ImGui::Checkbox("global##traj", &show_global_traj);

  ImGui::Separator();

  std::vector<const char*> camera_modes = {"STANDARD", "FPS", "TPS", "TOPDOWN"};
  ImGui::SetNextItemWidth(92);
  if (ImGui::Combo("camera_mode", &camera_mode, camera_modes.data(), camera_modes.size())) {
    apply_camera_mode();
  }

  const std::vector<const char*> color_modes = {"FLAT", "HEIGHT", "INTENSITY", "NORMAL", "INDEX"};
  bool any_height = false;
  bool any_intensity = false;
  for (int group = 0; group < colors.size(); group++) {
    auto& style = colors[group];
    bool changed = false;
    ImGui::SetNextItemWidth(92);
    changed |= ImGui::Combo(style.name.c_str(), &style.mode, color_modes.data(), color_modes.size());
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("HEIGHT: z_range rainbow, NORMAL: |normal| as RGB, INDEX: one color per scan/submap");
    if (style.mode == FLAT) {
      ImGui::SameLine();
      changed |= ImGui::ColorEdit3(("##color_" + style.name).c_str(), style.color.data(), ImGuiColorEditFlags_NoInputs);
    }
    if (changed) {
      update_colors(group);
    }
    any_height |= style.mode == HEIGHT;
    any_intensity |= style.mode == INTENSITY;
  }

  ImGui::SetNextItemWidth(92);
  if (ImGui::DragFloat("point_size", &point_size, 0.001f, 0.001f, 10.0f, "%.3f")) {
    viewer->shader_setting().set_point_size(point_size);
  }

  ImGui::SetNextItemWidth(92);
  if (ImGui::SliderFloat("points_alpha", &points_alpha, 0.0f, 1.0f)) {
    for (int i = 0;; i++) {
      const auto found = viewer->find_drawable("submap_" + std::to_string(i));
      if (!found.first) {
        break;
      }
      found.first->set_alpha(points_alpha).make_transparent();
    }
  }

  bool axes_changed = false;
  for (auto& a : axes) {
    ImGui::SetNextItemWidth(150);
    axes_changed |= ImGui::DragFloat2((a.name + " (len, rad)").c_str(), a.length_radius.data(), 0.005f, 0.001f, 100.0f, "%.3f");
  }
  if (axes_changed) {
    update_axes();
  }

  ImGui::SetNextItemWidth(92);
  if (ImGui::DragFloat("traj_width", &traj_width, 0.1f, 1.0f, 20.0f, "%.1f")) {
    for (const auto& name : {"traj_odom", "traj_submap", "traj_global"}) {
      set_line_width(viewer->find_drawable(name).second, traj_width);
    }
  }

  ImGui::SetNextItemWidth(92);
  if (ImGui::DragFloat("factors_width", &factors_width, 0.1f, 1.0f, 20.0f, "%.1f")) {
    set_line_width(viewer->find_drawable("factors").second, factors_width);
  }

  if (any_height) {
    std::vector<const char*> z_range_modes = {"AUTO", "LOCAL", "MANUAL"};
    bool update_z_range = false;

    ImGui::SetNextItemWidth(150);
    update_z_range |= ImGui::Combo("z_range_mode", &z_range_mode, z_range_modes.data(), z_range_modes.size());
    update_z_range |= ImGui::DragFloatRange2("z_range", &z_range[0], &z_range[1], 0.1f, -10000.0f, 10000.0f);
    if (update_z_range) {
      Eigen::Vector2f z = z_range;
      if (z_range_mode == 0) {
        z += auto_z_range;
      } else if (z_range_mode == 1) {
        z += Eigen::Vector2f::Constant(last_submap_z);
      }
      viewer->shader_setting().add<Eigen::Vector2f>("z_range", z);
    }
  }

  if (any_intensity) {
    ImGui::Checkbox("auto_intensity_range", &auto_intensity_range);
    if (auto_intensity_range) {
      intensity_range[0] = intensity_dist.min();
      intensity_range[1] = intensity_dist.max();
    }

    ImGui::SetNextItemWidth(150);
    ImGui::DragFloatRange2("intensity_range", &intensity_range[0], &intensity_range[1], 0.1f, -65536.0f, 65536.0f);
    viewer->shader_setting().add<Eigen::Vector2f>("cmap_range", Eigen::Vector2f(intensity_range[0], intensity_range[1]));
    viewer->sub_viewer("submap")->shader_setting().add<Eigen::Vector2f>("cmap_range", Eigen::Vector2f(intensity_range[0], intensity_range[1]));
  }

  ImGui::End();

  if (show_odometry_status) {
    ImGui::Begin("odometry status", &show_odometry_status, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("frame ID:%d", last_id);
    ImGui::Text("rate:%.1f Hz (sensor %.1f Hz)", frontend_hz, sensor_hz);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Wall-clock rate of odometry output vs rate of the scan stamps. Front-end lags if lower than sensor.");
    ImGui::Text("points:%d", last_num_points);
    ImGui::Text("median dist:%.3f", last_median_distance);

    // Only GPU odometry attaches voxelmaps to frames
    if (!last_voxel_resolutions.empty()) {
      std::stringstream sst;
      sst << "voxel_resolution: ";
      for (double r : last_voxel_resolutions) {
        sst << fmt::format("{:.3f}", r) << " ";
      }
      const std::string text = sst.str();
      ImGui::Text("%s", text.c_str());
    }

    ImGui::Text("point time:%.3f ~ %.3f s", last_point_stamps.first, last_point_stamps.second);
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("First/last per-point time offset of the raw scan, relative to the scan stamp (i.e. scan duration)");
    ImGui::Text("vel:%.3f %.3f %.3f", last_imu_vel[0], last_imu_vel[1], last_imu_vel[2]);
    ImGui::Text("bias:%.3f %.3f %.3f %.3f %.3f %.3f", last_imu_bias[0], last_imu_bias[1], last_imu_bias[2], last_imu_bias[3], last_imu_bias[4], last_imu_bias[5]);
    ImGui::End();
  }

  if (show_mapping_tools) {
    ImGui::Begin("mapping tools", &show_mapping_tools, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::DragFloat("Min overlap", &min_overlap, 0.01f, 0.01f, 1.0f);
    if (ImGui::Button("Find overlapping submaps")) {
      logger->info("finding overlapping submaps...");
      GlobalMappingCallbacks::request_to_find_overlapping_submaps(min_overlap);
    }

    if (ImGui::Button("Optimize")) {
      logger->info("optimizing...");
      GlobalMappingCallbacks::request_to_optimize();
    }
    ImGui::End();
  }

  if (show_memory_stats) {
    ImGui::Begin("memory stats", &show_memory_stats, ImGuiWindowFlags_AlwaysAutoResize);

    size_t points_cpu = 0;
    size_t points_gpu = 0;
    size_t voxelmap_cpu = 0;
    size_t voxelmap_gpu = 0;
    size_t odom_cpu = 0;
    size_t odom_gpu = 0;

    for (const auto& m : submap_memstats) {
      points_cpu += m.frame_cpu_bytes;
      points_gpu += m.frame_gpu_bytes;
      voxelmap_cpu += m.voxelmap_cpu_bytes;
      voxelmap_gpu += m.voxelmap_gpu_bytes;
      odom_cpu += m.odom_cpu_bytes;
      odom_gpu += m.odom_gpu_bytes;
    }

    size_t factors_cpu = 0;
    size_t factors_gpu = 0;

    for (const auto& m : global_factor_memstats) {
      factors_cpu += m.cpu_bytes;
      factors_gpu += m.gpu_bytes;
    }

    constexpr double mb = 1.0 / (1024.0 * 1024.0);
    const size_t total_cpu = points_cpu + voxelmap_cpu + odom_cpu + factors_cpu;
    const size_t total_gpu = points_gpu + voxelmap_gpu + odom_gpu + factors_gpu + total_gl_bytes;
    const double total_cpu_mb = total_cpu * mb;
    const double total_gpu_mb = total_gpu * mb;

    ImGui::Text("Global mapping memory usage");
    if (ImGui::BeginTable("Global mapping memory usage", 3, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
      const auto show_item = [=](const char* name, size_t cpu, size_t gpu) {
        const double cpu_mb = cpu * mb;
        const double gpu_mb = gpu * mb;

        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%s", name);
        ImGui::TableNextColumn();
        ImGui::Text("%.2f MB / %.1f %%", cpu_mb, cpu_mb / total_cpu_mb * 100.0);
        ImGui::TableNextColumn();
        ImGui::Text("%.2f MB / %.1f %%", gpu * mb, gpu_mb / total_gpu_mb * 100.0);
      };

      ImGui::TableSetupColumn("Item");
      ImGui::TableSetupColumn("CPU");
      ImGui::TableSetupColumn("GPU");
      ImGui::TableHeadersRow();

      show_item("Total", total_cpu, total_gpu);
      show_item("Points", points_cpu, points_gpu);
      show_item("Voxelmap", voxelmap_cpu, voxelmap_gpu);
      show_item("Odom frames", odom_cpu, odom_gpu);
      show_item("Factors", factors_cpu, factors_gpu);
      show_item("OpenGL", 0, total_gl_bytes);

      ImGui::EndTable();
    }

    ImGui::End();
  }
}
}  // namespace glim