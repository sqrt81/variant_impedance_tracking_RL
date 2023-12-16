#ifndef A1_GYM_ENV_H_
#define A1_GYM_ENV_H_

#include <Eigen/Eigen>

#include <mutex>
#include <set>
#include <deque>

#include "param_loader.h"

struct mjModel_;
struct mjData_;
struct mjvCamera_;
struct mjvOption_;
struct mjvScene_;
struct mjrContext_;

namespace mujoco_parallel_sim {

class A1GymEnv
{
public:
    constexpr static int kACT_SIZE = 72;
    constexpr static int kOBS_SIZE = 67;

    ATTR_API A1GymEnv() = default;

    ATTR_API A1GymEnv(const A1GymEnv &other) = delete;

    ATTR_API A1GymEnv(A1GymEnv &&other);

    ATTR_API ~A1GymEnv();

    // Must call after SetSimPars is called
    ATTR_API void InitFromMujocoModel(const mjModel_ *model_ref,
                                      const A1RLParam &param);

    ATTR_API void Step(const double *action,
              double *obs, bool &done, double &reward, bool &timeout);

    // applying external impulse on torso during next step
    void ApplyImpulse(double momentum_x, double momentum_y)
    {
        external_momentum_x_ = momentum_x;
        external_momentum_y_ = momentum_y;
    }

    void StartRendering(
            mjvCamera_  *camera,
            mjvOption_  *render_options,
            mjvScene_   *scene,
            mjrContext_ *context)
    {
        std::lock_guard g(render_mutex_);
        make_context_required_ = true;
        camera_ = camera;
        render_options_ = render_options;
        scene_ = scene;
        context_ = context;
    }

    // it seems this function must be called on main thread
    ATTR_API void Render();

    void StopRendering()
    {
        std::lock_guard g(render_mutex_);
        camera_ = nullptr;
        render_options_ = nullptr;
        scene_ = nullptr;
        context_ = nullptr;
    }

    // Sim and terrain params are applied in the next eposide
    // Must call before InitFromMujocoModel() or the first eposide
    // is unintialized.
    void SetSimPars(
            const Eigen::Vector2d &xy_offset,
            double raise_height)
    {
        foot_xy_offset_ = xy_offset;
        raise_height_ = raise_height;
    }

    void SetTerrain(
            double terrain_height,
            int terrain_id)
    {
        terrain_height_ = terrain_height;
        terrain_id_ = terrain_id;
        change_terrain_required_ = true;
    }

    void GetStatics(double &vel_mean,
                    double &elapsed_time,
                    double &terrain_height)
    {
        vel_mean = last_mean_x_vel_;
        elapsed_time = last_elapsed_time_;
        terrain_height = last_terrain_height_;
    }

    void RequireReset()
    {
        traj_change_cnt_ = -1;
        ResetEnv();
    }

    void GetDifficulty(double &v_cmd, double &terrain_height)
    {
        v_cmd = target_vel_.x();
        terrain_height = terrain_height_;
    }

    void GetMaxDifficulty(double &v_cmd, double &terrain_height)
    {
        v_cmd = config_->vel_x_max;
        terrain_height = config_->terrain_height;
    }

private:
    struct TrajConfig {
        Eigen::Vector3d stance_pos_offset[4];
        Eigen::Vector3d jpos_beg[4];
        Eigen::Vector3d jvel_beg[4];
        Eigen::Vector3d jacc_beg[4];
        Eigen::Vector3d jpos_end[4];
        Eigen::Vector3d jvel_end[4];
        Eigen::Vector3d jacc_end[4];
        Eigen::Vector3d jpos_mid[4];
        Eigen::Vector3d jvel_mid[4];
        Eigen::Vector3d start_lift_jpos[4];
        Eigen::Vector3d start_lift_jvel[4];
        Eigen::Matrix<double, 2, 3> start_swing_par[4];
        Eigen::Vector2d vel;
        double phase_offset[4];
        double knee_inc[4];
        double raise_time_ratio;
        double duty_ratio;
        double period;
        double torso_height;
        double raise_acc;
        double half_stance_time;
        double half_cartesian_time;
        double half_joint_time;
        double t_acc;
    };

    using Array12d = Eigen::Array<double, 12, 1>;
    using InputArray = Eigen::Array<double, kACT_SIZE, 1>;

    void ApplyControlForce(const double *action);

    void ComputeReward(double &reward, bool &done, bool &timeout);

    void ComputeObservations(double *obs);

    ATTR_API void ResetEnv();

    // defines new target
    void Curriculum();

    void BuildTrajConfig();

    void GenerateTrajBuffer();

    void DynamicsRandomization();

    void InitStateRandomization();

    void ChangeTerrain();

    void ComputeTargetJointPos(
            double time, Array12d &jpos, bool *feet_td) const;

    void TargetJPosOnStart(
            double time, Array12d &jpos, bool *feet_td) const;

    void TargetJPosOnStart2(
            double time, Array12d &jpos, bool *feet_td) const;

    void TargetJPosFromBuffer(
            double time, Array12d &jpos, bool *feet_td) const;

    void StartJPosFromBuffer(
            double time, Array12d &jpos, bool *feet_td) const;

    bool HasCollision() const;

    // mujoco model
    const mjModel_  *ref_model_ = nullptr;
    mjModel_    *model_ = nullptr;
    mjData_     *data_ = nullptr;
    double step_dt_;
    // only foot are allowed to touch other bodies
    std::set<int> unallowed_contact_body_;
    int terrain_geom_id_; // geom id for uneven terrain
    int obstacle_id_[2]; // geom id for obstacles

    // updated in ApplyControlForce()
    bool feet_td_[4];
    Array12d target_jpos_;
    Array12d target_jvel_;
    Array12d jpos_err_;
    Array12d jvel_err_;
    Array12d last_target_jpos_;

    // updated in step()
    std::deque<InputArray> delayed_actions_;
//    double last_action_[kACT_SIZE]; // this is for delayed action
    Array12d average_abs_torq_square_;
    double average_pos_jerk_;
    double average_vel_jerk_;
    bool collision_;
    Eigen::Quaterniond quat_;
    Eigen::Vector3d rpy_;
    double mean_x_vel_;
    double push_timer_ = 0.;
    double turn_timer_ = 0.;
    Eigen::Vector2d cur_vel_cmd_;
    double external_momentum_x_;
    double external_momentum_y_;

    // rendering
    std::mutex render_mutex_;
    bool make_context_required_ = false;
    mjvCamera_  *camera_ = nullptr;
    mjvOption_  *render_options_ = nullptr;
    mjvScene_   *scene_ = nullptr;
    mjrContext_ *context_ = nullptr;

    // config for simulation parameters
    std::mutex config_mutex_;
    bool change_terrain_required_;
    double terrain_height_;
    int terrain_id_;
    Eigen::Quaterniond env_ori_;

    // config for trajectory generation
    Eigen::Vector2d target_vel_;
    Eigen::Vector2d foot_xy_offset_;
    double duty_ratio_;
    double period_;
    double raise_height_;
    double torso_height_;
    double target_yaw_;
    TrajConfig traj_cfg_;
    int traj_change_cnt_ = -1;  // if difficulty is not changed,
                                // change trajectory per 5 eposide

    // generated trajectory buffer
    std::vector<Array12d> traj_buffer_;
    std::vector<std::array<bool, 4>> td_buffer_;
    std::vector<Array12d> start_traj_buffer_;
    std::vector<std::array<bool, 4>> start_td_buffer_;

    // config for building obs & rewards
    const A1RLParam *config_;

    // statics for last eposide
    double last_mean_x_vel_;
    double last_elapsed_time_;
    double last_terrain_height_;
};

} // namespace mujoco_parallel_sim

#endif // A1_GYM_ENV_H_
