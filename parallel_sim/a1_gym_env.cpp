#include "a1_gym_env.h"

#include <iostream>
#include <mujoco/mujoco.h>

namespace mujoco_parallel_sim {

namespace {

template <typename T>
inline T square(T x)
{
    return x * x;
}

inline double random_scale(double half_range)
{
    return 1. + (rand() * half_range / RAND_MAX * 2) - half_range;
}

inline double rand_float()
{
    return rand() * 2. / RAND_MAX - 1.;
}

std::mutex err_mutex;

inline Eigen::Vector3d Quaternion2EulerZYX(const Eigen::Quaterniond &q)
{
    Eigen::Vector3d rpy;

    const double sinr_cosp = 2 * (q.w() * q.x() + q.y() * q.z());
    const double cosr_cosp = 1 - 2 * (square(q.x()) + square(q.y()));
    rpy(0) = std::atan2(sinr_cosp, cosr_cosp);

    const double sinp = 2 * (q.w() * q.y() - q.z() * q.x());
    if (std::abs(sinp) >= 1)
        rpy(1) = std::copysign(M_PI_2, sinp);
    else
        rpy(1) = std::asin(sinp);

    const double siny_cosp = 2 * (q.w() * q.z() + q.x() * q.y());
    const double cosy_cosp = 1 - 2 * (square(q.y()) + square(q.z()));
    rpy(2) = std::atan2(siny_cosp, cosy_cosp);

    return rpy;
}

using Arr3CRef = const Eigen::Map<const Eigen::Array3d>;

// A1 params
constexpr double hip_width_ = 0.08505;
constexpr double thigh_len_ = 0.2;
constexpr double shin_len_ = 0.2;

constexpr double joint_min_[] = {
    -0.4, -0.9, -2.5, -0.4, -0.9, -2.5, -0.4, -0.9, -2.5, -0.4, -0.9, -2.5
};
constexpr double joint_max_[] = {
    0.4, 1.4, -0.9, 0.4, 1.4, -0.9, 0.4, 1.4, -0.9, 0.4, 1.4, -0.9
};
constexpr double joint_vel_max_[] = {
    45., 30., 30., 45., 30., 30., 45., 30., 30., 45., 30., 30.
};

Eigen::Vector3d A1InvKin(const Eigen::Vector3d &pos)
{
    const double l2_2 = square(thigh_len_);
    const double l3_2 = square(shin_len_);
    const double n_2 = square(pos.y()) + square(pos.z());
    const double n = std::sqrt(n_2);

    const double q1 = std::acos(std::clamp(hip_width_ / n, -1., 1.))
            + std::atan2(pos.z(), pos.y());
    const double w_2 = n_2 - square(hip_width_);
    const double w = std::sqrt(w_2);
    const double l_2 = square(pos.x()) + w_2;
    const double l = std::sqrt(l_2);
    const double phi = - std::atan(pos.x() / w);
    const double cos_alpha = (l2_2 + l_2 - l3_2) / (2 * thigh_len_ * l);
    const double cos_beta = (l_2 - l2_2 - l3_2) / (2 * thigh_len_ * shin_len_);
    const double alpha = std::acos(std::clamp(cos_alpha, -1., 1.));
    const double beta = std::acos(std::clamp(cos_beta, -1., 1.));
    const double q2 = phi + alpha;
    const double q3 = std::min(- beta, - 0.95);

    return Eigen::Vector3d(q1, q2, q3);
}

// jacobian for FL leg
Eigen::Matrix3d A1Jacobian(const Eigen::Vector3d &jpos)
{
    const double c1 = std::cos(jpos[0]);
    const double s1 = std::sin(jpos[0]);
    const double c2 = std::cos(jpos[1]);
    const double s2 = std::sin(jpos[1]);
    const double c23 = std::cos(jpos[1] + jpos[2]);
    const double s23 = std::sin(jpos[1] + jpos[2]);
    const double shin_z = - thigh_len_ * c2 - shin_len_ * c23;
    const double shin_x = - thigh_len_ * s2 - shin_len_ * s23;

    Eigen::Matrix3d jacobian = Eigen::Matrix3d::Zero();
    jacobian(0, 1) = shin_z;
    jacobian(0, 2) = - shin_len_ * c23;
    jacobian(1, 0) = - shin_z * c1 - hip_width_ * s1;
    jacobian(1, 1) = shin_x * s1;
    jacobian(1, 2) = - shin_len_ * s23 * s1;
    jacobian(2, 0) = - shin_z * s1 + hip_width_ * c1;
    jacobian(2, 1) = - shin_x * c1;
    jacobian(2, 2) = shin_len_ * s23 * c1;

    return jacobian;
}

// generate different terrain
void UnevenTerrain(mjModel &model, int terrain_geom_id, double terrain_height)
{
    const int nsize = model.hfield_nrow[0] * model.hfield_ncol[0];
    for (int i = 0; i < nsize; i++)
        model.hfield_data[i] = rand() * 1. / RAND_MAX;

    model.hfield_size[2] = terrain_height;
    model.geom_pos[terrain_geom_id * 3 + 2] = - terrain_height / 2;
}

void Stairs(mjModel &model, int terrain_geom_id, double terrain_height)
{
    const int nrow = model.hfield_nrow[0];
    const int ncol = model.hfield_ncol[0];
    // the width of the stairs are set to be in range of 0.1m to 0.4m
    const int col_max = 0.4 * ncol / model.hfield_size[0];
    const int col_min = 0.1 * ncol / model.hfield_size[0];
    const int col_range = col_max - col_min;
    int next_col = -1;
    double col_height = 0;
    for (int i = 0; i < ncol; i++) {
        auto data_ptr = model.hfield_data + i;
        if (i > next_col) {
            // decide height and next height change row
            col_height = col_height > 0.5 ? 0. : 1.;
            next_col += col_min
                    + (int)(((long long)col_range) * rand() / RAND_MAX);
        }
        for (int j = 0; j < nrow; j++) {
            *data_ptr = col_height;
            data_ptr += ncol;
        }
    }

    model.hfield_size[2] = terrain_height;
    model.geom_pos[terrain_geom_id * 3 + 2] = - terrain_height / 2;
}

void SideStairs(mjModel &model, int terrain_geom_id, double terrain_height)
{
    const int nrow = model.hfield_nrow[0];
    const int ncol = model.hfield_ncol[0];
    // the width of the stairs are set to be in range of 0.05m to 0.15m
    const int row_max = 0.15 * ncol / model.hfield_size[0];
    const int row_min = 0.05 * ncol / model.hfield_size[0];
    const int row_range = row_max - row_min;
    int next_row = -1;
    double row_height = 0;
    auto data_ptr = model.hfield_data;
    for (int i = 0; i < nrow; i++) {
        if (i > next_row) {
            // flip height and decide next height change row
            row_height = row_height > 0.5 ? 0. : 1.;
            next_row += row_min
                    + (int)(((long long)row_range) * rand() / RAND_MAX);
        }
        for (int j = 0; j < ncol; j++) {
            *data_ptr = row_height;
            data_ptr++;
        }
    }

    model.hfield_size[2] = terrain_height;
    model.geom_pos[terrain_geom_id * 3 + 2] = - terrain_height / 2;
}

} // anonymous namespace

A1GymEnv::A1GymEnv(A1GymEnv &&other)
{
    std::lock_guard this_guard(config_mutex_);

    this->ref_model_ = other.ref_model_;
    this->model_ = other.model_;
    this->data_ = other.data_;
    this->last_target_jpos_ = other.last_target_jpos_;
    this->step_dt_ = other.step_dt_;
    this->unallowed_contact_body_ = other.unallowed_contact_body_;

    this->camera_ = other.camera_;
    this->render_options_ = other.render_options_;
    this->scene_ = other.scene_;
    this->context_ = other.context_;

    this->change_terrain_required_ = other.change_terrain_required_;
    this->terrain_height_ = other.terrain_height_;
    this->target_vel_ = other.target_vel_;
    this->foot_xy_offset_ = other.foot_xy_offset_;
    this->duty_ratio_ = other.duty_ratio_;
    this->period_ = other.period_;
    this->raise_height_ = other.raise_height_;
    this->torso_height_ = other.torso_height_;
    this->traj_cfg_ = other.traj_cfg_;

    this->config_ = other.config_;
}

A1GymEnv::~A1GymEnv()
{
    mj_deleteData(this->data_);
    mj_deleteModel(this->model_);
}

void A1GymEnv::InitFromMujocoModel(
        const mjModel_ *model_ref, const A1RLParam &param)
{
    this->ref_model_ = model_ref;
    this->model_ = mj_copyModel(nullptr, model_ref);
    this->data_ = mj_makeData(this->model_);
    this->step_dt_ = param.step_dt;
    this->config_ = &param;

    // register body id for contact detection
    unallowed_contact_body_.clear();
    for (int i = 0; i < model_->nbody; i++) {
        const std::string_view root_name(
                    model_->names + model_->name_bodyadr[model_->body_rootid[i]]);
        if (root_name != "trunk") {
            // we only care about bodies on the robot
            continue;
        }

        const std::string_view body_name(
                    model_->names + model_->name_bodyadr[i]);
        if (body_name.find("_calf") == std::string_view::npos
                && body_name.find("_foot") == std::string_view::npos)
            unallowed_contact_body_.insert(i);
    }

    if (model_->nhfield >= 1) {
        terrain_height_ = 0.;
        terrain_id_ = rand() % 3;
        change_terrain_required_ = true;

        for (int i = 0; i < model_->ngeom; i++) {
            const std::string_view geom_name(
                        model_->names + model_->name_geomadr[i]);
            if (geom_name == "uneven_terrain") {
                terrain_geom_id_ = i;
            }
            else if (geom_name == "obstacle_1") {
                obstacle_id_[0] = i;
            }
            else if (geom_name == "obstacle_2") {
                obstacle_id_[1] = i;
            }
        }
    }

    ResetEnv();
}

void A1GymEnv::Step(const double *action,
                    double *obs, bool &done, double &reward, bool &timeout)
{
    const Eigen::Map<const Array12d> safe_jpos_max(joint_max_);
    const Eigen::Map<const Array12d> safe_jpos_min(joint_min_);
    const Eigen::Map<const Array12d> safe_jvel_max(joint_vel_max_);
    average_abs_torq_square_.setZero();
    average_pos_jerk_ = 0;
    average_vel_jerk_ = 0;
    collision_ = false;

    const int sim_iter = step_dt_ / model_->opt.timestep;
    const double delayed_time
            = data_->time + config_->policy_delay
            - model_->opt.timestep * sim_iter * (delayed_actions_.size() - 1);
    const double torso_force_x
            = external_momentum_x_ / (sim_iter * model_->opt.timestep);
    const double torso_force_y
            = external_momentum_y_ / (sim_iter * model_->opt.timestep);
    external_momentum_x_ = 0;
    external_momentum_y_ = 0;

    InputArray last_action[2];
    last_action[0] = delayed_actions_.front();
    delayed_actions_.pop_front();
    delayed_actions_.push_back(Eigen::Map<const InputArray>(action));
    last_action[1] = delayed_actions_.front();

    for (int i = 0; i < sim_iter; i++) {
        if (data_->time < delayed_time)
            ApplyControlForce(last_action[0].data());
        else
            ApplyControlForce(last_action[1].data());
        data_->qfrc_applied[0] = torso_force_x;
        data_->qfrc_applied[1] = torso_force_y;
        mj_step(model_, data_);

        // update pos & vel jerk
        const Eigen::Map<const Array12d> jpos(data_->qpos + 7);
        Eigen::Map<Array12d> jvel(data_->qvel + 6);
        average_pos_jerk_ += (jpos > safe_jpos_max).sum()
                           + (jpos < safe_jpos_min).sum();
        average_vel_jerk_ += (jvel.abs() > safe_jvel_max).sum();
        jvel = jvel.cwiseMax(- safe_jvel_max).cwiseMin(safe_jvel_max);

//        data_->qpos[0] = 0.;
//        data_->qpos[1] = 0.;
//        data_->qpos[2] = 0.5;
//        data_->qpos[3] = 1.;
//        data_->qpos[4] = 0.;
//        data_->qpos[5] = 0.;
//        data_->qpos[6] = 0.;
    };

    // if the robot moves to the boundary, we place it back
    bool replace_robot = false;
    if (data_->qpos[0] > 18.) {
        data_->qpos[0] -= 18.;
        replace_robot = true;
    }
    else if (data_->qpos[0] < -1.) {
        data_->qpos[0] += 18.;
        replace_robot = true;
    }
    if (data_->qpos[1] > 1.) {
        data_->qpos[1] -= 2.;
        replace_robot = true;
    }
    else if (data_->qpos[1] < - 1.) {
        data_->qpos[1] += 2.;
        replace_robot = true;
    }
    if (replace_robot)
        data_->qpos[2] = model_->qpos0[2];

    push_timer_ += step_dt_;
    if (push_timer_ > config_->push_interval) {
        // push robot to change its velocity
        push_timer_ -= config_->push_interval;
        Eigen::Map<Eigen::Vector3d>(data_->qvel)
                += Eigen::Vector3d::Random() * 0.6;
        Eigen::Map<Eigen::Vector3d>(data_->qvel + 3)
                += Eigen::Vector3d::Random() * 2.;
    }

    turn_timer_ += step_dt_;
    if (turn_timer_ > config_->turn_interval) {
        // push robot to change its velocity
        turn_timer_ -= config_->turn_interval;
        target_yaw_ = rand_float() * 0.3;
    }

    // statistics
    quat_ = Eigen::Quaterniond(data_->qpos[3], data_->qpos[4],
            data_->qpos[5], data_->qpos[6]);
    rpy_ = Quaternion2EulerZYX(quat_);
    // use a low-pass filter for vx
    const Eigen::Rotation2D rot(target_yaw_);
    const Eigen::Map<Eigen::Vector2d> vel_xy(data_->qvel);
    mean_x_vel_ = mean_x_vel_ * 0.95 + (rot.inverse() * vel_xy).x() * 0.05;

    average_abs_torq_square_ /= sim_iter;
    average_pos_jerk_ /= sim_iter;
    average_vel_jerk_ /= sim_iter;

    // compute reward & check if terminate
    ComputeReward(reward, done, timeout);

    if (done) {
        last_mean_x_vel_ = mean_x_vel_;
        last_elapsed_time_ = data_->time;
        last_terrain_height_ = terrain_height_;
        mean_x_vel_ = 0.;
        ResetEnv();
    }

    // compute observation
    ComputeObservations(obs);
}

void A1GymEnv::Render()
{
    std::lock_guard g(render_mutex_);
    if (context_ != nullptr) {
        if (make_context_required_) {
            mjv_makeScene(model_, scene_, 2000);
            mjr_makeContext(model_, context_, mjFONTSCALE_150);
            make_context_required_ = false;
        }
        mjv_updateScene(model_, data_, render_options_, NULL,
                        camera_, mjCAT_ALL, scene_);
    }
}

void A1GymEnv::ApplyControlForce(const double *action)
{
    // get control target

    if (traj_cfg_.vel.x() < 0.5) {
        // keep t = 0 for 1 s to generate a standing trajectory on start
        const double t = std::max(data_->time - 1., 0.);
        if (t < traj_cfg_.period * 1.5) {
            if (t < traj_cfg_.period)
                cur_vel_cmd_.setZero();
            StartJPosFromBuffer(t, target_jpos_, feet_td_);
        }
        else {
            cur_vel_cmd_ = traj_cfg_.vel;
            TargetJPosFromBuffer(t, target_jpos_, feet_td_);
        }
    }
    else {
        cur_vel_cmd_ = traj_cfg_.vel;
        TargetJPosFromBuffer(data_->time, target_jpos_, feet_td_);
    }

    target_jvel_ = (target_jpos_ - last_target_jpos_)
            / model_->opt.timestep;

    // get feedback & error
    const Eigen::Map<const Array12d> jpos_fb(data_->qpos + 7);
    const Eigen::Map<const Array12d> jvel_fb(data_->qvel + 6);

    jpos_err_ = target_jpos_ - jpos_fb;
    jvel_err_ = target_jvel_ - jvel_fb;

    // prepare torque output
//    Array12d last_torq = Eigen::Map<Array12d>(data_->ctrl);
    Eigen::Map<Array12d> torq_output(data_->ctrl);

    // get kp, kd, torq_ff, compute torq_output
    // torq = kp * (jpos_target - jpos_fb) + kd * (jvel_target - jvel_fb)
    //      + torq_ff
    for (int i = 0; i < 4; i++) {
        const int idx = i * 3;
        // simulate noise (disabled)
        Eigen::Array3d scale_kp = 1. + Eigen::Array3d::Random() * 0.;
        Eigen::Array3d scale_kd = 1. + Eigen::Array3d::Random() * 0.;
        Eigen::Array3d scale_tau = 1. + Eigen::Array3d::Random() * 0.;
        if (feet_td_[i]) {
            Arr3CRef kp(action + idx);
            Arr3CRef kd(action + idx + 12);
            Arr3CRef torq_ff(action + idx + 24);
            torq_output.segment<3>(idx)
                    = kp * jpos_err_.segment<3>(idx) * scale_kp
                    + kd * jvel_err_.segment<3>(idx) * scale_kd
                    + torq_ff * scale_tau;
        }
        else {
            Arr3CRef kp(action + idx + 36);
            Arr3CRef kd(action + idx + 36 + 12);
            Arr3CRef torq_ff(action + idx + 36 + 24);
            torq_output.segment<3>(idx)
                    = kp * jpos_err_.segment<3>(idx) * scale_kp
                    + kd * jvel_err_.segment<3>(idx) * scale_kd
                    + torq_ff * scale_tau;
        }
    }

    last_target_jpos_ = target_jpos_;

//    for (int i = 0; i < 12; i++)
//        torq_output[i] = std::clamp(
//                    torq_output[i], last_torq[i] - 1., last_torq[i] + 1.);

    average_abs_torq_square_ += torq_output.square();
}

void A1GymEnv::ComputeReward(double &reward, bool &done, bool &timeout)
{
    // velocity tracking reward
    Eigen::Map<const Eigen::Vector3d> base_vel(data_->qvel);
    const Eigen::Vector2d lin_vel_diff
            = (quat_.inverse() * base_vel).head<2>() - cur_vel_cmd_;
    const double lin_vel_err
            = square(lin_vel_diff.x()) + square(lin_vel_diff.y());
    const double ang_vel_err = square(data_->qvel[5]); // rot_vel_z error
    const double rew_vel_tracking
            = config_->rew_vel_xy * std::exp(- lin_vel_err / 0.02)
            + config_->rew_rotvel * std::exp(- ang_vel_err / 0.01);

    // torque penalty
    const double rew_torq
            = config_->rew_torque * average_abs_torq_square_.sum();

    // joint tracking reward
    const double jpos_err_sum = jpos_err_.square().sum();
    const double rew_joint_tracking
            = config_->rew_tracking * std::exp(- jpos_err_sum / 0.5);

    // orientation reward
    const double rew_front
            = config_->rew_front * std::exp(
                - square(rpy_[2] - target_yaw_) / 0.01);
    const double up_axis_error = rpy_.head<2>().squaredNorm();
    const double rew_up
            = config_->rew_up * std::exp(- up_axis_error / 0.02);

    // height reward
    const double height_error = square(data_->qpos[2] - torso_height_);
    const double rew_height
            = config_->rew_height * std::exp(- height_error / 0.05);

    // jpos & jvel jerk penalty
    const double rew_jerk
            = config_->rew_jerk * average_pos_jerk_
            + config_->rew_vjerk * average_vel_jerk_;

    reward = rew_vel_tracking + rew_torq + rew_joint_tracking
            + rew_front + rew_up + rew_height + rew_jerk;
    reward = std::max(reward, 0.);
    reward *= step_dt_; // normalize reward

    timeout = data_->time > config_->eposide_len;
    done = HasCollision() || timeout;
}

void A1GymEnv::ComputeObservations(double *obs)
{
    Eigen::Map<Eigen::Matrix<double, kOBS_SIZE, 1>> obs_vec(obs);
    Eigen::Map<const Eigen::Vector3d> base_vel(data_->qvel);
    Eigen::Map<const Eigen::Vector3d> base_rot_vel(data_->qvel + 3);

    // add noises
    obs_vec.segment<3>(0) = quat_.inverse() * (
                base_vel - Eigen::Vector3d::Random() * 0.1) * 0.4;
    obs_vec.segment<3>(3) = quat_.inverse() * (
                base_rot_vel - Eigen::Vector3d::Random() * 0.2) * 0.25;
    obs_vec.segment<3>(6) = - (quat_.inverse() * env_ori_
                               * Eigen::Vector3d::UnitZ())
            + Eigen::Vector3d::Random() * 0.05;
    obs_vec[9] = std::cos(rpy_[2] - target_yaw_);
    obs_vec[10] = std::sin(rpy_[2] - target_yaw_);
    obs_vec.segment<2>(11) = cur_vel_cmd_ * 0.4;
    obs_vec[13] = 0.; // no rot vel
    obs_vec[14] = (traj_cfg_.period - config_->period_mean)
            * config_->period_scale_inv;
    obs_vec[15] = (traj_cfg_.torso_height - config_->height_mean)
            * config_->height_scale_inv;
    obs_vec[16] = (traj_cfg_.duty_ratio - config_->duty_ratio_mean)
            * config_->duty_ratio_scale_inv;
    obs_vec.segment<12>(17         ) = jpos_err_;
    obs_vec.segment<12>(17 +     12) = jvel_err_ * 0.05;
    obs_vec.segment<12>(17 + 2 * 12) = target_jpos_;
    obs_vec.segment<12>(17 + 3 * 12) = target_jvel_ * 0.05;
    obs_vec[65] = feet_td_[0];
    obs_vec[66] = feet_td_[1];
}

void A1GymEnv::ResetEnv()
{
    Curriculum();

    if (traj_change_cnt_ == 0) {
        // this indicates trajectory param has changed
        BuildTrajConfig();
        GenerateTrajBuffer();
    }

    if (change_terrain_required_) {
        ChangeTerrain();
        change_terrain_required_ = false;
        // update gui after terrain model changed
        make_context_required_ = true;
    }

    DynamicsRandomization();

    mj_resetData(model_, data_);

    InitStateRandomization();

    bool feet_td[4]; // not used
//    ComputeTargetJointPos(0., last_target_jpos_, feet_td);
    TargetJPosFromBuffer(0., last_target_jpos_, feet_td);
}

void A1GymEnv::Curriculum()
{
    const bool upgrade = last_elapsed_time_ > config_->eposide_len * 0.9
            && (target_vel_.x() - last_mean_x_vel_) < 0.2;
    const bool degrade = last_elapsed_time_ < config_->eposide_len * 0.5
            || (target_vel_.x() - last_mean_x_vel_) > 0.5;

    // check if it can upgrade or degrade
    const bool can_upgrade = target_vel_.x() < config_->vel_x_max
            || terrain_height_ < config_->terrain_height;
    const bool can_degrade = target_vel_.x() > config_->vel_x_min
            || terrain_height_ > 0;
    // generate new cmd_vel & terrain height
    if (config_->do_curriculum) {
        if (upgrade && can_upgrade) {
            target_vel_.x() += 0.3 + 0.1 * rand_float();
            if (terrain_height_ < config_->terrain_height) {
                terrain_height_ = std::min(
                            terrain_height_ + 0.01 + 0.005 * rand_float(),
                            config_->terrain_height);
                terrain_id_ = rand() % 3;
                change_terrain_required_ = true;
            }
            traj_change_cnt_ = 0;
        }
        else if (degrade && can_degrade) {
            target_vel_.x() -= 0.3 + 0.1 * rand_float();
            if (terrain_height_ > 0) {
                terrain_height_ = std::max(
                            terrain_height_ - 0.01 - 0.005 * rand_float(),
                            0.);
                terrain_id_ = rand() % 3;
                change_terrain_required_ = true;
            }
            traj_change_cnt_ = 0;
        }
        else {
            if (traj_change_cnt_ >= 5) {
                target_vel_.x() += 0.1 * rand_float();
                traj_change_cnt_ = 0;
                terrain_id_ = rand() % 3;
                change_terrain_required_ = true;
            }
            else
                traj_change_cnt_++;
        }
    }
    else {
        target_vel_.x() = rand_float()
                * (config_->vel_x_max - config_->vel_x_min)
                + config_->vel_x_min;
        traj_change_cnt_++;
    }

    // other commands
    if (traj_change_cnt_ == 0) {
        target_vel_.x() = std::clamp(target_vel_.x(),
                                     config_->vel_x_min, config_->vel_x_max);
        target_vel_.y() = rand_float() * config_->vel_y_max;
        duty_ratio_ = config_->duty_ratio_mean
                + rand_float() * config_->duty_ratio_scale;
        period_ = config_->period_mean
                + rand_float() * config_->period_scale;
        torso_height_ = config_->height_mean
                + rand_float() * config_->height_scale;

        // clamp period to avoid unrealistic foot pos
        const double foot_range
                = period_ * (duty_ratio_ * 0.8 + 0.2) * 0.5
                * std::abs(target_vel_.x());
        const double max_foot_range
                = std::sqrt(square(0.35) - square(torso_height_));
        if (foot_range > max_foot_range * 0.9) {
            // reduce period
            period_ = period_ * max_foot_range / foot_range;
        }
    }

    target_yaw_ = 0.;
}

void A1GymEnv::BuildTrajConfig()
{
    // basic scalars
    traj_cfg_.vel = target_vel_;
    traj_cfg_.torso_height = torso_height_;
    traj_cfg_.period = period_;
    traj_cfg_.duty_ratio = duty_ratio_;
    traj_cfg_.raise_time_ratio = (1. - duty_ratio_) * 0.2;
    traj_cfg_.half_stance_time = period_ * duty_ratio_ * 0.5;

    if (rand() % 2 == 0) {
        traj_cfg_.phase_offset[0] = 0.;
        traj_cfg_.phase_offset[1] = 0.5;
        traj_cfg_.phase_offset[2] = 0.5;
        traj_cfg_.phase_offset[3] = 0.;
    }
    else {
        traj_cfg_.phase_offset[0] = 0.5;
        traj_cfg_.phase_offset[1] = 0.;
        traj_cfg_.phase_offset[2] = 0.;
        traj_cfg_.phase_offset[3] = 0.5;
    }

    // cartesian phase config
    const double cartesian_raise_time = period_ * traj_cfg_.raise_time_ratio;
    const double raise_vel = raise_height_ * 2 / 3 / cartesian_raise_time;
    traj_cfg_.raise_acc = raise_vel / cartesian_raise_time;

    for (int i = 0; i < 4; i++) {
        const bool right_side_leg = (i % 2 == 0);
        traj_cfg_.stance_pos_offset[i].x() = foot_xy_offset_.x();
        traj_cfg_.stance_pos_offset[i].y()
                = foot_xy_offset_.y() * (right_side_leg ? -1 : 1);
        traj_cfg_.stance_pos_offset[i].z() = 0.;
    }

    // joint traj phase begin & end config
    traj_cfg_.half_cartesian_time
            = duty_ratio_ * 0.5 * period_ + cartesian_raise_time;
    Eigen::Vector3d pos_beg; // joint traj begin pos
    pos_beg.head<2>() = - target_vel_ * traj_cfg_.half_cartesian_time;
    pos_beg.z() = - torso_height_ + raise_height_ / 3;
    Eigen::Vector3d vel_beg;
    vel_beg.head<2>() = - target_vel_;
    vel_beg.z() = raise_height_ * 2 / 3 / cartesian_raise_time;
    Eigen::Vector3d pos_end;
    pos_end.head<2>() = - pos_beg.head<2>();
    pos_end.z() = pos_beg.z();
    Eigen::Vector3d vel_end;
    vel_end.head<2>() = vel_beg.head<2>();
    vel_end.z() = - vel_beg.z();

    for (int i = 0; i < 4; i++) {
        Eigen::Vector3d pos_for_leg;
        Eigen::Vector3d vel_for_leg;
        Eigen::Matrix3d J;
        // leg order: FR, FL, RR, RL
        const bool right_side_leg = (i % 2 == 0);

        pos_for_leg = pos_beg + traj_cfg_.stance_pos_offset[i];
        vel_for_leg = vel_beg;
        if (right_side_leg) { // shift to left legs for InvKin & Jacobian
            pos_for_leg.y() = - pos_for_leg.y();
            vel_for_leg.y() = - vel_for_leg.y();
        }
        traj_cfg_.jpos_beg[i] = A1InvKin(pos_for_leg);
        J = A1Jacobian(traj_cfg_.jpos_beg[i]);
        traj_cfg_.jvel_beg[i] = J.inverse() * vel_for_leg;

        pos_for_leg = pos_end + traj_cfg_.stance_pos_offset[i];
        vel_for_leg = vel_end;
        if (right_side_leg) {
            pos_for_leg.y() = - pos_for_leg.y();
            vel_for_leg.y() = - vel_for_leg.y();
        }
        traj_cfg_.jpos_end[i] = A1InvKin(pos_for_leg);
        J = A1Jacobian(traj_cfg_.jpos_end[i]);
        traj_cfg_.jvel_end[i] = J.inverse() * vel_for_leg;

        if (traj_cfg_.jvel_end[i].hasNaN()) {
            std::lock_guard g(err_mutex);
            std::cout << "J det: " << J.determinant() << std::endl;
            std::cout << "jpos: " << traj_cfg_.jpos_end[i].transpose()
                      << std::endl;
            std::cout << "pos: " << pos_for_leg.transpose() << std::endl;
        }

        if (right_side_leg) { // the hip joint should be flipped
            traj_cfg_.jpos_beg[i][0] = - traj_cfg_.jpos_beg[i][0];
            traj_cfg_.jpos_end[i][0] = - traj_cfg_.jpos_end[i][0];
            traj_cfg_.jvel_beg[i][0] = - traj_cfg_.jvel_beg[i][0];
            traj_cfg_.jvel_end[i][0] = - traj_cfg_.jvel_end[i][0];
        }
    }

    // joint traj phase config
    const double mu = 0.3;
    const double t_joint = period_ - 2 * traj_cfg_.half_cartesian_time;
    const double t_acc = t_joint * mu;
    traj_cfg_.half_joint_time = t_joint / 2;
    traj_cfg_.t_acc = t_acc;
    const double half_no_acc_time
            = traj_cfg_.half_joint_time - traj_cfg_.t_acc;

    for (int i = 0; i < 4; i++) {
        traj_cfg_.jvel_mid[i]
                = ((traj_cfg_.jpos_end[i] - traj_cfg_.jpos_beg[i]) / t_joint
                   - (traj_cfg_.jvel_beg[i] + traj_cfg_.jvel_end[i])
                        * 0.5 * mu) / (1. - mu);
        traj_cfg_.jacc_beg[i]
                = (traj_cfg_.jvel_mid[i] - traj_cfg_.jvel_beg[i]) / t_acc;
        traj_cfg_.jacc_end[i]
                = (traj_cfg_.jvel_end[i] - traj_cfg_.jvel_mid[i]) / t_acc;
        traj_cfg_.jpos_mid[i]
                = traj_cfg_.jpos_beg[i] + t_acc * traj_cfg_.jvel_beg[i]
                + 0.5 * square(t_acc) * traj_cfg_.jacc_beg[i]
                + half_no_acc_time * traj_cfg_.jvel_mid[i];

        const double j_knee_top
                = 2 * std::asin((torso_height_ - raise_height_)
                                / thigh_len_ / 2) - M_PI;
        traj_cfg_.knee_inc[i] = j_knee_top - traj_cfg_.jpos_mid[i][2];

        bool err = false;
        if (traj_cfg_.jvel_mid[i].hasNaN()) {
            std::lock_guard g(err_mutex);
            std::cout << "jvel_mid err " << std::endl;
            err = true;
        }
        if (traj_cfg_.jpos_mid[i].hasNaN()) {
            std::lock_guard g(err_mutex);
            std::cout << "jpos_mid err " << std::endl;
            err = true;
        }
        if (traj_cfg_.jvel_beg[i].hasNaN()) {
            std::lock_guard g(err_mutex);
            std::cout << "jvel_beg err " << std::endl;
            err = true;
        }
        if (traj_cfg_.jvel_end[i].hasNaN()) {
            std::lock_guard g(err_mutex);
            std::cout << "jvel_end err " << std::endl;
            err = true;
        }
        if (traj_cfg_.jacc_beg[i].hasNaN()) {
            std::lock_guard g(err_mutex);
            std::cout << "jacc_beg err " << std::endl;
            err = true;
        }
        if (traj_cfg_.jpos_beg[i].hasNaN()) {
            std::lock_guard g(err_mutex);
            std::cout << "jpos_beg err " << std::endl;
            err = true;
        }
        if (traj_cfg_.jpos_end[i].hasNaN()) {
            std::lock_guard g(err_mutex);
            std::cout << "jpos_end err " << std::endl;
            err = true;
        }

        if (err)
            exit(1);
    }

    // initial step configuration, when there's no forward velocity
    // every leg has its initial step configuration
    for (int i = 0; i < 4; i++) {
        const bool right_side_leg = (i % 2 == 0);
        Eigen::Vector3d raise_pos_end
                = traj_cfg_.stance_pos_offset[i];
        raise_pos_end.z() = - torso_height_ + raise_height_ / 3;
        Eigen::Vector3d raise_vel_end;
        raise_vel_end.head<2>().setZero();
        raise_vel_end.z()
                = raise_height_ * 2 / 3 / cartesian_raise_time;

        if (right_side_leg) {
            raise_pos_end.y() = - raise_pos_end.y();
            raise_vel_end.y() = - raise_vel_end.y();
        }
        traj_cfg_.start_lift_jpos[i] = A1InvKin(raise_pos_end);
        traj_cfg_.start_lift_jvel[i]
                = A1Jacobian(traj_cfg_.start_lift_jpos[i]).inverse()
                * raise_vel_end;
        if (right_side_leg) {
            traj_cfg_.start_lift_jpos[i][0]
                    = - traj_cfg_.start_lift_jpos[i][0];
            traj_cfg_.start_lift_jvel[i][0]
                    = - traj_cfg_.start_lift_jvel[i][0];
        }

        // do interpolation
        Eigen::Matrix2d par_matrix;
        const double t = traj_cfg_.half_joint_time;
        const double t_2 = square(t);
        const double t_3 = t_2 * t;
        par_matrix << t_2, t_3, 2 * t, 3 * t_2;
        Eigen::Matrix<double, 2, 3> y;
        for (int j = 0; j < 3; j++) {
            y(0, j) = traj_cfg_.jpos_mid[i][j]
                    - traj_cfg_.start_lift_jpos[i][j]
                    - traj_cfg_.start_lift_jvel[i][j] * t;
            y(1, j) = traj_cfg_.jvel_mid[i][j]
                    - traj_cfg_.start_lift_jvel[i][j];
            if (j == 2)
                y(0, j) += traj_cfg_.knee_inc[i];
        }
        traj_cfg_.start_swing_par[i] = par_matrix.inverse() * y;
    }
}

void A1GymEnv::GenerateTrajBuffer()
{
    const int size = 200;
    traj_buffer_.resize(size);
    td_buffer_.resize(size);
    const double dt = traj_cfg_.period / size;

    for (int i = 0; i < size; i++) {
        const double time = i * dt;
        ComputeTargetJointPos(time, traj_buffer_[i], td_buffer_[i].data());
    }

    const int start_size = size * 3 / 2;
    start_traj_buffer_.resize(start_size);
    start_td_buffer_.resize(start_size);

    for (int i = 0; i < start_size; i++) {
        const double time = i * dt;
        TargetJPosOnStart2(
                    time, start_traj_buffer_[i], start_td_buffer_[i].data());
    }
}

void A1GymEnv::DynamicsRandomization()
{
    if (!config_->do_dyn_rand) {
        // just set friction
        for (int i = 0; i < model_->ngeom; i++) {
            model_->geom_friction[i * 3] = 0.5;
        }
        env_ori_.setIdentity();
        return;
    }

    for (int i = 0; i < model_->nbody; i++) {
        model_->body_mass[i]
                = ref_model_->body_mass[i] * random_scale(0.3);
        model_->body_inertia[i * 3    ]
                = ref_model_->body_inertia[i * 3    ] * random_scale(0.1);
        model_->body_inertia[i * 3 + 1]
                = ref_model_->body_inertia[i * 3 + 1] * random_scale(0.1);
        model_->body_inertia[i * 3 + 2]
                = ref_model_->body_inertia[i * 3 + 2] * random_scale(0.1);
        model_->body_ipos[i * 3    ]
                = ref_model_->body_ipos[i * 3    ] * random_scale(0.1);
        model_->body_ipos[i * 3 + 1]
                = ref_model_->body_ipos[i * 3 + 1] * random_scale(0.1);
        model_->body_ipos[i * 3 + 2]
                = ref_model_->body_ipos[i * 3 + 2] * random_scale(0.1);
    }

    for (int i = 0; i < model_->ngeom; i++) {
        model_->geom_friction[i * 3]
                = 0.5 * random_scale(0.4);
    }

    env_ori_ = Eigen::AngleAxisd(rand_float() * 0.1, Eigen::Vector3d::UnitX())
            * Eigen::AngleAxisd(rand_float() * 0.2, Eigen::Vector3d::UnitY());

    Eigen::Map<Eigen::Vector3d>(model_->opt.gravity)
            = env_ori_ * Eigen::Map<const Eigen::Vector3d>(
                ref_model_->opt.gravity);

    mj_setConst(model_, data_);
}

void A1GymEnv::InitStateRandomization()
{
    const Eigen::Array3d rpy = (Eigen::Array3d::Random() * 2 - 1.) * 0.1;
    Eigen::Quaterniond init_quat;
    init_quat = Eigen::AngleAxisd(rpy[2], Eigen::Vector3d::UnitZ())
            * Eigen::AngleAxisd(rpy[1], Eigen::Vector3d::UnitY())
            * Eigen::AngleAxisd(rpy[0], Eigen::Vector3d::UnitX());
    data_->qpos[3] = init_quat.w();
    data_->qpos[4] = init_quat.x();
    data_->qpos[5] = init_quat.y();
    data_->qpos[6] = init_quat.z();

    Eigen::Map<Eigen::Array3d> vel(data_->qvel);
    vel.x() = target_vel_.x() + rand_float() * 0.2; // hot start
    vel.y() = rand_float() * 0.1;

    Eigen::Map<Array12d> jpos(data_->qpos + 7);
    Eigen::Map<Array12d> jvel(data_->qvel + 6);
    jpos += (Array12d::Random() * 2 - 1.) * 0.1;
    jvel =  (Array12d::Random() * 2 - 1.) * 1.;

    // randomize last action
    delayed_actions_.clear();
    const int n_substep = step_dt_ / model_->opt.timestep;
    const int action_delay_cnt
            = config_->policy_delay / (n_substep * model_->opt.timestep) + 1;
    for (int i = 0; i < action_delay_cnt; i++) {
        InputArray last_act = InputArray::Zero();
        last_act.segment<12>(24) = Array12d::Random() * 15.;
        last_act.segment<12>(60) = Array12d::Random() * 15.;
        delayed_actions_.push_back(last_act);
    }
}

void A1GymEnv::ChangeTerrain()
{
    if (model_->nhfield != 1) {
        std::cout << "Found " << model_->nhfield << " height fields."
                  << std::endl;
        return;
    }

    switch (terrain_id_) {
    case 0:
        UnevenTerrain(*model_, terrain_geom_id_, terrain_height_);
        break;
    case 1:
        Stairs(*model_, terrain_geom_id_, terrain_height_);
        break;
    case 2:
        SideStairs(*model_, terrain_geom_id_, terrain_height_ * 0.6);
        break;
    }
}

void A1GymEnv::ComputeTargetJointPos(
        double time, Array12d &jpos, bool *feet_td) const
{
    const double phase = std::fmod(time / traj_cfg_.period, 1.);

    for (int i = 0; i < 4; i++) {
        const bool right_side_leg = (i % 2 == 0);
        // hard-coding trotting gait
        // assuming phase in [0., 1.)
        double leg_phase = phase + traj_cfg_.phase_offset[i];
        if (leg_phase >= 1.)
            leg_phase -= 1.;
        const double leg_moving_time = leg_phase * traj_cfg_.period;
        const double cartesian_time = (leg_phase < 0.5) ?
                    leg_moving_time : leg_moving_time - traj_cfg_.period;

        if (abs(cartesian_time) < traj_cfg_.half_cartesian_time) {
            // cartesian moving time
            Eigen::Vector3d foot_pos;
            foot_pos.head<2>() = - cartesian_time * traj_cfg_.vel;

            foot_pos.z() = - traj_cfg_.torso_height;
            const double raise_time
                    = abs(cartesian_time) - traj_cfg_.half_stance_time;
            if (raise_time > 0) {
                // foot raising, add raise height
                foot_pos.z()
                        += square(raise_time) * traj_cfg_.raise_acc * 0.5;
                feet_td[i] = false;
            }
            else
                // foot standing
                feet_td[i] = true;

            foot_pos += traj_cfg_.stance_pos_offset[i];

            // compute jpos from foot cartesian pos
            if (right_side_leg)
                foot_pos.y() = - foot_pos.y();
            jpos.segment<3>(i * 3) = A1InvKin(foot_pos);
            if (right_side_leg)
                jpos[i * 3] = - jpos[i * 3];
        }
        else {
            // leg swinging, use joint trajectory target
            feet_td[i] = false;
            const double shifted_dur
                    = leg_moving_time - traj_cfg_.period * 0.5;
            const double acc_time
                    = traj_cfg_.half_joint_time - abs(shifted_dur);
            if (acc_time < traj_cfg_.t_acc) {
                const double acc_time_2 = 0.5 * square(acc_time);
                if (shifted_dur < 0.) {
                    // accelerating
                    jpos.segment<3>(i * 3)
                            = traj_cfg_.jpos_beg[i]
                            + acc_time * traj_cfg_.jvel_beg[i]
                            + acc_time_2 * traj_cfg_.jacc_beg[i];
                }
                else {
                    // decelerating
                    jpos.segment<3>(i * 3)
                            = traj_cfg_.jpos_end[i]
                            - acc_time * traj_cfg_.jvel_end[i]
                            + acc_time_2 * traj_cfg_.jacc_end[i];
                }
            }
            else {
                jpos.segment<3>(i * 3) = traj_cfg_.jpos_mid[i]
                        + shifted_dur * traj_cfg_.jvel_mid[i];
            }

            // add knee pos difference
            jpos[i * 3 + 2] += traj_cfg_.knee_inc[i]
                    * (std::cos(shifted_dur / traj_cfg_.half_joint_time
                                * M_PI) + 1.) * 0.5;
        }
    }
}

void A1GymEnv::TargetJPosOnStart(
        double time, Array12d &jpos, bool *feet_td) const
{
    int standing_idx[2];
    int moving_idx[2];

    if (traj_cfg_.phase_offset[0] > traj_cfg_.phase_offset[1]) {
        standing_idx[0] = 0;
        standing_idx[1] = 3;
        moving_idx[0] = 1;
        moving_idx[1] = 2;
    }
    else {
        standing_idx[0] = 1;
        standing_idx[1] = 2;
        moving_idx[0] = 0;
        moving_idx[1] = 3;
    }

    // at beginning, all feet are on the ground
    // at the end of the first period, each foot should be in position
    // of periodic motion
    feet_td[standing_idx[0]] = true;
    feet_td[standing_idx[1]] = true;
    const std::size_t sz = traj_buffer_.size();
    jpos.segment<3>(standing_idx[0] * 3)
            = traj_buffer_[sz / 2 - 1].segment<3>(standing_idx[0] * 3);
    jpos.segment<3>(standing_idx[1] * 3)
            = traj_buffer_[sz / 2 - 1].segment<3>(standing_idx[1] * 3);

    if (time < traj_cfg_.half_stance_time) {
        // standing
        jpos.segment<3>(moving_idx[0] * 3)
                = traj_buffer_[0].segment<3>(moving_idx[0] * 3);
        jpos.segment<3>(moving_idx[1] * 3)
                = traj_buffer_[0].segment<3>(moving_idx[1] * 3);
        feet_td[moving_idx[0]] = true;
        feet_td[moving_idx[1]] = true;
    }
    else if (time < traj_cfg_.half_cartesian_time) {
        // cartesian raising
        feet_td[moving_idx[0]] = false;
        feet_td[moving_idx[1]] = false;
        const double raise_time = time - traj_cfg_.half_stance_time;
        for (int i = 0; i < 2; i++) {
            const bool right_side_leg = (moving_idx[i] % 2 == 0);
            Eigen::Vector3d foot_pos = traj_cfg_.stance_pos_offset[moving_idx[i]];
            foot_pos.z() = - traj_cfg_.torso_height
                    + square(raise_time) * traj_cfg_.raise_acc * 0.5;

            if (right_side_leg)
                foot_pos.y() = - foot_pos.y();
            jpos.segment<3>(moving_idx[i] * 3) = A1InvKin(foot_pos);
            if (right_side_leg)
                jpos[moving_idx[i] * 3] = - jpos[moving_idx[i] * 3];
        }
    }
    else {
        // joint traj moving
        feet_td[moving_idx[0]] = false;
        feet_td[moving_idx[1]] = false;
        for (int i = 0; i < 2; i++) {
            const auto &par = traj_cfg_.start_swing_par[moving_idx[i]];
            const double dt = time - traj_cfg_.half_cartesian_time;
            const double dt_2 = square(dt);
            const double dt_3 = dt * dt_2;
            jpos.segment<3>(moving_idx[i] * 3)
                    = traj_cfg_.start_lift_jpos[moving_idx[i]]
                    + traj_cfg_.start_lift_jvel[moving_idx[i]] * dt
                    + par.topRows<1>().transpose() * dt_2
                    + par.bottomRows<1>().transpose() * dt_3;
        }
    }

}

void A1GymEnv::TargetJPosOnStart2(
        double time, Array12d &jpos, bool *feet_td) const
{
    for (int i = 0; i < 4; i++) {
        const bool right_side_leg = (i % 2 == 0);
        const double foot_start_time = (1. - traj_cfg_.phase_offset[i])
                * traj_cfg_.period;
        const double relative_time = time - foot_start_time;
        if (foot_start_time < traj_cfg_.half_stance_time) {
            // the foot should be on ground at the end of first period
            // so keep it stationary on the entire first period
            if (time < traj_cfg_.period) {
                feet_td[i] = td_buffer_[0][i];
                jpos.segment<3>(i * 3)
                        = traj_buffer_[0].segment<3>(i * 3);
            }
            else {
                const int sz = (signed)traj_buffer_.size();
                const double dt = traj_cfg_.period / sz;
                const int idx = (int)(time / dt);
                const double alpha = time / dt - idx;
                const int next_idx = (idx + 1) % sz;
                feet_td[i] = td_buffer_[idx % sz][i];
                jpos.segment<3>(i * 3)
                        = traj_buffer_[idx % sz].segment<3>(i * 3) * (1. - alpha)
                        + traj_buffer_[next_idx].segment<3>(i * 3) * alpha;
            }
        }
        else if (relative_time < traj_cfg_.half_stance_time) {
            // standing
            const int sz = (signed)traj_buffer_.size();
            const double dt = traj_cfg_.period / sz;
            const int idx = (int)(foot_start_time / dt);
            const double alpha = foot_start_time / dt - idx;
            const int next_idx = (idx + 1) % sz;
            jpos.segment<3>(i * 3)
                    = traj_buffer_[idx % sz].segment<3>(i * 3) * (1. - alpha)
                    + traj_buffer_[next_idx].segment<3>(i * 3) * alpha;
            feet_td[i] = true;
        }
        else if (relative_time < traj_cfg_.half_cartesian_time) {
            // cartesian raising
            feet_td[i] = false;
            const double raise_time
                    = relative_time - traj_cfg_.half_stance_time;
            Eigen::Vector3d foot_pos = traj_cfg_.stance_pos_offset[i];
            foot_pos.z() = - traj_cfg_.torso_height
                    + square(raise_time) * traj_cfg_.raise_acc * 0.5;

            if (right_side_leg)
                foot_pos.y() = - foot_pos.y();
            jpos.segment<3>(i * 3) = A1InvKin(foot_pos);
            if (right_side_leg)
                jpos[i * 3] = - jpos[i * 3];
        }
        else if (relative_time < traj_cfg_.period / 2) {
            // joint traj moving with no xy plane velocity
            feet_td[i] = false;
            const auto &par = traj_cfg_.start_swing_par[i];
            const double dt = relative_time - traj_cfg_.half_cartesian_time;
            const double dt_2 = square(dt);
            const double dt_3 = dt * dt_2;
            jpos.segment<3>(i * 3)
                    = traj_cfg_.start_lift_jpos[i]
                    + traj_cfg_.start_lift_jvel[i] * dt
                    + par.topRows<1>().transpose() * dt_2
                    + par.bottomRows<1>().transpose() * dt_3;
        }
        else {
            // periodic motion
            const int sz = (signed)traj_buffer_.size();
            const double dt = traj_cfg_.period / sz;
            const int idx = (int)(time / dt);
            const double alpha = time / dt - idx;
            const int next_idx = (idx + 1) % sz;
            feet_td[i] = td_buffer_[idx % sz][i];
            jpos.segment<3>(i * 3)
                    = traj_buffer_[idx % sz].segment<3>(i * 3) * (1. - alpha)
                    + traj_buffer_[next_idx].segment<3>(i * 3) * alpha;
        }
    }
}

void A1GymEnv::TargetJPosFromBuffer(
        double time, Array12d &jpos, bool *feet_td) const
{
    const int sz = (signed)traj_buffer_.size();
    const double dt = traj_cfg_.period / sz;
    const int idx = (int)(time / dt);
    const double alpha = time / dt - idx;
    const int next_idx = (idx + 1) % sz;

    jpos = traj_buffer_[idx % sz] * (1. - alpha)
            + traj_buffer_[next_idx] * alpha;
    const auto &td_buffer = td_buffer_[idx % sz];
    for (int i = 0; i < 4; i++)
        feet_td[i] = td_buffer[i];
}

void A1GymEnv::StartJPosFromBuffer(
        double time, Array12d &jpos, bool *feet_td) const
{
    const int sz = (signed)start_traj_buffer_.size();
    const double dt = traj_cfg_.period * 3 / (sz * 2);
    const int idx = std::min(sz - 1, (int)(time / dt));
    const double alpha = time / dt - idx;
    const int next_idx = std::min(sz - 1, idx + 1);;

    jpos = start_traj_buffer_[idx] * (1. - alpha)
            + start_traj_buffer_[next_idx] * alpha;
    const auto &td_buffer = start_td_buffer_[idx];
    for (int i = 0; i < 4; i++)
        feet_td[i] = td_buffer[i];
}

bool A1GymEnv::HasCollision() const
{
    // check contacts
    for (int i = 0; i < data_->ncon; i++) {
        const int geom1 = data_->contact[i].geom1;
        const int geom2 = data_->contact[i].geom2;

        if (unallowed_contact_body_.find(model_->geom_bodyid[geom1])
                != unallowed_contact_body_.end()
         || unallowed_contact_body_.find(model_->geom_bodyid[geom2])
                != unallowed_contact_body_.end())
            return true;
    }

    return false;
}

} // namespace mujoco_parallel_sim
