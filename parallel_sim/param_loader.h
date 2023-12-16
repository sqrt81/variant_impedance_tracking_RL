#ifndef PARAM_LOADER_H_
#define PARAM_LOADER_H_

#include <map>

#define ATTR_API __attribute__ ((visibility ("default")))

namespace mujoco_parallel_sim {

struct A1RLParam
{
    // config
    double step_dt;

    double vel_x_min;
    double vel_x_max;
    double vel_y_min;
    double vel_y_max;
    double acc;
    double height_min;
    double height_max;
    double period_min;
    double period_max;
    double duty_ratio_min;
    double duty_ratio_max;
    double eposide_len;
    double policy_delay;

    // intervals
    double push_interval;
    double turn_interval;

    // reward
    double rew_vel_xy;
    double rew_rotvel;
    double rew_height;
    double rew_torque;
    double rew_tracking;
    double rew_up;
    double rew_front;
    double rew_jerk;
    double rew_vjerk;

    // curriculum
    double terrain_height;
    bool do_dyn_rand;
    bool do_curriculum;

    // config, computed by LoadParamFromDict()
    double period_mean;
    double period_scale;
    double period_scale_inv;
    double height_mean;
    double height_scale;
    double height_scale_inv;
    double duty_ratio_mean;
    double duty_ratio_scale;
    double duty_ratio_scale_inv;
};

ATTR_API void LoadParamFromDict(A1RLParam& dst,
                                const std::map<std::string, double> &src);

} // namespace mujoco_parallel_sim

#endif // PARAM_LOADER_H_
