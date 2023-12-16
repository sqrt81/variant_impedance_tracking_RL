#include "param_loader.h"

#include <iostream>

namespace mujoco_parallel_sim {

double FindElemOrDie(const std::map<std::string, double> &set,
                     const std::string &key)
{
    auto iter = set.find(key);
    if (iter == set.end()) {
        std::cout << "Error: can't find key '" << key << "'." << std::endl;
        exit(1);
    }
    return iter->second;
}

double FindElem(const std::map<std::string, double> &set,
                const std::string &key,
                double default_val)
{
    auto iter = set.find(key);
    if (iter == set.end()) {
        std::cout << "key '" << key << "' not found, using default: "
                  << default_val << "." << std::endl;
        return default_val;
    }
    return iter->second;
}

void LoadParamFromDict(A1RLParam& dst,
                       const std::map<std::string, double> &src)
{
    dst.step_dt         = FindElemOrDie(src, "step_dt");

    dst.vel_x_min       = FindElemOrDie(src, "vel_x_min");
    dst.vel_x_max       = FindElemOrDie(src, "vel_x_max");
    dst.vel_y_min       = FindElemOrDie(src, "vel_y_min");
    dst.vel_y_max       = FindElemOrDie(src, "vel_y_max");
    dst.acc             = FindElemOrDie(src, "acc");
    dst.height_min      = FindElemOrDie(src, "height_min");
    dst.height_max      = FindElemOrDie(src, "height_max");
    dst.period_min      = FindElemOrDie(src, "period_min");
    dst.period_max      = FindElemOrDie(src, "period_max");
    dst.duty_ratio_min  = FindElemOrDie(src, "duty_ratio_min");
    dst.duty_ratio_max  = FindElemOrDie(src, "duty_ratio_max");
    dst.eposide_len     = FindElemOrDie(src, "eposide_len");
    dst.policy_delay    = FindElemOrDie(src, "policy_delay");

    dst.push_interval   = FindElemOrDie(src, "push_interval");
    dst.turn_interval   = FindElemOrDie(src, "turn_interval");

    dst.rew_vel_xy      = FindElemOrDie(src, "rew_vel_xy");
    dst.rew_rotvel      = FindElemOrDie(src, "rew_rotvel");
    dst.rew_height      = FindElemOrDie(src, "rew_height");
    dst.rew_torque      = FindElemOrDie(src, "rew_torque");
    dst.rew_tracking    = FindElemOrDie(src, "rew_tracking");
    dst.rew_up          = FindElemOrDie(src, "rew_up");
    dst.rew_front       = FindElemOrDie(src, "rew_front");
    dst.rew_jerk        = FindElemOrDie(src, "rew_jerk");
    dst.rew_vjerk       = FindElemOrDie(src, "rew_vjerk");

    dst.terrain_height  = FindElemOrDie(src, "terrain_height");
    dst.do_dyn_rand     = (FindElem(src, "do_dyn_rand", 1) > 0.5);
    dst.do_curriculum   = (FindElem(src, "do_curriculum", 1) > 0.5);

    dst.height_mean         = 0.5 * (dst.height_min + dst.height_max);
    dst.height_scale        = 0.5 * (dst.height_max - dst.height_min);
    dst.height_scale_inv    = 1. / dst.height_scale;
    dst.period_mean         = 0.5 * (dst.period_min + dst.period_max);
    dst.period_scale        = 0.5 * (dst.period_max - dst.period_min);
    dst.period_scale_inv    = 1. / dst.period_scale;
    dst.duty_ratio_mean     = 0.5 * (dst.duty_ratio_min + dst.duty_ratio_max);
    dst.duty_ratio_scale    = 0.5 * (dst.duty_ratio_max - dst.duty_ratio_min);
    dst.duty_ratio_scale_inv    = 1. / dst.duty_ratio_scale;
}

} // namespace mujoco_parallel_sim
