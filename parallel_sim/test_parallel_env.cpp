#include "parallel_sim.h"

#include <signal.h>
#include <chrono>
#include <iostream>
#include <Eigen/Eigen>

namespace {

const char* helper_msg = "args: num_envs, num_threads";

bool terminate = false;

void SignalHandler(int signal)
{
    if (signal == SIGINT)
        terminate = true;
}

}

int main(int argc, char **argv)
{
    bool render = false;
    int num_envs = 1;
    int num_threads = 1;

    if (argc == 1)
        render = true;
    else if (argc != 3) {
        std::cout << helper_msg << std::endl;
        exit(1);
    }
    else {
        num_envs = std::atoi(argv[1]);
        num_threads = std::atoi(argv[2]);
    }

    if (num_envs <= 0 || num_threads <= 0) {
        std::cout << helper_msg << std::endl;
        exit(1);
    }

    mujoco_parallel_sim::ParallelSimEnv env(num_envs, num_threads);
    std::map<std::string, double> par_dict = {
        std::pair("step_dt", 0.02),

        std::pair("vel_x_min", 0.),
        std::pair("vel_x_max", 4.),
        std::pair("vel_y_min", -0.1),
        std::pair("vel_y_max", 0.1),
        std::pair("acc", 0.33),
        std::pair("height_min", 0.3),
        std::pair("height_max", 0.35),
        std::pair("period_min", 0.3),
        std::pair("period_max", 0.4),
        std::pair("duty_ratio_min", 0.25),
        std::pair("duty_ratio_max", 0.3),
        std::pair("eposide_len", 10.),
        std::pair("policy_delay", 0.0),

        std::pair("push_interval", 4.),
        std::pair("turn_interval", 3.),

        std::pair("rew_vel_xy", 2.),
        std::pair("rew_rotvel", 0.5),
        std::pair("rew_height", 0.5),
        std::pair("rew_torque", -1e-6),
        std::pair("rew_tracking", 1.),
        std::pair("rew_up", 0.2),
        std::pair("rew_front", 0.5),
        std::pair("rew_jerk", -0.5),
        std::pair("rew_vjerk", -0.5),

        std::pair("terrain_height", 0.06),
    };
    env.LoadParams(par_dict);
    env.LoadModelXml("unitree_a1/hfield.xml");

    Eigen::Array<double, -1, -1, Eigen::RowMajor> obs;
    Eigen::Array<double, -1, -1, Eigen::RowMajor> act;
    Eigen::ArrayXd rew;
    bool done[num_envs];
    bool timeout[num_envs];
    int step_cnt = 0;

    obs.resize(num_envs, mujoco_parallel_sim::A1GymEnv::kOBS_SIZE);
    act.resize(num_envs, mujoco_parallel_sim::A1GymEnv::kACT_SIZE);
    rew.resize(num_envs);

    for (int i = 0; i < num_envs; i++) {
        for (int j = 0; j < 12; j++) {
            act(i, j     ) = act(i, j + 36) = 100.; // kp
            act(i, j + 12) = act(i, j + 48) = 5.;   // kd
            act(i, j + 24) = act(i, j + 60) = 0.;   // torq
        }
    }

    signal(SIGINT, SignalHandler);
    auto tm_start = std::chrono::system_clock::now();

    while (!terminate) {
        env.Step(act.data(), obs.data(), done, rew.data(), timeout);

        if (render)
            env.Render();

        step_cnt++;
    }

    auto tm_end = std::chrono::system_clock::now();

    // compute statistics
    std::chrono::duration<double> elapsed = tm_end - tm_start;
    double vel, eposide_len, terrain_height;
    env.GetStatistics(vel, eposide_len, terrain_height);
    std::cout << "step: " << step_cnt << ", \t"
              << "duration: " << elapsed.count() << "s" << std::endl;
    std::cout << "Total samples: " << step_cnt * num_envs << std::endl;
    std::cout << "Average sample rate: "
              << step_cnt * num_envs / elapsed.count() << " per second"
              << std::endl;
    std::cout << "Mean eposide len: " << eposide_len << std::endl;
    std::cout << "Terrain height: " << terrain_height << std::endl;
}

