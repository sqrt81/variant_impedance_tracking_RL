#ifndef PARALLEL_SIM_H_
#define PARALLEL_SIM_H_

#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "safe_queue.hpp"
#include "a1_gym_env.h"
#include "param_loader.h"

struct mjModel_;
struct mjData_;

namespace mujoco_parallel_sim {

class ParallelSimEnv {
public:
    ATTR_API ParallelSimEnv(int num_envs, int num_threads);
    ATTR_API ~ParallelSimEnv();

    ATTR_API void LoadParams(const std::map<std::string, double> &par_dict);

    ATTR_API bool LoadModelXml(const std::string &file_path);

    ATTR_API void Step(const double *act,
                       double *obs, bool *done, double *reward,
                       bool *timeout, double *v_mean = nullptr);

    void GetStatistics(double &v_mean, double &eposide_time_mean,
                       double &mean_terrain_height) const
    {
        v_mean = mean_vel_;
        eposide_time_mean = mean_eposide_len_;
        mean_terrain_height = mean_terrain_height_;
    }

    ATTR_API void SetToEvalMode(double v_cmd, double terrain_height,
                                int terrain_idx);

    ATTR_API void GetEnvDifficulty(double *v_cmd, double *terrain_height);
    ATTR_API void GetEnvMaxDifficulty(double *v_cmd, double *terrain_height);

    ATTR_API void Render();

private:
    void ThreadPoolFunc(int thread_idx);

    bool AllThreadsIdle() const;

    std::condition_variable threads_cv_;
    std::mutex threads_mutex_;
    std::condition_variable finish_cv_;
    std::mutex finish_mutex_;

    std::vector<std::thread> thread_pool_;
    std::vector<A1GymEnv> sims_;
    std::vector<uint8_t> threads_busy_;

    bool terminated_ = false;

    SafeQueue<int> idx_to_update_;
    SafeQueue<int> all_idx_;

    const double *act_buf_ptr_;
    double *obs_buf_ptr_;
    bool *done_buf_ptr_;
    double *reward_buf_ptr_;
    bool *timeout_buf_ptr_;
    double *v_mean_buf_ptr_;

    // statistics
    std::mutex statistics_mutex_;
    double mean_eposide_len_ = 0.;
    double mean_vel_ = 0.;
    double mean_terrain_height_ = 0.;

    // original param
    A1RLParam base_param_;
    // one param for each simulation
    std::vector<A1RLParam> params_;
};

} //namespace mujoco_parallel_sim

#endif // PARALLEL_SIM_H_
