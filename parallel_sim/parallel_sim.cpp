#include "parallel_sim.h"

#include <mujoco/mujoco.h>

#include <iostream>

#include <GLFW/glfw3.h>

namespace mujoco_parallel_sim {

namespace  {

mjModel* ref_model = nullptr;       // MuJoCo model
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context
GLFWwindow *window = nullptr;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// mouse button callback
void mouse_button(GLFWwindow* window, int , int , int ) {
  // update button state
  button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
  button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
  button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

  // update mouse position
  glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos) {
  // no buttons down: nothing to do
  if (!button_left && !button_middle && !button_right) {
    return;
  }

  // compute mouse displacement, save
  double dx = xpos - lastx;
  double dy = ypos - lasty;
  lastx = xpos;
  lasty = ypos;

  // get current window size
  int width, height;
  glfwGetWindowSize(window, &width, &height);

  // get shift key state
  bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                    glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

  // determine action based on mouse button
  mjtMouse action;
  if (button_right) {
    action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
  } else if (button_left) {
    action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
  } else {
    action = mjMOUSE_ZOOM;
  }

  // move camera
  mjv_moveCamera(ref_model, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* , double , double yoffset) {
  // emulate vertical mouse motion = 5% of window height
  mjv_moveCamera(ref_model, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

} // anonymous namespace

ParallelSimEnv::ParallelSimEnv(int num_envs, int num_threads)
{
    sims_.resize(num_envs);
    thread_pool_.reserve(num_threads);
    threads_busy_.resize(num_threads, 0);

    for (int i = 0; i < num_envs; i++) {
        all_idx_.Push(i);
    }

    for (int i = 0; i < num_threads; i++)
        thread_pool_.emplace_back(
                    std::thread(&ParallelSimEnv::ThreadPoolFunc, this, i));
}

ParallelSimEnv::~ParallelSimEnv()
{
    terminated_ = true;
    threads_cv_.notify_all();
    for (auto &thread : thread_pool_)
        thread.join();

    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    if (ref_model) {
        mj_deleteModel(ref_model);
        ref_model = nullptr;
    }

    if (window)
        glfwDestroyWindow(window);
}

void ParallelSimEnv::LoadParams(const std::map<std::string, double> &par_dict)
{
    LoadParamFromDict(base_param_, par_dict);
    params_.clear();
    params_.resize(sims_.size(), base_param_);

    // limit 20% of envs to have less challenging terrain & speed command
    const int num_env = sims_.size();
    const int num_easier_env = num_env * 0.4;

    if (num_easier_env == 0)
        return; // no limit as there is not enough terrain

    const int cols = std::sqrt(num_env);

    int row_idx = 0;
    int col_idx = 0;
    for (int i = 0; i < num_easier_env; i++) {
        double difficulty_ratio = (i * 1.) / num_easier_env;
        int skip_idx = row_idx * cols + col_idx;
        if (skip_idx >= num_env) {
            col_idx++;
            row_idx = 0;
            skip_idx = col_idx;
        }
        else
            row_idx++;
        params_[skip_idx].terrain_height
                = base_param_.terrain_height * difficulty_ratio;
        params_[num_env - 1 - i].vel_x_max
                = base_param_.vel_x_max * difficulty_ratio;
    }
}

bool ParallelSimEnv::LoadModelXml(const std::string &file_path)
{
    char error[1000];
    ref_model = mj_loadXML(file_path.c_str(), 0, error, 1000);

    if (ref_model == nullptr) {
        std::cout << "Failed to load xml '" << file_path << "'." << std::endl
                  << error << std::endl;
        return false;
    }

    auto param_ptr = params_.begin();
    for (auto &sim : sims_) {
        sim.SetSimPars(Eigen::Vector2d(0., 0.08), 0.1);
        sim.InitFromMujocoModel(ref_model, *param_ptr);
        param_ptr++;
    }

    return true;
}

void ParallelSimEnv::Step(
        const double *act,
        double *obs, bool *done, double *reward,
        bool *timeout, double *v_mean)
{
    act_buf_ptr_ = act;
    obs_buf_ptr_ = obs;
    done_buf_ptr_ = done;
    reward_buf_ptr_ = reward;
    timeout_buf_ptr_ = timeout;
    v_mean_buf_ptr_ = v_mean;

    idx_to_update_ = all_idx_;
    threads_cv_.notify_all();

    int wait_cnt = 0;
    do {
        // just wait for all working threads to finish
        std::unique_lock<std::mutex> lock(finish_mutex_);
        finish_cv_.wait(lock);
        wait_cnt++;
    }
    while(!AllThreadsIdle()); // do a final check

    if (wait_cnt > 1) // should not happen
        std::cout << "Error! cv is notified " << wait_cnt
                  << "times. " << std::endl;
}

void ParallelSimEnv::SetToEvalMode(
        double v_cmd, double terrain_height, int terrain_idx)
{
    for (auto &par : params_) {
        par.vel_x_max = v_cmd;
        par.vel_x_min = v_cmd;
        par.do_dyn_rand = false;
        par.do_curriculum = false;
        par.height_scale_inv = 0;
        par.period_scale_inv = 0;
        par.duty_ratio_scale_inv = 0;
        par.height_scale = 0;
        par.period_scale = 0;
        par.duty_ratio_scale = 0;
    }
    for (auto &sim : sims_) {
        sim.SetTerrain(terrain_height, terrain_idx);
        sim.RequireReset();
    }
}

void ParallelSimEnv::GetEnvDifficulty(double *v_cmd, double *terrain_height)
{
    const int num_envs = sims_.size();
    for (int i = 0; i < num_envs; i++) {
        sims_[i].GetDifficulty(v_cmd[i], terrain_height[i]);
    }
}

void ParallelSimEnv::GetEnvMaxDifficulty(double *v_cmd, double *terrain_height)
{
    const int num_envs = sims_.size();
    for (int i = 0; i < num_envs; i++) {
        sims_[i].GetMaxDifficulty(v_cmd[i], terrain_height[i]);
    }
}

void ParallelSimEnv::Render()
{
    if (window == nullptr) {
        // initialization
        if (!glfwInit()) {
          std::cout << "Could not initialize GLFW" << std::endl;
          exit(1);
        }

        window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        // initialize visualization data structures
        mjv_defaultCamera(&cam);
        mjv_defaultOption(&opt);
        mjv_defaultScene(&scn);
        mjr_defaultContext(&con);

        // install GLFW mouse and keyboard callbacks
        glfwSetCursorPosCallback(window, mouse_move);
        glfwSetMouseButtonCallback(window, mouse_button);
        glfwSetScrollCallback(window, scroll);

        // just render the first env
        if (sims_.size() >= 1)
            sims_[0].StartRendering(&cam, &opt, &scn, &con);
        else {
            std::cout << "No env to render!" << std::endl;
            exit(1);
        }
    }

    sims_[0].Render();
    // get framebuffer viewport
    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
    mjr_render(viewport, &scn, &con);

    // swap OpenGL buffers (blocking call due to v-sync)
    glfwSwapBuffers(window);

    // process pending GUI events, call GLFW callbacks
    glfwPollEvents();
}

void ParallelSimEnv::ThreadPoolFunc(int thread_idx)
{
    bool has_tasks = false;
    int task_idx = -1;
    uint8_t &thread_busy_ = threads_busy_[thread_idx];

    while (!terminated_) {
        {
            std::unique_lock<std::mutex> lock(threads_mutex_);
            thread_busy_ = idx_to_update_.TryPop(task_idx);

            if (!thread_busy_) {
                if (has_tasks) { // had tasks last time, but idle now
                    if (AllThreadsIdle()) // all threads are idle
                        finish_cv_.notify_one(); // notify the main thread
                }

                has_tasks = false;
                task_idx = -1;
                threads_cv_.wait(lock);
            }
            else
                has_tasks = true;
        }

        if (has_tasks) {
            sims_[task_idx].Step(
                        act_buf_ptr_ + task_idx * A1GymEnv::kACT_SIZE,
                        obs_buf_ptr_ + task_idx * A1GymEnv::kOBS_SIZE,
                        done_buf_ptr_[task_idx], reward_buf_ptr_[task_idx],
                        timeout_buf_ptr_[task_idx]);
            if (done_buf_ptr_[task_idx]) {
                double v_mean;
                double episode_time;
                double terrain_height;
                sims_[task_idx].GetStatics(
                            v_mean, episode_time, terrain_height);
                if (v_mean_buf_ptr_ != nullptr)
                    v_mean_buf_ptr_[task_idx] = v_mean;

                {
                    std::lock_guard s_guard(statistics_mutex_);
                    mean_eposide_len_ = 0.98 * mean_eposide_len_
                            + 0.02 * episode_time;
                    mean_vel_ = 0.98 * mean_vel_ + 0.02 * v_mean;
                    mean_terrain_height_ = 0.98 * mean_terrain_height_
                            + 0.02 * terrain_height;
                }
            }
        }
    }
}

bool ParallelSimEnv::AllThreadsIdle() const
{
    for (uint8_t busy : threads_busy_)
        if (busy)
            return false;

    return true;
}

} //namespace mujoco_parallel_sim

