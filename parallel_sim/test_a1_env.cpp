#include "a1_gym_env.h"

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <iostream>
#include <thread>
#include <Eigen/Eigen>

#include "eigen_model_from_file.hpp"

namespace  {

mjModel* ref_model = NULL;          // MuJoCo model
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context
double obs[mujoco_parallel_sim::A1GymEnv::kOBS_SIZE];
bool env_paused = false;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;

// keyboard callback
void keyboard(GLFWwindow* , int key, int , int act, int ) {
  // p: print obs && pause env
  if (act==GLFW_PRESS && key==GLFW_KEY_P) {
    env_paused = !env_paused;
    if (env_paused) {
        Eigen::Map<Eigen::Array<double, 1,
                mujoco_parallel_sim::A1GymEnv::kOBS_SIZE>>
                obs_arr(obs);
        std::cout << "Scaled variables: " << std::endl;
        std::cout << "Linear vel:       " << obs_arr.segment<3>(0) << std::endl;
        std::cout << "Angular vel:      " << obs_arr.segment<3>(3) << std::endl;
        std::cout << "Grav direction:   " << obs_arr.segment<3>(6) << std::endl;
        std::cout << "yaw:              " << obs_arr.segment<2>(9) << std::endl;
        std::cout << "cmd:              " << obs_arr.segment<3>(11) << std::endl;
        std::cout << "stat:             " << obs_arr.segment<3>(14) << std::endl;
        std::cout << "fl&br step:       " << obs_arr[65] << std::endl;
        std::cout << "fr&bl step:       " << obs_arr[66] << std::endl;
        std::cout << "============================================" << std::endl;
    }
  }
}

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

int main()
{
    mujoco_parallel_sim::A1GymEnv env;

    constexpr bool use_policy = true;
    EigenModelFromFile *policy = nullptr;

    if (use_policy)
        policy = new EigenModelFromFile("../models/net_delay_12cm");

    char error[1000];
    ref_model = mj_loadXML(
                "unitree_a1/hfield.xml", 0, error, 1000);
    mujoco_parallel_sim::A1RLParam par;
    std::map<std::string, double> par_dict = {
        std::pair("step_dt", 0.02),

        std::pair("vel_x_min", 2.),
        std::pair("vel_x_max", 3.),
        std::pair("vel_y_min", -0.1),
        std::pair("vel_y_max", 0.1),
        std::pair("acc", 0.33),
        std::pair("height_min", 0.29),
        std::pair("height_max", 0.31),
        std::pair("period_min", 0.23),
        std::pair("period_max", 0.27),
        std::pair("duty_ratio_min", 0.27),
        std::pair("duty_ratio_max", 0.33),
        std::pair("eposide_len", 10.),
        std::pair("policy_delay", 0.02),

        std::pair("push_interval", 4.),
        std::pair("turn_interval", 10.),
        std::pair("do_dyn_rand", 0.),
        std::pair("do_curriculum", 0.),

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
    mujoco_parallel_sim::LoadParamFromDict(par, par_dict);
    env.SetSimPars(Eigen::Vector2d(0., 0.08), 0.12);
    env.InitFromMujocoModel(ref_model, par);

    // init GLFW
    if (!glfwInit()) {
      mju_error("Could not initialize GLFW");
    }

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    env.StartRendering(&cam, &opt, &scn, &con);

    double action[mujoco_parallel_sim::A1GymEnv::kACT_SIZE] = {0.};
    bool done;
    double rew;
    bool timeout;

    for (int i = 0; i < 12; i++) {
        // apply pd control
        action[i     ] = action[i + 36] = 100; // kp
        action[i + 12] = action[i + 48] = 5.; // kd
        action[i + 24] = action[i + 60] = 0.; // no torq
    }

    while (!glfwWindowShouldClose(window)) {
        if (!env_paused) {
            env.Step(action, obs, done, rew, timeout);

            if (use_policy) {
                const Eigen::VectorXf obs_float
                        = Eigen::Map<Eigen::VectorXd>(
                            obs, mujoco_parallel_sim::A1GymEnv::kOBS_SIZE)
                        .cast<float>();
                Eigen::ArrayXf a = (*policy)(obs_float);
                a = a.cwiseMax(-1.).cwiseMin(1.);

                if (a.size() == 72) {
                    a.segment<12>( 0) = (a.segment<12>( 0) * 1.5 + 3.).exp();
                    a.segment<12>(12) = (a.segment<12>(12) + 1.5).exp();
                    a.segment<12>(24) =  a.segment<12>(24) * 15.;
                    a.segment<12>(36) = (a.segment<12>(36) * 1.5 + 3.).exp();
                    a.segment<12>(48) = (a.segment<12>(48) + 1.5).exp();
                    a.segment<12>(60) =  a.segment<12>(60) * 15.;
                    Eigen::Map<Eigen::VectorXd>(
                                action, mujoco_parallel_sim::A1GymEnv::kACT_SIZE)
                            = a.cast<double>();
                }
                else {
                    Eigen::Map<Eigen::VectorXd> d(
                                action, mujoco_parallel_sim::A1GymEnv::kACT_SIZE);
                    d.segment<12>( 0).setConstant(28);
                    d.segment<12>(12).setConstant(0.7);
                    d.segment<12>(24) =  a.segment<12>(0).cast<double>() * 15.;
                    d.segment<12>(36).setConstant(28);
                    d.segment<12>(48).setConstant(0.7);
                    d.segment<12>(60) =  a.segment<12>(12).cast<double>() * 15.;
                }
            }
        }

        env.Render();
//        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    env.StopRendering();

    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteModel(ref_model);

    if (use_policy)
        delete policy;

    return 0;
}
