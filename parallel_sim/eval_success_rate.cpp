#include "a1_gym_env.h"

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>
#include <iostream>
#include <thread>
#include <stdexcept>
#include <Eigen/Eigen>

#include "eigen_model_from_file.hpp"

namespace  {

mjModel* ref_model = nullptr;       // reference model
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

bool env_paused = false;
bool run_once = false;

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
  }

  if (act==GLFW_PRESS && key==GLFW_KEY_1) {
    env_paused = false;
    run_once = true;
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

int main(int argc, char **argv)
{
    double terrain_height = 0.;
    int terrain_idx = 0;
    bool use_gui = false;
    int test_case_cnt = 100;
    double momentum_x = 0;
    double momentum_y = 0;
    double v_cmd = 2.0;

    // arg parse
    bool terrain_height_provided = false;
    bool terrain_idx_provided = false;
    bool test_case_cnt_provided = false;
    bool momentum_x_provided = false;
    bool momentum_y_provided = false;
    bool cmd_provided = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "v")
            use_gui = true;
        else {
            try {
                if (!test_case_cnt_provided) {
                    test_case_cnt = std::stoi(std::string(argv[i]));
                    test_case_cnt_provided = true;
                }
                else if (!momentum_x_provided) {
                    momentum_x = std::stof(std::string(argv[i]));
                    momentum_x_provided = true;
                }
                else if (!momentum_y_provided) {
                    momentum_y = std::stof(std::string(argv[i]));
                    momentum_y_provided = true;
                }
                else if (!cmd_provided) {
                    v_cmd = std::stof(std::string(argv[i]));
                    cmd_provided = true;
                }
                else if (!terrain_height_provided) {
                    terrain_height = std::stof(std::string(argv[i]));
                    terrain_height_provided = true;
                }
                else if (!terrain_idx_provided) {
                    terrain_idx = std::stoi(std::string(argv[i]));
                    terrain_idx_provided = true;
                }
            } catch (std::invalid_argument) {
                continue;
            }
        }
    }

    mujoco_parallel_sim::A1GymEnv env;
    EigenModelFromFile policy("../models/net_delay_12cm");

    char error[1000];
    ref_model = mj_loadXML(
                "unitree_a1/hfield.xml", 0, error, 1000);
    int obstacle_id = -1;
    for (int i = 0; i < ref_model->ngeom; i++)
        if (std::string(ref_model->names + ref_model->name_geomadr[i])
                == "obstacle") {
            obstacle_id = i;
            break;
        }

    const double obstacle_height = ref_model->geom_pos[obstacle_id * 3 + 2]
            + ref_model->geom_size[obstacle_id * 3 + 2];

    mujoco_parallel_sim::A1RLParam par;
    std::map<std::string, double> par_dict = {
        std::pair("step_dt", 0.02),

        std::pair("vel_x_min", v_cmd),
        std::pair("vel_x_max", v_cmd),
        std::pair("vel_y_min", 0.),
        std::pair("vel_y_max", 0.),
        std::pair("acc", 0.33),
        // fixed height & period & duty ratio
        std::pair("height_min", 0.2999),
        std::pair("height_max", 0.3001),
        std::pair("period_min", 0.2499),
        std::pair("period_max", 0.2501),
        std::pair("duty_ratio_min", 0.2999),
        std::pair("duty_ratio_max", 0.3001),
        std::pair("eposide_len", 7.),
        std::pair("policy_delay", 0.02),

        std::pair("push_interval", 4.e10), // no disturb
        std::pair("turn_interval", 3.e10),
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

        std::pair("terrain_height", 0.),
    };
    mujoco_parallel_sim::LoadParamFromDict(par, par_dict);
    par.height_scale_inv = 0;
    par.period_scale_inv = 0;
    par.duty_ratio_scale_inv = 0;
    par.height_scale = 0;
    par.period_scale = 0;
    par.duty_ratio_scale = 0;
    env.SetSimPars(Eigen::Vector2d(0., 0.08), 0.12);
    env.InitFromMujocoModel(ref_model, par);

    GLFWwindow* window = nullptr;

    if (use_gui) {
        // init GLFW
        if (!glfwInit()) {
            mju_error("Could not initialize GLFW");
        }

        // create window, make OpenGL context current, request v-sync
        window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
        glfwMakeContextCurrent(window);
        glfwSwapInterval(2);

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
    }

    double action[mujoco_parallel_sim::A1GymEnv::kACT_SIZE] = {0.};
    double obs[mujoco_parallel_sim::A1GymEnv::kOBS_SIZE];
    bool done;
    double rew;
    bool timeout;

    for (int i = 0; i < 12; i++) {
        // apply pd control
        action[i     ] = action[i + 36] = 100; // kp
        action[i + 12] = action[i + 48] = 5.; // kd
        action[i + 24] = action[i + 60] = 0.; // no torq
    }

    int tested_case = 0;
    int succeeded_case = 0;
    int step_cnt = 0;
    double v_final_sum = 0;
    bool fix_pd = true;
    double v_end[test_case_cnt];
    env.SetTerrain(terrain_height, terrain_idx);
    env.RequireReset();

    while (!use_gui || !glfwWindowShouldClose(window)) {
        if (!env_paused) {
            if (run_once) {
                env_paused = true;
                run_once = false;
            }
            if (step_cnt >= 100 && step_cnt < 125) {
                // apply push force from 2s to 2.5s
                env.ApplyImpulse(momentum_x / 25, momentum_y / 25);
            }
            env.Step(action, obs, done, rew, timeout);
            if (done) {
                double v_mean;
                double elapsed;
                double terrain_height;
                env.GetStatics(v_mean, elapsed, terrain_height);
                // if the robot fall before it can walk stably,
                // this episode doesn't count
                if (elapsed > 2.)
                    tested_case++;
                step_cnt = 0;
                if (timeout && v_mean > v_cmd - 0.5) {
                    v_end[succeeded_case] = v_mean;
                    succeeded_case++;
                    v_final_sum += v_mean;
                }

                if (tested_case >= test_case_cnt)
                    break;
            }
            else
                step_cnt++;

            {
                const Eigen::VectorXf obs_float
                        = Eigen::Map<Eigen::VectorXd>(
                            obs, mujoco_parallel_sim::A1GymEnv::kOBS_SIZE)
                        .cast<float>();
                Eigen::ArrayXf a = policy(obs_float);
                a = a.cwiseMax(-1.).cwiseMin(1.);

                if (a.size() == 72) {
                    fix_pd = false;
                    a.segment<12>( 0) = (a.segment<12>( 0) * 1.5 + 2.5).exp();
                    a.segment<12>(12) = (a.segment<12>(12)).exp();
                    a.segment<12>(24) =  a.segment<12>(24) * 15.;
                    a.segment<12>(36) = (a.segment<12>(36) * 1.5 + 2.5).exp();
                    a.segment<12>(48) = (a.segment<12>(48)).exp();
                    a.segment<12>(60) =  a.segment<12>(60) * 15.;
                    Eigen::Map<Eigen::VectorXd>(
                                action, mujoco_parallel_sim::A1GymEnv::kACT_SIZE)
                            = a.cast<double>();
                }
                else {
                    fix_pd = true;
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

        if (use_gui) {
            env.Render();

            // get framebuffer viewport
            mjrRect viewport = {0, 0, 0, 0};
            glfwGetFramebufferSize(window, &viewport.width, &viewport.height);
            mjr_render(viewport, &scn, &con);

            // swap OpenGL buffers (blocking call due to v-sync)
            glfwSwapBuffers(window);

            // process pending GUI events, call GLFW callbacks
            glfwPollEvents();
        }
    }

    if (use_gui) {
        env.StopRendering();

        mjv_freeScene(&scn);
        mjr_freeContext(&con);
        mj_deleteModel(ref_model);
    }

    const double v_final = succeeded_case ? v_final_sum / succeeded_case : 0;
    double v_mse = 0;
    for (int i = 0; i < succeeded_case; i++) {
        v_mse += (v_end[i] - v_final) * (v_end[i] - v_final);
    }
    const double v_std = succeeded_case ? std::sqrt(v_mse / succeeded_case) : 0;

//    std::cout << "Total cases: " << tested_case
//              << ", fix pd: " << (fix_pd ? "true" : "false")
//              << ", momentum: " << momentum_x << " " << momentum_y << std::endl
//              << "Height: " << obstacle_height
//              << ", success rate: "
//              << 1. * succeeded_case / tested_case
//              << ", v: " << v_final
//              << std::endl;
    if (momentum_x == 0 && momentum_y == 0) {
        std::cout << v_cmd << ", " << v_final << ", " << v_mse << "," << std::endl;
    }
    else if (momentum_x != 0) {
        std::cout << momentum_x << ", " << 1. * succeeded_case / tested_case
                  << ", " << std::endl;
    }
    else if (momentum_y != 0) {
        std::cout << momentum_y << ", " << 1. * succeeded_case / tested_case
                  << ", " << std::endl;
    }
//    std::cout << 1. * succeeded_case / tested_case << ", " << v_final << "," << std::endl;

    return 0;
}
