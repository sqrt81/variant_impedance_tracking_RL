import _mj_parallel
from _mj_parallel import ParallelSim as SimEnv
import numpy as np
import time

if __name__ == "__main__":
    par_dict = {
        "vel_x_min": 0.,
        "vel_x_max": 4.,
        "vel_y_min": -0.1,
        "vel_y_max": 0.1,
        "acc": 0.33,
        "height_min": 0.3,
        "height_max": 0.35,
        "period_min": 0.3,
        "period_max": 0.4,
        "duty_ratio_min": 0.25,
        "duty_ratio_max": 0.3,
        "eposide_len": 10,
        
        "rew_vel_xy": 2.,
        "rew_rotvel": 0.5,
        "rew_height": 0.5,
        "rew_torque": -1e-6,
        "rew_tracking": 1.,
        "rew_up": 0.2,
        "rew_front": 0.5,
        "rew_jerk": -0.5,
        "rew_vjerk": -0.5,
    }
    
    env_size = 20
    num_threads = 10
    
    env = SimEnv(env_size, num_threads)
    env.LoadParam(par_dict)
    env.LoadModelXml("../../../mujoco_menagerie/unitree_a1/hfield.xml")
    act = np.zeros([env_size, env.kACT_SIZE])
    act[:,  0:12] = 100.
    act[:, 12:24] = 5.
    act[:, 36:48] = 100.
    act[:, 48:60] = 5.
    obs = np.zeros([env_size, env.kOBS_SIZE])
    done = np.zeros([env_size,], dtype=bool)
    timeout = np.zeros([env_size,], dtype=bool)
    rew = np.zeros([env_size,])
    
    tm_beg = time.time()
    
    for _ in range(500):
        env.Step(act, obs, done, rew, timeout)
        #env.Render()

    tm_end = time.time()
    print("time:", tm_end - tm_beg)
    stat = env.GetStat()
    print(stat)


