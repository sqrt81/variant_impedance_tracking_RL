
from typing import Dict, Any, Tuple

import numpy as np
import os
import torch

from rl_games.common.ivecenv import IVecEnv
import _mj_parallel

from gym import spaces

class MujocoA1(IVecEnv):
    __impl = _mj_parallel.ParallelSim

    def __init__(self, cfg, headless, rl_device):

        self.cfg = cfg
        self.rl_device = rl_device
        self.headless = headless
        
        # params
        self.fix_pd = self.cfg["env"]["fixPD"]
        self.rl_params = {}
        self.rl_params["step_dt"] = self.cfg["env"]["stepDt"]
        self.rl_params["vel_x_min"] = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"][0]
        self.rl_params["vel_x_max"] = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"][1]
        self.rl_params["vel_y_min"] = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"][0]
        self.rl_params["vel_y_max"] = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"][1]
        self.rl_params["acc"]       = self.cfg["env"]["randomCommandVelocityRanges"]["acc"]
        self.rl_params["period_min"]    = self.cfg["env"]["randomCommandVelocityRanges"]["period"][0]
        self.rl_params["period_max"]    = self.cfg["env"]["randomCommandVelocityRanges"]["period"][1]
        self.rl_params["duty_ratio_min"]    = self.cfg["env"]["randomCommandVelocityRanges"]["duty_ratio"][0]
        self.rl_params["duty_ratio_max"]    = self.cfg["env"]["randomCommandVelocityRanges"]["duty_ratio"][1]
        self.rl_params["height_min"]    = self.cfg["env"]["randomCommandVelocityRanges"]["torso_height"][0]
        self.rl_params["height_max"]    = self.cfg["env"]["randomCommandVelocityRanges"]["torso_height"][1]
        self.rl_params["eposide_len"]   = self.cfg["env"]["learn"]["episodeLength_s"]
        self.rl_params["terrain_height"] = self.cfg["env"]["terrain"]["max_height"]
        self.rl_params["policy_delay"] = self.cfg["env"]["policyDelay"]
        
        self.rl_params["push_interval"] = self.cfg["env"]["disturb"]["pushInterval_s"]
        self.rl_params["turn_interval"] = self.cfg["env"]["disturb"]["turnInterval_s"]
        self.rl_params["do_dyn_rand"] = self.cfg["env"]["learn"].get("dynamicRandomization", 1.)
        self.rl_params["do_curriculum"] = self.cfg["env"]["learn"].get("curriculum", 1.)
        
        self.clip_obs = self.cfg["env"]["clipObservations"]
        self.clip_actions = self.cfg["env"]["clipActions"]

        # reward scales
        self.rl_params["rew_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rl_params["rew_rotvel"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rl_params["rew_height"] = self.cfg["env"]["learn"]["heightRewardScale"]
        self.rl_params["rew_torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rl_params["rew_tracking"] = self.cfg["env"]["learn"]["trackingRewardScale"]
        self.rl_params["rew_up"] = self.cfg["env"]["learn"]["upRewardScale"]
        self.rl_params["rew_front"] = self.cfg["env"]["learn"]["frontRewardScale"]
        self.rl_params["rew_jerk"] = self.cfg["env"]["learn"]["jerkRewardScale"]
        self.rl_params["rew_vjerk"] = self.cfg["env"]["learn"]["vjerkRewardScale"]
        
        self.num_threads = self.cfg["sim"]["num_threads"]

        self.cfg["env"]["numObservations"] = self.__impl.kOBS_SIZE
        self.cfg["env"]["numActions"] = 24 if self.fix_pd else self.__impl.kACT_SIZE
        
        self.num_actions = self.cfg["env"]["numActions"]
        self.num_obs = self.cfg["env"]["numObservations"]
        self.num_envs = self.cfg["env"]["numEnvs"]
        self.action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
        self.observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        
        global mjc_env_obj
        self._env_impl = self.__impl(self.num_envs, min(self.num_envs, self.num_threads))
        mjc_env_obj = self._env_impl
        self._env_impl.LoadParam(self.rl_params)
        self._env_impl.LoadModelXml(os.path.join(os.path.dirname(__file__), "unitree_a1/hfield.xml"))

        self.device = "cpu"
        self.eval_mode = False
        self.allocate_buffers()
        
    def allocate_buffers(self):
        self.act_buf = torch.zeros(
            (self.num_envs, self.__impl.kACT_SIZE), device=self.device, dtype=torch.float64)
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float64)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float64)
        self.done_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.obs_dict = {}
        
        self.act_np = self.act_buf.numpy()
        self.obs_np = self.obs_buf.numpy()
        self.rew_np = self.rew_buf.numpy()
        self.done_np = self.done_buf.numpy()
        self.timeout_np = self.timeout_buf.numpy()
        self.vel_np = np.zeros(0)
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        assert actions.isfinite().all()
        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        if self.fix_pd:
            self.act_buf[:, 24:36] = action_tensor[:,  0:12].type_as(self.act_buf)
            self.act_buf[:, 60:72] = action_tensor[:, 12:24].type_as(self.act_buf)
        else:
            self.act_buf[:] = action_tensor.type_as(self.act_buf)
        self.process_actions()
        
        self._env_impl.Step(self.act_np, self.obs_np, self.done_np, self.rew_np, self.timeout_np, self.vel_np)
        
        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device).to(torch.long)
        self.obs_dict["obs"] = self.obs_buf.to(self.rl_device).to(torch.float).clamp_(-self.clip_obs, self.clip_obs)
        reward = self.rew_buf.to(self.rl_device).to(torch.float)
        reset = self.done_buf.to(self.rl_device).to(torch.long)
        self.extras["episode"] = self._env_impl.GetStat()
        if len(self.vel_np) > 0:
            self.extras["v_mean"] = self.vel_np
        
        if self.headless == False:
            self.render()
        
        return self.obs_dict, reward, reset, self.extras
    
    def reset(self) -> torch.Tensor:
        zero_actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.device)
        self.obs_dict, _, _, _ = self.step(zero_actions)
        return self.obs_dict
    
    def render(self):
        self._env_impl.Render()
    
    def process_actions(self):
        if self.fix_pd:
            self.act_buf[:,  0:12] = 20
            self.act_buf[:, 12:24] = 0.5
            self.act_buf[:, 36:48] = 20
            self.act_buf[:, 48:60] = 0.5
        else:
            self.act_buf[:,  0:12] = torch.exp(self.act_buf[:,  0:12] * 1.5 + 2.5)
            self.act_buf[:, 12:24] = torch.exp(self.act_buf[:, 12:24])
            self.act_buf[:, 36:48] = torch.exp(self.act_buf[:, 36:48] * 1.5 + 2.5)
            self.act_buf[:, 48:60] = torch.exp(self.act_buf[:, 48:60])
        self.act_buf[:, 24:36] = self.act_buf[:, 24:36] * 15.
        self.act_buf[:, 60:72] = self.act_buf[:, 60:72] * 15.
    
    def get_difficulties(self):
        v_cmd = np.zeros(self.num_envs, dtype=np.float64)
        h_cmd = np.zeros(self.num_envs, dtype=np.float64)
        
        self._env_impl.GetDifficulty(v_cmd, h_cmd)
        return v_cmd, h_cmd
    
    def get_max_difficulties(self):
        v_max = np.zeros(self.num_envs, dtype=np.float64)
        h_max = np.zeros(self.num_envs, dtype=np.float64)
        
        self._env_impl.GetMaxDifficulty(v_max, h_max)
        return v_max, h_max

    def set_eval_mode(self, vel, height, terrain_idx):
        if not self.eval_mode:
            self._env_impl.LoadModelXml(os.path.join(os.path.dirname(__file__), "unitree_a1/hfield_raw.xml"))
            self.vel_np = np.zeros(self.num_envs)
            self.eval_mode = True
        self._env_impl.SetToEvalMode(vel, height, terrain_idx)

    def get_number_of_agents(self):
        return 1

    def get_env_info(self):
        info = {}
        info['action_space'] = self.action_space
        info['observation_space'] = self.observation_space
        return info



