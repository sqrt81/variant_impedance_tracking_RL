import os
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Dict
from hydra.utils import to_absolute_path
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner

import mujoco_a1

import yaml

OmegaConf.register_new_resolver('eq', lambda x, y: x.lower()==y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
# allows us to resolve default arguments which are copied in multiple places in the config. used primarily for
# num_ensv
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)


def omegaconf_to_dict(d: DictConfig)->Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    if cfg.test:
        cfg.task.env.numEnvs = 1
        #cfg.task.env.learn.curriculum = 0.
        cfg.num_envs = 1
        cfg.pipeline = 'cpu'
        cfg.sim_device = 'cpu'
        cfg.rl_device = 'cpu'
        #cfg.headless = False

    def create_mujoco_a1_env():
        return mujoco_a1.MujocoA1(omegaconf_to_dict(cfg.task), headless=cfg.headless, rl_device=cfg.rl_device)

    # register the rl-games adapter to use inside the runner
    vecenv.register('mujoco_parallel',
                    lambda config_name, num_actors, **kwargs: env_configurations.configurations[config_name]['env_creator'](**kwargs))

    env_configurations.register('mujoco_parallel', {
        'vecenv_type': 'mujoco_parallel',
        'env_creator': lambda **kwargs: create_mujoco_a1_env(**kwargs),
    })

    rlg_config_dict = omegaconf_to_dict(cfg.train)

    runner = Runner()
    runner.load(rlg_config_dict)
    
    ckpt = os.path.join('runs', cfg.train.params.config.name, "nn",
                        cfg.train.params.config.name + ".pth")
    runner.run({
        'train': not cfg.test,
        'play': cfg.test,
        'checkpoint': ckpt if cfg.test else None
    })

if __name__ == "__main__":
    launch_rlg_hydra()

