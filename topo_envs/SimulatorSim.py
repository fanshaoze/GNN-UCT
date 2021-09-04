import datetime

import torch

from algs.gp import GPModel
from topo_envs.surrogateRewardSim import SurrogateRewardTopologySim

from topo_data_util.embedding import tf_embed

import numpy as np


class SimulatorRewardTopologySim(SurrogateRewardTopologySim):
    def __init__(self, *args):
        pass

    def get_surrogate_eff(self, state):
        pass

    def get_surrogate_vout(self, state):
        pass

    def get_surrogate_reward(self, state):
        pass
