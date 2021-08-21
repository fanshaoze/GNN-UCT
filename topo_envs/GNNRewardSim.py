import torch

from topo_envs.surrogateRewardSim import SurrogateRewardTopologySim
from UCT_for_CD_analytics.ucts.TopoPlanner import TopoGenState


class GNNRewardSim(SurrogateRewardTopologySim):
    def __init__(self, eff_model_file, vout_model_file, debug=False, *args):
        super().__init__(debug, *args)

        self.eff_model = torch.load(eff_model_file)
        self.vout_model = torch.load(vout_model_file)

    def get_surrogate_eff(self, state:TopoGenState):
        # TODO
        pass

    def get_surrogate_vout(self, state:TopoGenState):
        # TODO
        pass
