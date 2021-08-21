import torch

from PM_GNN.code.topo_data import split_balance_data
from topo_envs.surrogateRewardSim import SurrogateRewardTopologySim
from UCT_for_CD_analytics.ucts.TopoPlanner import TopoGenState
from PM_GNN.code.generate_dataset import generate_topo_for_GNN_model
from PM_GNN.code.ml_utils import get_effi_and_vout_with_model


class GNNRewardSim(SurrogateRewardTopologySim):
    def __init__(self, eff_model_file, vout_model_file, debug=False, *args):
        super().__init__(debug, *args)

        self.eff_model = torch.load(eff_model_file)
        self.vout_model = torch.load(vout_model_file)

    def get_surrogate_eff(self, state:TopoGenState):
        dataset = generate_topo_for_GNN_model(state)
        train_loader, val_loader, test_loader = split_balance_data(dataset, False, batch_size=32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ???return what???get_effi_and_vout_with_model(test_loader, self.eff_model, device, num_node=4, model_index=0)
        pass

    def get_surrogate_vout(self, state:TopoGenState):
        # TODO
        pass
