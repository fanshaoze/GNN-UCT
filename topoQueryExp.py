import csv
import json
import sys, os
import numpy as np
import torch
import time

import config
from arguments import get_args

sys.path.append(os.path.join(sys.path[0], 'topo_data_util'))
sys.path.append(os.path.join(sys.path[0], 'transformer_SVGP'))
sys.path.append(os.path.join(sys.path[0], 'PM_GNN/code'))
if config.task == 'uct_3_comp':
    sys.path.append(os.path.join(sys.path[0], 'UCT_for_CD_no_sweep'))
    from UCT_for_CD_no_sweep.main import main as run_uct
elif config.task == 'rs_5_comp':
    sys.path.append(os.path.join(sys.path[0], 'UCT_5_only_epr_standard'))
    from UCT_5_only_epr_standard.main import main as run_rs
else:
    raise Exception('unknown task ' + config.task)

from util import feed_random_seeds


def gp_reward_uct_exp(args):
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'

    output_csv_header = ['random seed', 'query num',
                         'reward after ' + str(args.query_times) + ' queries',
                         'time after ' + str(args.query_times) + ' queries']

    if args.model == 'analytics':
        sim_init = None  # None forces uct to use NGSPice Simulator
    elif args.model == 'simulator':
        pass  # None forces uct to use NGSPice Simulator

    elif args.model == 'gp':
        from topo_envs.GPRewardSim import GPRewardTopologySim

        def sim_init(*a):
            return GPRewardTopologySim('efficiency.pt', 'vout.pt', args.debug, *a)
    elif args.model == 'transformer':
        from topo_envs.TransformerRewardSim import TransformerRewardSim

        postfix = '_cpu' if device == 'cpu' else ''

        def sim_init(*a):
            return TransformerRewardSim('transformer_SVGP/save_model/batch_gp_eff' + postfix + '.pt',
                                        'transformer_SVGP/save_model/batch_gp_vout' + postfix + '.pt',
                                        'transformer_SVGP/vocab.json',
                                        device,
                                        args.debug,
                                        *a)
    elif args.model == 'gnn':
        from topo_envs.GNNRewardSim import GNNRewardSim

        def sim_init(*a):
            return GNNRewardSim('reg_eff.pt', 'reg_vout.pt', args.debug, *a)
    else:
        raise Exception('unknown model ' + args.model)

    if args.seed_range is not None:
        seed_range = range(args.seed_range[0], args.seed_range[1])
    elif args.seed is not None:
        seed_range = [args.seed]
    else:
        # just run random seed 0 by default
        seed_range = [0]

    results = {}
    for k in args.k_list:
        results[k] = {}
        for _traj_num in args.traj:
            results[k][_traj_num] = []
    for traj_num in args.traj:
        results[traj_num] = []
    rows = [] # record rewards after each query, size: num of query * num of random seed
    last_rows = []
    for seed in seed_range:
        result = {}
        for k in args.k_list:
            result[k] = []
        for traj_num in args.traj:
            feed_random_seeds(seed)

            start_time = time.time()
            if config.task == 'uct_3_comp':
                from UCT_for_CD_no_sweep.config import _configs
                info = run_uct(Sim=sim_init, traj=traj_num, args_=_configs)
            elif config.task == 'rs_5_comp':
                info = run_rs(Sim=sim_init, traj=traj_num, args_file_name='UCT_5_only_epr_standard/config')
            else:
                raise Exception('unknown task ' + config.task)

            sim = info['sim']
            cand_states = sim.no_isom_seen_state_list
            query_num = info['query_num']


            if args.model == 'simulator':
                # sort all (reward, eff, vout) by rewards
                sorted_performances = sorted(sim.graph_2_reward.values(), key=lambda _: _[0])
            else:
                # for surrogate models
                surrogate_rewards = np.array([sim.get_reward(state) for state in cand_states])

            print(args.k_list)
            for k in args.k_list:
                if args.model == 'simulator':
                    top_k = sorted_performances[-k:]
                    top_1 = sorted_performances[-1]
                    # this is dummy for simulator
                    surrogate_top_k = []
                else:
                    # k topologies with highest surrogate rewards
                    candidate_indices = surrogate_rewards.argsort()[-k:]

                    surrogate_top_k = [(sim.get_reward(cand_states[idx]), sim.get_surrogate_eff(cand_states[idx]),
                                        sim.get_surrogate_vout(cand_states[idx]))
                                       for idx in candidate_indices]

                    # the true (reward, eff, vout) of top k topologies decided by the surrogate model
                    top_k = [sim.get_true_performance(cand_states[idx]) for idx in candidate_indices]
                    # the top one (reward, eff, vout) in the set above
                    top_1 = max(top_k, key=lambda _: _[0])

                    with open('./UCT_for_CD_no_sweep/SimulatorAnalysis/database/key-sim-result.json', 'w') as f:
                        json.dump(sim.key_sim_effi_, f)
                    f.close()

                end_time = time.time()
                execution_time = round(end_time - start_time, 2)

                result[k].append((query_num, {'top_k': top_k, 'top_1': top_1, 'surrogate_top_k': surrogate_top_k,
                                           'time': execution_time}))
                # TODO for debug 'surrogate_top_k': surrogate_top_k
                results[k][traj_num].append([seed, query_num, top_1[0], execution_time])

    file_signal = str(os.getpid())
    for k in args.k_list:
        for traj_num in args.traj:
            with open('Results/raw-5-comp_result_top_' + str(k) + '_traj_' + str(traj_num) + '_' + file_signal + '.csv', 'w') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(output_csv_header)
                csv_writer.writerows(results[k][traj_num])

        # save to file after each random seed
        with open(args.output + '.json', 'w') as f:
            json.dump(results, f)



if __name__ == '__main__':
    args = get_args()
    gp_reward_uct_exp(args)
