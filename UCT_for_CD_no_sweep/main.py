import datetime
import os

from utils.util import mkdir, init_position, generate_depth_list, generate_traj_List, \
    remove_tmp_files, save_results_to_csv
from Algorithms.SerialUCF import serial_UCF_test
from Algorithms.GeneticSearch import genetic_search
from Algorithms.test_analytic_simulator import anay_read_test
from Viz.VIZ import viz_test


def generate_traj_lists(trajectories, test_number):
    traj_lists = []
    for traj in trajectories:
        traj_list = []
        for _ in range(test_number):
            traj_list.append(traj)
        traj_lists.append(traj_list)
    return traj_lists


def main(name='', traj=None, Sim=None, args_=None, uct_tree_list=None, keep_uct_tree=False):
    mkdir("figures")
    mkdir("Results")
    if args_ is None:
        from config import _configs
        configs = _configs
    else:
        configs = args_

    configs["min_vout"] = configs["target_vout"] - configs["range_percentage"] * abs(configs["target_vout"])
    configs["max_vout"] = configs["target_vout"] + configs["range_percentage"] * abs(configs["target_vout"])
    depth_list = generate_depth_list(configs["dep_start"], configs["dep_end"], configs["dep_step_len"])

    anal_output_results = [[] for i in range(configs['test_number'] + 1)]
    simu_output_results = [[] for i in range(configs['test_number'] + 1)]

    date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    pid_str = str(os.getpid())
    result_folder = pid_str + '/' + date_str

    if traj == None:
        traj_list = configs['trajectories']
        print(traj_list)
    else:
        traj_list = [traj]

    # traj_list = [20, 21, 22, 23]
    for traj in traj_list:
        date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        if configs["algorithm"] == "UCF":
            if configs["root_parallel"] is False:
                result, anal_results, simu_results = serial_UCF_test(trajectory=traj,
                                                                     test_number=configs['test_number'],
                                                                     result_folder=result_folder,
                                                                     configs=configs,
                                                                     Sim=Sim,
                                                                     uct_tree_list=uct_tree_list,
                                                                     keep_uct_tree=keep_uct_tree)
                if Sim is None:  # just for uct with simulator or analytics
                    for i in range(configs['test_number'] + 1):
                        anal_output_results[i].extend(anal_results[i])
                        simu_output_results[i].extend(simu_results[i])
        # elif configs["algorithm"] == "GeneticSearch" or configs["algorithm"] == "GS":
        #     genetic_search(configs, date_str)
        # elif configs["algorithm"] == "VIZ":
        #     mkdir("Viz/TreeStructures")
        #     viz_test(depth_list, traj_list, configs, date_str)
    if Sim is None:  # just for uct with simulator or analytics
        anal_out_file_name = "Results/UCT-" + str(configs["target_vout"]) + "-anal-" + date_str + "-" + str(
            os.getpid()) + ".csv"
        save_results_to_csv(anal_out_file_name, anal_output_results)
        simu_out_file_name = "Results/UCT-" + str(configs["target_vout"]) + "-simu-" + date_str + "-" + str(
            os.getpid()) + ".csv"
        save_results_to_csv(simu_out_file_name, simu_output_results)
    remove_tmp_files()

    return result


if __name__ == '__main__':
    main('PyCharm')
