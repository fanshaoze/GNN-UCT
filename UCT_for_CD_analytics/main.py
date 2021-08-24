import datetime
from utils.util import mkdir, get_args, init_position, generate_depth_list, generate_traj_List
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


def main(name='', traj=None, Sim=None, args_file_name="config", uct_tree_list=None, keep_uct_tree=False):
    mkdir("figures")
    mkdir("Results")
    configs = {}
    get_args(args_file_name, configs)
    configs["min_vout"] = configs["target_vout"] - configs["range_percentage"] * abs(configs["target_vout"])
    configs["max_vout"] = configs["target_vout"] + configs["range_percentage"] * abs(configs["target_vout"])
    depth_list = generate_depth_list(configs["dep_start"], configs["dep_end"], configs["dep_step_len"])

    if traj is None:
        traj_list = generate_traj_List(configs["traj_start"], configs["traj_end"], configs["traj_step_len"])
    else:
        traj_list = [traj]

    # traj_list = [20, 21, 22, 23]
    date_str = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if configs["algorithm"] == "UCF":
        if configs["root_parallel"] is False:
            result = serial_UCF_test(depth_list, traj_list, configs, date_str, Sim=Sim,
                                     uct_tree_list=uct_tree_list, keep_uct_tree=keep_uct_tree)
    elif configs["algorithm"] == "GeneticSearch" or configs["algorithm"] == "GS":
        genetic_search(configs, date_str)
    elif configs["algorithm"] == "VIZ":
        mkdir("Viz/TreeStructures")
        viz_test(depth_list, traj_list, configs, date_str)
    # elif configs["algorithm"] == "TEST":
    #     hash_read_test(depth_list, traj_list, configs, date_str)
    return result


if __name__ == '__main__':
    main('PyCharm')
