import _thread
import json
import threading
import multiprocessing
import os
import time
import sys
import math
import random
from ucts import uct
from ucts import TopoPlanner
import datetime
from utils.util import mkdir, get_sim_configs, save_reward_hash, get_steps_traj, read_approve_path, \
    read_joint_component_prob
from SimulatorAnalysis import UCT_data_collection

import gc


def merge_act_nodes(dest_act_node, act_node):
    dest_act_node.avg_return_ = dest_act_node.avg_return_ * dest_act_node.num_visits_ + \
                                act_node.avg_return_ * act_node.num_visits_
    dest_act_node.num_visits_ += act_node.num_visits_
    dest_act_node.avg_return_ = dest_act_node.avg_return_ / dest_act_node.num_visits_


def get_action_from_trees(uct_tree_list, uct_tree, tree_num=4):
    contain_action_flag = 0
    uct_tree.root_.act_vect_ = []
    for i in range(tree_num):
        for j in range(len(uct_tree_list[i].node_vect_)):
            contain_action_flag = 0
            for x in range(len(uct_tree.root_.act_vect_)):
                if uct_tree.root_.act_vect_[x].equal(uct_tree_list[i].act_vect_[j]):
                    contain_action_flag = 1
                    break
            if contain_action_flag == 1:
                if j < len(uct_tree_list[i].node_vect_) and uct_tree_list[i].node_vect_[j] is not None:
                    merge_act_nodes(uct_tree.root_.node_vect_[x], uct_tree_list[i].node_vect_[j])
            else:
                if j < len(uct_tree_list[i].node_vect_) and uct_tree_list[i].node_vect_[j] is not None:
                    uct_tree.root_.act_vect_.append(uct_tree_list[i].act_vect_[j].duplicate())
                    uct_tree.root_.node_vect_.append(uct_tree_list[i].node_vect_[j])
    act_node = uct_tree.get_action()
    return act_node


def get_action_from_planners(uct_planner_list, uct_tree, tree_num=4):
    contain_action_flag = 0
    uct_tree.root_.act_vect_ = []
    uct_tree.root_.node_vect_ = []
    for i in range(tree_num):
        for j in range(len(uct_planner_list[i].root_.node_vect_)):
            contain_action_flag = 0
            for x in range(len(uct_tree.root_.act_vect_)):
                if uct_tree.root_.act_vect_[x].equal(uct_planner_list[i].root_.act_vect_[j]):
                    contain_action_flag = 1
                    break
            if contain_action_flag == 1:
                if j < len(uct_planner_list[i].root_.node_vect_) and \
                        uct_planner_list[i].root_.node_vect_[j] is not None:
                    merge_act_nodes(uct_tree.root_.node_vect_[x], uct_planner_list[i].root_.node_vect_[j])
            else:
                if j < len(uct_planner_list[i].root_.node_vect_) and \
                        uct_planner_list[i].root_.node_vect_[j] is not None:
                    uct_tree.root_.act_vect_.append(uct_planner_list[i].root_.act_vect_[j].duplicate())
                    uct_tree.root_.node_vect_.append(uct_planner_list[i].root_.node_vect_[j])
    act_node = uct_tree.get_action()
    return act_node


def get_action_from_trees_vote(uct_planner_list, uct_tree, tree_num=4):
    action_nodes = []
    counts = {}
    for i in range(tree_num):
        action_nodes.append(uct_planner_list[i].get_action())
    for i in range(len(action_nodes)):
        tmp_count = 0
        if counts.get(action_nodes[i]) is None:
            for j in range(len(action_nodes)):
                if action_nodes[j].equal(action_nodes[i]):
                    tmp_count += 1
            counts[action_nodes[i]] = tmp_count
    for action, tmp_count in counts.items():
        if tmp_count == max(counts.values()):
            selected_action = action
    return selected_action


def serial_UCF_test(trajectory, test_number, configs, result_folder, Sim=None, uct_tree_list=None, keep_uct_tree=False):
    if Sim is None:
        Sim = TopoPlanner.TopoGenSimulator
        inside_sim = True
    else:
        inside_sim = False
    # path = './SimulatorAnalysis/database/analytic-expression.json'
    # is_exits = os.path.exists(path)
    # if not is_exits:
    #     UCT_data_collection.key_expression_dict()
    print("finish reading key-expression")
    out_file_folder = 'Results/' + result_folder + '/'
    mkdir(out_file_folder)
    out_file_name = out_file_folder + str(trajectory) + '-result.txt'
    out_round_folder = 'Results/' + result_folder + '/' + str(trajectory)
    mkdir(out_round_folder)
    figure_folder = "figures/" + result_folder + "/"
    mkdir(figure_folder)

    sim_configs = get_sim_configs(configs)
    start_time = datetime.datetime.now()

    fo = open(out_file_name, "w")
    fo.write("max_depth,num_runs,avg_step\n")
    avg_step_list = []
    anal_results = [['efficiency', 'vout', 'reward', 'DC_para', 'query']]
    simu_results = [['efficiency', 'vout', 'reward', 'DC_para', 'query']]

    fo = open(out_file_name, "w")
    avg_step_list = []
    init_nums = 1
    results = []

    for test_idx in range(test_number):
        num_runs = trajectory
        avg_steps = 0
        print()
        fo.write("----------------------------------------------------------------------" + "\n")
        uct_simulators = []
        uct_tree_list = []
        approved_path_freq = {}
        component_condition_prob = {}

        key_expression = UCT_data_collection.read_no_sweep_analytics_result()
        key_sim_effi = UCT_data_collection.read_no_sweep_sim_result()

        steps = 0
        cumulate_plan_time = 0
        r = 0
        tree_size = 0
        sim = Sim(sim_configs, approved_path_freq,
                  component_condition_prob,
                  key_expression, key_sim_effi,
                  None, configs['num_component'])

        uct_tree = uct.UCTPlanner(sim, -1, num_runs, configs["ucb_scalar"], configs["gamma"],
                                  configs["leaf_value"], configs["end_episode_value"],
                                  configs["deterministic"], configs["rave_scalar"], configs["rave_k"],
                                  configs['component_default_policy'], configs['path_default_policy'])
        uct_simulators.clear()
        uct_tree_list.clear()
        for n in range(configs["tree_num"]):
            uct_simulators.append(Sim(sim_configs, approved_path_freq,
                                      component_condition_prob,
                                      key_expression, key_sim_effi,
                                      None, configs['num_component']))

            uct_tree_list.append(
                uct.UCTPlanner(uct_simulators[n], -1, int(num_runs / configs["tree_num"]),
                               configs["ucb_scalar"], configs["gamma"], configs["leaf_value"],
                               configs["end_episode_value"], configs["deterministic"],
                               configs["rave_scalar"], configs["rave_k"], configs['component_default_policy'],
                               configs['path_default_policy']))

        # For fixed commponent type
        init_nodes = []
        # init_nodes = [0, 3, 1]
        for e in init_nodes:
            action = TopoPlanner.TopoGenAction('node', e)
            sim.act(action, False)
        edges = []
        # adj = sim.get_adjust_parameter_set()
        # print(sim.get_adjust_parameter_set())
        # {2: {5}, 5: {2}, 1: {8}, 8: {1}, 0: {3}, 3: {0}, 4: {7}, 7: {4, 6}, 6: {7}})
        # init_nodes = [0, 3, 1]
        # edges = [[0, 3], [1, 8], [2, 5], [-1, -1], [4, 7], [-1, -1], [6, 7], [-1, -1], [-1, -1]]
        # duty_cycle = 0.5
        # edges = [[0, 8], [1, 4], [2, 6], [3, 7], [-1, -1], [5, 7], [-1, -1], [-1, -1], [-1, -1]]
        # # # edges = [[0, 4], [1, 8], [2, 5]]
        # #
        # for edge in edges:
        #     action = TopoPlanner.TopoGenAction('edge', edge)
        #     sim.act(action, False)
        #
        # duty_cycle = 0.5
        # action = TopoPlanner.TopoGenAction('duty_cycle', duty_cycle)
        # sim.act(action, False)
        # sim.get_reward()
        # # sim.get_reward_using_sim()
        # print(sim.reward)
        # return
        # topologies = [sim.get_state()]
        # nets_to_ngspice_files(topologies, configs, configs['num_component'])
        # simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
        # effis = analysis_topologies(configs, len(topologies), configs['num_component'])
        # print("effis of topo:", effis)
        # return

        uct_tree.set_root_node(sim.get_state(), sim.get_actions(), r, sim.is_terminal())
        for n in range(configs["tree_num"]):
            uct_tree_list[n].set_root_node(sim.get_state(), sim.get_actions(), r, sim.is_terminal())
        total_step = configs['num_component'] - 3 - len(init_nodes) + 3 + 2 * (configs['num_component'] - 3 + 1)
        steps_traj = get_steps_traj(total_step * num_runs, total_step, 6, 2, True)
        print(steps_traj)
        while not sim.is_terminal():
            # fo.write(str(steps)+"------step ---------------------------------------------" + "\n")
            plan_start = datetime.datetime.now()

            step_traj = steps_traj[sim.current.step - len(init_nodes) -
                                   (configs['num_component'] - 3 - len(init_nodes))]
            step_traj = int(step_traj / configs["tree_num"])

            for n in range(configs["tree_num"]):
                print("sim.current.step", sim.current.step)
                tree_size_tmp, tree_tmp, depth, node_list = \
                    uct_tree_list[n].plan(step_traj, True, sim.current.step - (len(edges) + len(init_nodes)))

                tree_size += tree_size_tmp
                fo.write("increased tree size:" + str(tree_size_tmp) + "\n")

            _root = uct_tree_list[0].root_
            for child_idx in range(len(_root.node_vect_)):
                child = _root.node_vect_[child_idx]
                print("action ", child_idx, " :", _root.act_vect_[child_idx].type, "on",
                      _root.act_vect_[child_idx].value)
                fo.write("action " + str(child_idx) + " :" + str(_root.act_vect_[child_idx].type) + "on" +
                         str(_root.act_vect_[child_idx].value) + "\n")
                print("action child ", child_idx, " avg_return:", child.avg_return_)
                fo.write("action child " + str(child_idx) + " avg_return:" + str(child.avg_return_) + "\n")
                fo.write("action child " + str(child_idx) + " num_visits_:" + str(child.num_visits_) + "\n")
                child_node = child.state_vect_[0]
                child_state = child_node.state_
                print("state child ", child_idx, " reward:", child_node.reward_)
                fo.write(
                    "state child " + str(child_idx) + "child node reward:" + str(child_node.reward_) + "\n")
                print("state ", child_idx, "ports:", child_state.port_pool)
                fo.write("state child " + str(child_idx) + "ports:" + str(child_state.port_pool) + "\n")
                print("state child", child_idx, "graph:", child_state.graph)
                fo.write("state child " + str(child_idx) + "graph:" + str(child_state.graph) + "\n")
            plan_end_1 = datetime.datetime.now()
            instance_plan_time = (plan_end_1 - plan_start).seconds
            cumulate_plan_time += instance_plan_time

            action = None
            if configs["act_selection"] == "Pmerge":
                action = get_action_from_planners(uct_tree_list, uct_tree, configs["tree_num"])
            elif configs["act_selection"] == "Tmerge":
                action = get_action_from_trees(uct_tree_list, uct_tree, configs["tree_num"])
            elif configs["act_selection"] == "Vote":
                action = get_action_from_trees_vote(uct_tree_list, uct_tree, configs["tree_num"])

            if configs["output"]:
                print("{}-action:".format(steps), end='')
                action.print()
                fo.write("take the action: type:" + str(action.type) +
                         " value: " + str(action.value) + "\n")
                print("{}-state:".format(steps), end='')

            r = sim.act(action)

            for n in range(configs["tree_num"]):
                uct_tree_list[n].update_root_node(action, sim.get_state())
            avg_cumulate_reward = r
            steps += 1
            print("instant reward:", uct_tree.root_.reward_, "cumulate reward: ", avg_cumulate_reward,
                  "planning time:", instance_plan_time, "cumulate planning time:", cumulate_plan_time)
            fo.write("instant reward:" + str(uct_tree.root_.reward_) +
                     "cumulate reward: " + str(avg_cumulate_reward) +
                     "planning time:" + str(instance_plan_time) +
                     "cumulate planning time:" + str(cumulate_plan_time))

        topologies = [sim.get_state()]
        # topologies = [sim.replace_component_name()]
        # nets_to_ngspice_files(topologies, configs, configs['num_component'])
        # simulate_topologies(len(topologies), configs['num_component'], configs["sys_os"])
        # effis = analysis_topologies(configs, len(topologies), configs['num_component'])
        effis = []
        max_result = 0
        max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                 'max_sim_effi': -1, 'max_sim_vout': -500}

        sim.graph_2_reward = uct_simulators[0].graph_2_reward
        sim.current_max = uct_simulators[0].current_max
        sim.no_isom_seen_state_list = uct_simulators[0].no_isom_seen_state_list

        if inside_sim:
            sim.current, sim.reward, sim.current.parameters = sim.get_max_seen()
            max_result = sim.reward
            if configs['reward_method'] == 'analytics':
                effis = [sim.get_effi_info()]
                if len(sim.topk) == 0:
                    max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                             'max_sim_effi': -1, 'max_sim_vout': -500}
                else:
                    max_sim_reward_result = uct_simulators[0].get_tops_sim_info()
                for_simulation_file = 'round' + '_' + str(test_idx)
                results_tmp = uct_simulators[0].generate_topk_simulation_base_dict()
                with open(out_round_folder + '/' + for_simulation_file + '.json', 'w') as f:
                    json.dump(results_tmp, f)
                f.close()
            elif configs['reward_method'] == 'simulator':
                effis = [sim.get_effi_info_using_sim()]
                max_sim_reward_result = {}
                if len(sim.topk) == 0:
                    max_sim_reward_result = {'max_sim_state': 'empty', 'max_sim_reward': 0, 'max_sim_para': -1,
                                             'max_sim_effi': -1, 'max_sim_vout': -500}
                else:
                    max_sim_reward_result['max_sim_state'] = sim.topk[-1][0]
                    max_sim_reward_result['max_sim_reward'] = sim.topk[-1][1]
                    max_sim_reward_result['max_sim_para'] = effis[0]['parameter']
                    max_sim_reward_result['max_sim_effi'] = effis[0]['efficiency']
                    max_sim_reward_result['max_sim_vout'] = effis[0]['output_voltage']
            else:
                effis = None

            # max_sim_reward_result = {'max_sim_effi': 0, 'max_sim_vout': 0, 'max_sim_reward': 0, 'max_sim_para': 0}

            print("effis of topo:", effis)
        print("#####################Game:", "  steps: ", steps, "  average cumulate reward: ",
              avg_cumulate_reward)
        avg_steps += steps

        avg_steps = avg_steps / configs["game_num"]
        fo.write("Final topology of game " + ":\n")
        fo.write("component_pool:" + str(sim.current.component_pool) + "\n")
        fo.write(str(sim.current.parameters) + "\n")
        fo.write("port_pool:" + str(sim.current.port_pool) + "\n")
        fo.write("graph:" + str(sim.current.graph) + "\n")
        fo.write("efficiency:" + str(effis) + "\n")
        fo.write("final reward:" + str(avg_cumulate_reward) + "\n")
        fo.write("step:" + str(avg_steps) + "\n")
        total_query = sim.query_counter
        print(total_query)
        total_hash_query = sim.hash_counter
        for simulator in uct_simulators:
            total_query += simulator.query_counter
            total_hash_query += simulator.hash_counter
        fo.write("query time:" + str(total_query) + " total tree size:" + str(tree_size) + "\n")
        fo.write("hash query time:" + str(total_hash_query) + "\n")
        end_time = datetime.datetime.now()
        final_para_str = sim.current.parameters
        sim.get_state().visualize(
            "result with parameter:" + str(str(final_para_str)) + " ", figure_folder)
        fo.write("end at:" + str(end_time) + "\n")
        fo.write("start at:" + str(start_time) + "\n")
        fo.write("execute time:" + str((end_time - start_time).seconds) + " seconds\n")
        fo.write("result with parameter:" + str(str(final_para_str)) + "\n")
        fo.write("----------------------------------------------------------------------" + "\n")
        avg_step_list.append(avg_steps)

        fo.write("configs:" + str(configs) + "\n")
        fo.write("final rewards:" + str(avg_cumulate_reward) + "\n")

        result = "Traj: " + str(num_runs)
        print(effis)
        if effis != []:
            anal_result = [effis[0]['efficiency'], effis[0]['output_voltage'],
                           max_result, final_para_str, total_query]
            anal_results.append(anal_result)
        simu_result = [max_sim_reward_result['max_sim_effi'], max_sim_reward_result['max_sim_vout'],
                       max_sim_reward_result['max_sim_reward'], max_sim_reward_result['max_sim_para'],
                       total_query]
        simu_results.append(simu_result)
        # UCT_data_collection.save_analytics_result(uct_simulators[0].key_expression)
        UCT_data_collection.save_sta_result(uct_simulators[0].key_sta, 'sta_only_epr.json')
        print('hash counter', uct_simulators[0].hash_counter)
        print('hash length', len(uct_simulators[0].graph_2_reward))
        UCT_data_collection.save_sim_result(uct_simulators[0].key_sim_effi_)

        UCT_data_collection.save_no_sweep_analytics_result(uct_simulators[0].key_expression)
        # TODO save simulation rewards
        UCT_data_collection.save_no_sweep_sim_result(uct_simulators[0].key_sim_effi_)

        # del sim
        # del uct_simulators
        # del uct_tree
        # del uct_tree_list
        # gc.collect()

    print("figures are saved in:" + str(figure_folder) + "\n")
    print("outputs are saved in:" + out_file_name + "\n")

    for result in results:
        fo.write(result + "\n")
    fo.close()

    # save_reward_hash(sim)

    gc.collect()
    return {'sim': sim,
            'time': (end_time - start_time).seconds,
            'query_num': total_query,
            'state_list': uct_tree_list[0].get_all_states(),
            'uct_tree': uct_tree,
            'uct_tree_list': uct_tree_list
            }, anal_results, simu_results
