from abc import abstractmethod, ABC

import config

if config.task == 'uct_3_comp':
    from UCT_for_CD_analytics.ucts.TopoPlanner import TopoGenSimulator, calculate_reward, sort_dict_string
elif config.task == 'rs_5_comp':
    from UCT_5_only_epr_standard.ucts.TopoPlanner import TopoGenSimulator, calculate_reward, sort_dict_string
    from UCT_5_only_epr_standard.SimulatorAnalysis.gen_topo import key_circuit_from_lists, convert_to_netlist

from topo_data_util.topoGraph import TopoGraph
from topo_data_util.utils.graphUtils import nodes_and_edges_to_adjacency_matrix


class SurrogateRewardTopologySim(TopoGenSimulator, ABC):
    def __init__(self, debug, *args):
        self.debug = debug
        # for fair comparison with simulator, create a hash table here
        self.surrogate_hash_table = {}

        super().__init__(*args)

    def find_paths(self):
        """
        Useful for GP and transformer based surrogate model
        Return the list of paths in the current state
        e.g. ['VIN - inductor - VOUT', ...]
        """
        node_list, edge_list = self.get_state().get_nodes_and_edges()

        adjacency_matrix = nodes_and_edges_to_adjacency_matrix(node_list, edge_list)

        # convert graph to paths, and find embedding
        topo = TopoGraph(adj_matrix=adjacency_matrix, node_list=node_list)
        return topo.find_end_points_paths_as_str()

    def get_topo_key(self, state=None):
        """
        the key of topology used by hash table

        :return:  the key representation of the state (self.current if state == None)
        """
        if state is None:
            state = self.get_state()

        if config.task == 'uct_3_comp':
            topo_key = sort_dict_string(state.graph)
        elif config.task == 'rs_5_comp':
            list_of_node, list_of_edge, netlist, joint_list = convert_to_netlist(state.graph,
                                                                                 state.component_pool,
                                                                                 state.port_pool,
                                                                                 state.parent,
                                                                                 state.comp2port_mapping)
            topo_key = key_circuit_from_lists(list_of_edge, list_of_node, netlist)
        else:
            raise Exception()

        return topo_key

    def get_reward(self, state=None):
        """
        Use surrogate reward function
        imp-wise, not sure why keeping a reward attribute
        """
        if state is not None:
            self.set_state(state)

        if not self.current.graph_is_valid():
            self.reward = 0
            return self.reward

        topo_key = self.get_topo_key()
        if topo_key in self.surrogate_hash_table:
            self.hash_counter += 1

            return self.surrogate_hash_table[topo_key]
        else:
            self.query_counter += 1

            # eff = self.get_surrogate_eff(self.get_state())
            # vout = self.get_surrogate_vout(self.get_state())
            eff, vout, reward, parameter = self.get_surrogate_reward(self.get_state())

            # an object for computing reward
            # eff_obj = {'efficiency': eff,
            #            'output_voltage': vout}
            self.reward = reward

            if self.debug:
                print('estimated reward {}, eff {}, vout {}'.format(self.reward, eff, vout))
                print('true performance {}'.format(self.get_true_performance()))

            self.surrogate_hash_table[topo_key] = self.reward

        return self.reward

    @abstractmethod
    def get_surrogate_eff(self, state):
        """
        return the eff prediction of state, and of self.get_state() if None
        """
        pass

    @abstractmethod
    def get_surrogate_vout(self, state):
        """
        return the vout prediction of state, and of self.get_state() if None
        """
        pass

    def get_surrogate_reward(self, state):
        """
        return the vout prediction of state, and of self.get_state() if None
        """
        pass
    def save_dataset_to_file(self, dataset):
        """
        return the vout prediction of state, and of self.get_state() if None
        """
        pass

    def get_true_performance(self, state=None):
        """
        :return: [reward, eff, vout]
        """
        if state is not None:
            self.set_state(state)

        hash = self.get_topo_key()

        # if not in hash table, call ngspice
        if hash not in self.graph_2_reward.keys():
            super().get_reward()

        if hash in self.graph_2_reward.keys():
            if config.task == 'uct_3_comp':
                return self.graph_2_reward[hash]
            elif config.task == 'rs_5_comp':
                para, eff, vout = self.graph_2_reward[hash]

                eff_obj = {'efficiency': eff,
                           'output_voltage': vout}
                reward = calculate_reward(eff_obj)

                return [reward, eff, vout]
        else:
            return [0, 0, 0]

    def get_true_reward(self, state=None):
        return self.get_true_performance(state)[0]
