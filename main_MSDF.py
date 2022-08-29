from data_structure.state import State
from data_structure.Initialize import Initialize
from data_structure.action import Action
from data_structure.DQN import DeepQNetwork, EvalNet
from parameters import Parameters
from algorithm import Algorithm
import datetime
import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from debug import Debug


# ****************************************
# *       此主函数只运行分层强化学习          *
# ****************************************
episode = Parameters.episode
random_seed = 1
PN_num = Parameters.PN_num
VNF_num = Parameters.VNF_number
cpu_limit = Parameters.cpu_limit
training_stop_time = Parameters.training_stop_time
n_actions = PN_num
physical_node_cpu = Parameters.PN_total_cpu


def get_ob(state: State):
    pn_List = state.get_PN_list()
    vnf_List = state.get_VNF_list()
    O_list = []
    for iii in range(len(pn_List)):
        O_list.append(pn_List[iii].get_rest_cpu() / physical_node_cpu)
    for iii in range(len(vnf_List)):
        O_list.append(vnf_List[iii].get_req_cpu() / physical_node_cpu)
    O_ = np.array(O_list)
    return O_


def list_to_action(ac_list, state: State):
    list_length = len(ac_list)
    migration_vnf_list = [0] * V
    migration_des_list = [-1] * V
    vnf_List = state.get_VNF_list()
    for iii in range(list_length):
        vnf = vnf_List[iii]
        if ac_list[iii] != -1 and vnf.get_PN_ID() != ac_list[iii]:
            migration_vnf_list[iii] = 1
            migration_des_list[iii] = ac_list[iii]
    return Action(migration_vnf_list, migration_des_list)


if __name__ == '__main__':
    for cpu_limit_loop in range(6):
        cpu_limit = cpu_limit_loop/10 + 0.5   # 0.5, 0.6, 0.7, 0.8, 0.9, 1
        PN_list_init, VNF_list_init, SFC_list, topology = Initialize().run(PN_num, VNF_num, cpu_limit, random_seed)
        total_train_time = 0
        RL_list = []
        N = PN_num
        V = VNF_num
        for i in range(3):
            vnf_num_in_sfc = len(SFC_list[i].get_vnf_ID_list())
            eval_model = EvalNet(vnf_num_in_sfc * (PN_num - 1) + 1)
            target_model = EvalNet(vnf_num_in_sfc * (PN_num - 1) + 1)
            RL = DeepQNetwork(vnf_num_in_sfc * (PN_num - 1) + 1,  # 动作个数g(N-1) + 1个
                              PN_num + VNF_num,  # 特征个数为N个, 特征为物理节点的剩余资源比率
                              eval_model=eval_model,
                              target_model=target_model,
                              learning_rate=0.01,
                              reward_decay=0.99,
                              e_greedy=0.9,
                              replace_target_iter=100,
                              memory_size=1000,
                              batch_size=30,
                              e_greedy_increment=0.01
                              )
            RL_list.append(RL)
        for eps in range(episode):
            # 初始化物理节点、VNF、SFC列表
            PN_list_init, VNF_list_init, SFC_list, topology = Initialize().run(PN_num, VNF_num, cpu_limit, random_seed)

            traffic_list = [SFC_list[0].get_traffic()[0],
                            SFC_list[1].get_traffic()[0],
                            SFC_list[2].get_traffic()[0]]
            parameters = Parameters()
            # [alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3, down_time, penalty, basic_energy]
            state_next = None
            # ------------------------Training----------------------------

            for time_slot in range(training_stop_time):
                print("时刻", time_slot, ":")
                # print("\tSFC 1, 2, 3的瞬时流量分别为:", traffic_list)
                if time_slot == 0:
                    state_cur = State(PN_list_init, VNF_list_init, traffic_list)
                else:
                    state_cur = state_next
                state_cur = Algorithm.update_cpu_info(state_cur)

                # ---开始操作---
                state_now = Algorithm.deepcopy_state(state_cur)

                for i in range(3):
                    action_list = [-1] * V
                    ob = get_ob(state_now)
                    time_0 = datetime.datetime.now()
                    action_i = RL_list[i].choose_action(ob)
                    vnf_ID_list_in_sfc = SFC_list[i].get_vnf_ID_list()
                    if action_i == len(vnf_ID_list_in_sfc) * (N - 1):
                        action_detail = Action([0] * V, [-1] * V)
                    else:
                        vnf_chosen_index = action_i // (N - 1)
                        des_index = action_i % (N - 1)
                        vnf_chosen_ID = vnf_ID_list_in_sfc[vnf_chosen_index]
                        action_list[vnf_chosen_ID] = des_index
                        action_detail = list_to_action(action_list, state_now)

                    output_list = Algorithm.total_cost(state_now, action_detail, SFC_list, topology, parameters)
                    print('total cost:', output_list[0])
                    state_now = Algorithm.migrate(state_now, action_detail)
                    ob_next = get_ob(state_now)
                    RL_list[i].store_transition(ob, action_i, -output_list[0], ob_next)
                    RL_list[i].learn()

                true_action = Algorithm.get_action_from_change(state_cur, state_now)
                state_next = Algorithm.migrate(state_cur, true_action)
                print("over",  'episode = ', eps)
                traffic_list = [SFC_list[0].get_traffic()[time_slot + 1],
                                SFC_list[1].get_traffic()[time_slot + 1],
                                SFC_list[2].get_traffic()[time_slot + 1]]
                # 获取当前时刻不同SFC流的瞬时流量列表
                # state_next = state_cur
                state_next.set_traffic_list(traffic_list)
            # ------------------Training Over-------------------------------
            # -------------------Test Begin---------------------------------
            print("开始测试")
            # 初始化物理节点、VNF、SFC列表
            PN_list_init, VNF_list_init, SFC_list, topology = Initialize().run(PN_num, VNF_num, cpu_limit, random_seed)

            traffic_list = [SFC_list[0].get_traffic()[800],
                            SFC_list[1].get_traffic()[800],
                            SFC_list[2].get_traffic()[800]]
            parameters = Parameters()

            final_list_list = []
            for k in range(9):
                final_list_list.append([])

            # [alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3, down_time, penalty, basic_energy]

            state_next = None
            for time_slot in range(800, 1200):

                print("时刻", time_slot, ":", end='')
                # print("\tSFC 1, 2, 3的瞬时流量分别为:", traffic_list)
                if time_slot == 800:
                    state_cur = State(PN_list_init, VNF_list_init, traffic_list)
                else:
                    state_cur = state_next
                state_cur = Algorithm.update_cpu_info(state_cur)

                state_now = Algorithm.deepcopy_state(state_cur)

                for i in range(3):
                    action_list = [-1] * V
                    ob = get_ob(state_now)
                    time_0 = datetime.datetime.now()
                    action_i = RL_list[i].choose_action(ob)
                    vnf_ID_list_in_sfc = SFC_list[i].get_vnf_ID_list()
                    if action_i == len(vnf_ID_list_in_sfc) * (N - 1):
                        action_detail = Action([0] * V, [-1] * V)
                    else:
                        vnf_chosen_index = action_i // (N - 1)
                        des_index = action_i % (N - 1)
                        vnf_chosen_ID = vnf_ID_list_in_sfc[vnf_chosen_index]
                        action_list[vnf_chosen_ID] = des_index
                        action_detail = list_to_action(action_list, state_now)

                    output_list = Algorithm.total_cost(state_now, action_detail, SFC_list, topology, parameters)
                    state_now = Algorithm.migrate(state_now, action_detail)
                    ob_next = get_ob(state_now)
                    RL_list[i].store_transition(ob, action_i, -output_list[0], ob_next)
                true_action = Algorithm.get_action_from_change(state_cur, state_now)
                output_list = Algorithm.total_cost(state_cur, true_action, SFC_list, topology, parameters)
                state_next = Algorithm.migrate(state_cur, true_action)
                for k in range(9):
                    final_list_list[k].append(output_list[k])
                print('tot_cost:', output_list[0])

                traffic_list = [SFC_list[0].get_traffic()[time_slot + 1],
                                SFC_list[1].get_traffic()[time_slot + 1],
                                SFC_list[2].get_traffic()[time_slot + 1]]
                # 获取当前时刻不同SFC流的瞬时流量列表
                # state_next = state_cur
                state_next.set_traffic_list(traffic_list)

            dic = {
                'Total cost': final_list_list[0],
                'R cost': final_list_list[1],
                'E cost': final_list_list[2],
                'O cost': final_list_list[3],
                'opt cost': final_list_list[4],
                'mig cost': final_list_list[5],
                'var': final_list_list[6],
                'sp_var': final_list_list[7],
                'mig times': final_list_list[8],
            }
            cost = pd.DataFrame(dic)
            cost.to_csv('./result/cpu_limit_change/cpu_limit_' + str(cpu_limit) + '/MSDF/MSDF_result_VN' +
                        str(VNF_num) + '_episode' + str(eps) + '.csv', index=False)

            RL_list[-1].plot_cost('MSDF_VN' + str(VNF_num) + '_episode' + str(eps))
