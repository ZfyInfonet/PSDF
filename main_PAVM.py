from data_structure.state import State
from data_structure.Initialize import Initialize
from data_structure.action import Action
# from data_structure.VirtualNode import VNF
from data_structure.PhysicalNode import PhysicalNode
# from data_structure.SFC import SFC
from data_structure.DQN_PAVM import DeepQNetworkPriorityAware, EvalNet
from parameters import Parameters
from algorithm import Algorithm
import pandas as pd
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from debug import Debug

# ****************************************
# *       此主函数只运行分层强化学习          *
# ****************************************
random_seed = 1
episode = Parameters.episode
PN_num = Parameters.PN_num
VNF_num = Parameters.VNF_number
cpu_limit = Parameters.cpu_limit
training_stop_time = Parameters.training_stop_time
target_number = PN_num // 2
physical_node_cpu = Parameters.PN_total_cpu


def get_ob_PAVM(state: State, PN_ID):
    pn_List = state.get_PN_list()
    O_list = []
    for pn in pn_List:
        O_list.append(pn.get_rest_cpu())
    O_list.append(PN_ID)
    O_ = np.array(O_list)
    return O_


def get_node_priority(node_list):
    node_priority = [((physical_node_cpu - pn.get_rest_cpu()) / physical_node_cpu, pn.get_PN_ID()) for pn in node_list]
    return sorted(node_priority, key=lambda x: x[0], reverse=True)  # more occupied cpu, higher priority


def get_vnf_priority(physical_node: PhysicalNode, sfc_list, vnf_list):
    node_vnf_list = physical_node.get_vnf_list()
    sfc_req_cpu_list = []
    for sfc in sfc_list:
        sfc_req = 0
        sfc_vnf_ID_list = sfc.get_vnf_ID_list()
        for sfc_vnf_ID in sfc_vnf_ID_list:
            for vnf in vnf_list:
                if vnf.get_vnf_ID() == sfc_vnf_ID:
                    sfc_req += vnf.get_req_cpu()
                    break
        sfc_req_cpu_list.append(sfc_req)
    m_rc = max(sfc_req_cpu_list)
    vnf_priority = []
    for node_vnf in node_vnf_list:
        sfc_ID = node_vnf.get_sfc_ID()
        p_3 = sfc_req_cpu_list[sfc_ID] / m_rc
        p_5 = (physical_node_cpu - physical_node.get_rest_cpu()) / sfc_req_cpu_list[sfc_ID]
        p = (p_3 + p_5) / 2
        vnf_priority.append((p, node_vnf))
    return sorted(vnf_priority, key=lambda x: x[0], reverse=True)  # more p, higher priority


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


def get_reward(state: State, vnf_ID, sfc, Topology):
    pn_list = state.get_PN_list()
    vnf_list = state.get_VNF_list()
    D = 0
    U = 0
    for vnf in vnf_list:
        if vnf.get_vnf_ID() == vnf_ID:
            D = Algorithm.calculate_link_num(sfc, vnf, Topology, vnf_list)
    for pn in pn_list:
        used_ratio = (physical_node_cpu - pn.get_rest_cpu()) / physical_node_cpu
        if used_ratio > U:
            U = used_ratio

    if D == 0:
        D = 0.001
    rwd = 0.5 / D + 0.5 / U
    return rwd


if __name__ == '__main__':
    for VNF_num_loop in range(6):
        VNF_num = int(VNF_num_loop) + 10   # 0.5, 0.6, 0.7, 0.8, 0.9, 1
        ob_length = PN_num + 1
        action_number = PN_num - 1
        N = PN_num
        V = VNF_num
        eval_model = EvalNet(action_number)
        target_model = EvalNet(action_number)
        RL = DeepQNetworkPriorityAware(action_number,  # N - 1
                                       ob_length,  # feature number
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
        for eps in range(episode):
            total_train_time = 0

            PN_list_init, VNF_list_init, SFC_list, topology = Initialize().run(PN_num, VNF_num, cpu_limit, random_seed)

            traffic_list = [SFC_list[0].get_traffic()[0],
                            SFC_list[1].get_traffic()[0],
                            SFC_list[2].get_traffic()[0]]
            parameters = Parameters()
            # [alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3, down_time, penalty, basic_energy]
            state_next = None
            # ------------------------Training----------------------------
            for time_slot in range(training_stop_time):
                print("时刻", time_slot, ":", end='')
                # print("\tSFC 1, 2, 3的瞬时流量分别为:", traffic_list)
                if time_slot == 0:
                    state_cur = State(PN_list_init, VNF_list_init, traffic_list)
                else:
                    state_cur = state_next
                state_cur = Algorithm.update_cpu_info(state_cur)
                state_now = Algorithm.deepcopy_state(state_cur)

                is_end = 1
                PN_list = state_now.get_PN_list()
                VNF_list = state_now.get_VNF_list()
                for VNF in VNF_list:
                    VNF.set_flag(0)
                sfc_vnf_num_list = [len(x.get_vnf_ID_list()) for x in SFC_list]
                while is_end:
                    congestion_node_list = []
                    for PN in PN_list:
                        if PN.get_overload():
                            congestion_node_list.append(PN)
                    if len(congestion_node_list) == 0:
                        is_end = 0
                        break
                    node_tuple = get_node_priority(congestion_node_list)
                    src_node_tuple = node_tuple[0]
                    src_node_ID = src_node_tuple[1]
                    PN_temp = None
                    for PN in PN_list:
                        if PN.get_PN_ID() == src_node_ID:
                            PN_temp = PN
                            break
                    VNF_priority = get_vnf_priority(PN_temp, SFC_list, VNF_list)

                    if_get = 0
                    for VNF_tuple in VNF_priority:
                        VNF = VNF_tuple[1]
                        if VNF.flag == 1:
                            continue
                        else:
                            if_get = 1
                            sfc_temp = None
                            SFC_ID = 0
                            VNF_ID = VNF.get_vnf_ID()
                            for VNF_2 in VNF_list:
                                if VNF_2.get_vnf_ID() == VNF_ID:
                                    SFC_ID = VNF_2.get_sfc_ID()
                                    sfc_temp = SFC_list[SFC_ID]
                                    VNF_2.set_flag(1)
                                    break
                            sfc_vnf_num_list[SFC_ID] -= 1
                            if sfc_vnf_num_list[SFC_ID] <= 0:
                                is_end = 0
                    if if_get == 0:
                        is_end = 0
                    ob = get_ob_PAVM(state_now, src_node_ID)
                    action = RL.choose_action(ob)
                    PN_ID_list = list(range(N))
                    PN_ID_list.pop(src_node_ID)
                    des_node_ID = PN_ID_list[action]
                    action_list = [-1] * V
                    action_list[VNF_ID] = des_node_ID
                    action_detail = list_to_action(action_list, state_now)
                    state_now = Algorithm.migrate(state_now, action_detail)
                    ob_next = get_ob_PAVM(state_now, des_node_ID)
                    reward = get_reward(state_now, VNF_ID, sfc_temp, topology)
                    is_end_copy = is_end
                    RL.store_transition(ob, action, reward, ob_next, is_end_copy)

                    RL.learn()

                state_next = state_now
                traffic_list = [SFC_list[0].get_traffic()[time_slot + 1],
                                SFC_list[1].get_traffic()[time_slot + 1],
                                SFC_list[2].get_traffic()[time_slot + 1]]
                state_next.set_traffic_list(traffic_list)
                # 获取当前时刻不同SFC流的瞬时流量列表
                true_action = Algorithm.get_action_from_change(state_cur, state_now)
                output_list = Algorithm.total_cost(state_cur, true_action, SFC_list, topology, parameters)
                print("over", 'episode = ', eps, 'total_cost:', output_list[0])
            # ------------------Training Over-------------------------------
            # -------------------Test Begin---------------------------------
            print("Beginning test")

            final_list_list = []
            for k in range(9):
                final_list_list.append([])
            # 初始化物理节点、VNF、SFC列表
            PN_list_init, VNF_list_init, SFC_list, topology = Initialize().run(PN_num, VNF_num, cpu_limit, random_seed)

            traffic_list = [SFC_list[0].get_traffic()[800],
                            SFC_list[1].get_traffic()[800],
                            SFC_list[2].get_traffic()[800]]
            parameters = Parameters()
            # [alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3, down_time, penalty, basic_energy]
            cost_list = []
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

                is_end = 1
                PN_list = state_now.get_PN_list()
                VNF_list = state_now.get_VNF_list()
                for VNF in VNF_list:
                    VNF.set_flag(0)
                sfc_vnf_num_list = [len(x.get_vnf_ID_list()) for x in SFC_list]
                while is_end:
                    congestion_node_list = []
                    for PN in PN_list:
                        if PN.get_overload():
                            congestion_node_list.append(PN)
                    if len(congestion_node_list) == 0:
                        is_end = 0
                        break
                    node_tuple = get_node_priority(congestion_node_list)
                    src_node_tuple = node_tuple[0]
                    src_node_ID = src_node_tuple[1]
                    PN_temp = None
                    for PN in PN_list:
                        if PN.get_PN_ID() == src_node_ID:
                            PN_temp = PN
                            break
                    VNF_priority = get_vnf_priority(PN_temp, SFC_list, VNF_list)

                    if_get = 0
                    for VNF_tuple in VNF_priority:
                        VNF = VNF_tuple[1]
                        if VNF.flag == 1:
                            continue
                        else:
                            if_get = 1
                            sfc_temp = None
                            SFC_ID = 0
                            VNF_ID = VNF.get_vnf_ID()
                            for VNF_2 in VNF_list:
                                if VNF_2.get_vnf_ID() == VNF_ID:
                                    SFC_ID = VNF_2.get_sfc_ID()
                                    sfc_temp = SFC_list[SFC_ID]
                                    VNF_2.set_flag(1)
                                    break
                            sfc_vnf_num_list[SFC_ID] -= 1
                            if sfc_vnf_num_list[SFC_ID] <= 0:
                                is_end = 0
                    if if_get == 0:
                        is_end = 0
                    ob = get_ob_PAVM(state_now, src_node_ID)
                    action = RL.choose_action(ob)
                    PN_ID_list = list(range(N))
                    PN_ID_list.pop(src_node_ID)
                    des_node_ID = PN_ID_list[action]
                    action_list = [-1] * V
                    action_list[VNF_ID] = des_node_ID
                    action_detail = list_to_action(action_list, state_now)
                    state_now = Algorithm.migrate(state_now, action_detail)
                    ob_next = get_ob_PAVM(state_now, des_node_ID)
                    reward = get_reward(state_now, VNF_ID, sfc_temp, topology)
                    is_end_copy = is_end
                    RL.store_transition(ob, action, reward, ob_next, is_end_copy)

                    RL.learn()

                state_next = state_now
                traffic_list = [SFC_list[0].get_traffic()[time_slot + 1],
                                SFC_list[1].get_traffic()[time_slot + 1],
                                SFC_list[2].get_traffic()[time_slot + 1]]
                state_next.set_traffic_list(traffic_list)
                # 获取当前时刻不同SFC流的瞬时流量列表
                true_action = Algorithm.get_action_from_change(state_cur, state_now)
                output_list = Algorithm.total_cost(state_cur, true_action, SFC_list, topology, parameters)
                print("over", 'episode = ', eps, 'total_cost:', output_list[0])

                for k in range(9):
                    final_list_list[k].append(output_list[k])

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
            cost.to_csv('./result/VNF_num_change/' + str(VNF_num) + '/PAVM_result_VN' +
                        str(VNF_num) + '_episode' + str(eps) + '.csv', index=False)

            RL.plot_cost('PAVM_VN' + str(VNF_num) + '_episode' + str(eps))
