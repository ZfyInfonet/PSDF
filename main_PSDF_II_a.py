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
import threading


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result  # 如果子线程不使用join方法，此处可能会报没有self.result的错误
        except RuntimeError:
            return None


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# from debug import Debug


# ****************************************
# *       此主函数只运行一种算法             *
# ****************************************

random_seed = 1
episode = Parameters.episode
PN_num = Parameters.PN_num
VNF_num = Parameters.VNF_number
cpu_limit = Parameters.cpu_limit
training_stop_time = Parameters.training_stop_time
target_number = PN_num // 2
physical_node_cpu = Parameters.PN_total_cpu

available_cpu_total = physical_node_cpu * cpu_limit * PN_num
available_cpu_per_node = physical_node_cpu * cpu_limit


def calculate_variance(ob):
    ob_temp = ob.copy()
    np.delete(ob_temp, [-1])
    pN_list = ob_temp
    E_count = 0
    sigma_N = PN_num
    for element in pN_list:
        E_count += element
    E = E_count / sigma_N
    sigma = 0
    for element in pN_list:
        sigma += (element - E) ** 2
    sigma_pow_2 = sigma / sigma_N
    return sigma_pow_2


def get_ob_for_tot(state: State, VNF_ID):
    pn_List = state.get_PN_list()
    vnf_List = state.get_VNF_list()
    vnf_temp = None
    for vnf in vnf_List:
        if vnf.get_vnf_ID() == VNF_ID:
            vnf_temp = vnf
            break
    O_list = []
    vnf_request_ratio = vnf_temp.get_req_cpu() / physical_node_cpu
    for pn in pn_List:
        if pn.get_PN_ID() == vnf_temp.get_PN_ID():
            O_list.append(1 - pn.get_rest_cpu() / physical_node_cpu - vnf_request_ratio)
        else:
            O_list.append(1 - pn.get_rest_cpu() / physical_node_cpu)

    O_list.append(vnf_request_ratio)
    O_ = np.array(O_list)
    return O_, vnf_request_ratio


def get_ob_for_opt(state: State, VNF_ID):
    pn_List = state.get_PN_list()
    ordered_pn_List = sorted(pn_List, key=lambda x: x.get_rest_cpu(), reverse=False)
    vnf_List = state.get_VNF_list()
    vnf_temp = None
    for vnf in vnf_List:
        if vnf.get_vnf_ID() == VNF_ID:
            vnf_temp = vnf
            break
    O_list = []
    vnf_request_ratio = vnf_temp.get_req_cpu() / physical_node_cpu
    count = 0
    for pn in ordered_pn_List:
        if pn.get_PN_ID() == vnf_temp.get_PN_ID():
            O_list.append(1 - pn.get_rest_cpu() / physical_node_cpu - vnf_request_ratio)
        else:
            O_list.append(1 - pn.get_rest_cpu() / physical_node_cpu)
        count += 1
        if count == target_number:
            break
    O_list.append(vnf_request_ratio)
    O_ = np.array(O_list)
    return O_, vnf_request_ratio


def get_ob_for_ovr(state: State, VNF_ID):
    pn_List = state.get_PN_list()
    ordered_pn_List = sorted(pn_List, key=lambda x: x.get_rest_cpu(), reverse=True)
    vnf_List = state.get_VNF_list()
    vnf_temp = None
    for vnf in vnf_List:
        if vnf.get_vnf_ID() == VNF_ID:
            vnf_temp = vnf
            break
    O_list = []
    vnf_request_ratio = vnf_temp.get_req_cpu() / physical_node_cpu
    count = 0
    for pn in ordered_pn_List:
        if pn.get_PN_ID() == vnf_temp.get_PN_ID():
            O_list.append(1 - pn.get_rest_cpu() / physical_node_cpu - vnf_request_ratio)
        else:
            O_list.append(1 - pn.get_rest_cpu() / physical_node_cpu)
        count += 1
        if count == target_number:
            break
    O_list.append(vnf_request_ratio)
    O_ = np.array(O_list)
    return O_, vnf_request_ratio


def get_ob_next_for_tot(ob_for_tot, action_total, request_ratio: float):
    ob_for_tot[action_total] += request_ratio
    return ob_for_tot


def get_ob_next_for_opt(ob_for_opt, action_operation, request_ratio: float):
    ob_for_opt[action_operation] += request_ratio
    return ob_for_opt


def get_ob_next_for_ovr(ob_for_ovr, action_over, request_ratio: float):
    ob_for_ovr[action_over] += request_ratio
    return ob_for_ovr


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


def change_VNF_priority(state: State):
    state_copy = Algorithm.deepcopy_state(state)
    vnf_list = state_copy.get_VNF_list()
    vnf_list_ordered = sorted(vnf_list, key=lambda x: x.get_req_cpu(), reverse=False)
    VNF_priority = [vnf.get_vnf_ID() for vnf in vnf_list_ordered]
    return VNF_priority


if __name__ == '__main__':

    N = PN_num
    V = VNF_num
    ob_length_tot = 11
    ob_length_opt = 6
    ob_length_ovr = 6
    action_number_tot = 10
    action_number_opt = 5
    action_number_ovr = 5
    eval_model = EvalNet(action_number_opt)
    target_model = EvalNet(action_number_opt)
    opt_RL = DeepQNetwork(action_number_opt,  # 动作个数N个
                          ob_length_opt,  # 特征个数为N个, 特征为物理节点的剩余资源比率
                          eval_model=eval_model,
                          target_model=target_model,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=100,
                          memory_size=1000,
                          batch_size=30,
                          e_greedy_increment=0.01
                          )

    eval_model = EvalNet(action_number_ovr)
    target_model = EvalNet(action_number_ovr)
    ovr_RL = DeepQNetwork(action_number_ovr,  # 动作个数N个
                          ob_length_ovr,  # 特征个数为N个, 特征为物理节点的剩余资源比率
                          eval_model=eval_model,
                          target_model=target_model,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=100,
                          memory_size=1000,
                          batch_size=30,
                          e_greedy_increment=0.01
                          )

    eval_model = EvalNet(action_number_tot)
    target_model = EvalNet(action_number_tot)
    tot_RL = DeepQNetwork(action_number_tot,  # 动作个数N个
                          ob_length_tot,  # 特征个数为N个, 特征为物理节点的剩余资源比率
                          eval_model=eval_model,
                          target_model=target_model,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=100,
                          memory_size=1000,
                          batch_size=30,
                          e_greedy_increment=0.01
                          )

    total_train_time = 0
    for eps in range(episode):
        # 初始化物理节点、VNF、SFC列表
        PN_list_init, VNF_list_init, SFC_list, topology = Initialize().run(PN_num, VNF_num, cpu_limit, random_seed)

        traffic_list = [SFC_list[0].get_traffic()[0],
                        SFC_list[1].get_traffic()[0],
                        SFC_list[2].get_traffic()[0]]
        parameters = Parameters()
        # [alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3, down_time, penalty, basic_energy]
        state_next = None
        # traffic_save_list_list = [[traffic_list[0]] * 4, [traffic_list[1]] * 4, [traffic_list[2]] * 4]
        # ------------------------Training----------------------------

        for time_slot in range(training_stop_time):
            print("TIME SLOT:", time_slot, "begin:")
            # print("\tSFC 1, 2, 3的瞬时流量分别为:", traffic_list)
            if time_slot == 0:
                state_cur = State(PN_list_init, VNF_list_init, traffic_list)
            else:
                state_cur = state_next

            state_cur = Algorithm.update_cpu_info(state_cur)
            # traffic_save_list_list = Algorithm.update_traffic_list_list(traffic_save_list_list, traffic_list)
            # state_predict = Algorithm.predictor(state_cur, traffic_save_list_list)

            # -------------------
            Priority = change_VNF_priority(state_cur)
            state_now = Algorithm.deepcopy_state(state_cur)
            for i in Priority:
                print('\t----------------------Begin algorithm for VNF', i, '----------------------------')
                action_list_tot = [-1] * V
                action_list_opt = [-1] * V
                action_list_ovr = [-1] * V
                action_list_0 = [-1] * V

                VNF_list_now = state_now.get_VNF_list()
                PN_list_now = state_now.get_PN_list()
                ordered_PN_list_now = sorted(PN_list_now, key=lambda x: x.get_rest_cpu(), reverse=True)

                overload_flag = False  # Note that we set a special cpu_limit belongs to agents.
                for PN in PN_list_now:
                    if (1 - PN.get_rest_cpu() / physical_node_cpu) >= cpu_limit ** 2:
                        overload_flag = True
                        break

                # -----------Parallel Execute------------
                time_0 = datetime.datetime.now()

                ob_tot, vnf_req_ratio_tot = get_ob_for_tot(state_now, i)
                ob_opt, vnf_req_ratio_opt = get_ob_for_opt(state_now, i)
                ob_ovr, vnf_req_ratio_ovr = get_ob_for_ovr(state_now, i)

                action_opt_ordered = opt_RL.choose_action(ob_opt)
                action_ovr_ordered = ovr_RL.choose_action(ob_ovr)
                opt_target_PN = ordered_PN_list_now[-1 * action_opt_ordered - 1]
                ovr_target_PN = ordered_PN_list_now[action_ovr_ordered]

                action_tot = tot_RL.choose_action(ob_tot)
                action_opt = opt_target_PN.get_PN_ID()
                action_ovr = ovr_target_PN.get_PN_ID()
                action_0 = VNF_list_now[i].get_PN_ID()

                time_1 = datetime.datetime.now()
                total_train_time += (time_1 - time_0).microseconds / 4000000

                action_list_tot[i] = action_tot
                action_list_opt[i] = action_opt
                action_list_ovr[i] = action_ovr

                action_tot_detail = list_to_action(action_list_tot, state_now)
                action_opt_detail = list_to_action(action_list_opt, state_now)
                action_ovr_detail = list_to_action(action_list_ovr, state_now)
                action_0_detail = list_to_action(action_list_0, state_now)

                final_actions = [action_tot, action_opt, action_ovr, action_0]
                final_actions_detail = [action_tot_detail, action_opt_detail, action_ovr_detail, action_0_detail]

                time_2 = datetime.datetime.now()
                # calculate costs for comparing
                t1 = MyThread(Algorithm.total_cost, (state_now, action_tot_detail, SFC_list, topology, parameters))
                t1.start()
                t2 = MyThread(Algorithm.total_cost, (state_now, action_opt_detail, SFC_list, topology, parameters))
                t2.start()
                t3 = MyThread(Algorithm.total_cost, (state_now, action_ovr_detail, SFC_list, topology, parameters))
                t3.start()
                t4 = MyThread(Algorithm.total_cost, (state_now, action_0_detail, SFC_list, topology, parameters))
                t4.start()

                t1.join()
                t2.join()
                t3.join()
                t4.join()

                result_1 = t1.get_result()  # [tot_cost, R_cost, E_cost, O_cost, opt_cost, mig_cost]
                result_2 = t2.get_result()
                result_3 = t3.get_result()
                result_4 = t4.get_result()

                list_1 = [result_1[0], result_1[4]]
                list_2 = [result_2[0], result_2[4]]
                list_3 = [result_3[0], result_3[4]]
                list_4 = [result_4[0], result_4[4]]

                cost_list = [result_1[0], result_2[0], result_3[0], result_4[0]]
                min_cost_index = cost_list.index(min(cost_list))
                print('\tTotal cost list:', cost_list)

                if overload_flag and min(cost_list) > Parameters.penalty + Parameters.beta1 * VNF_num:
                    print("\tOverLoad and other agent can't work.")
                    final_action = action_ovr
                    final_action_detail = action_ovr_detail
                    print("\tChoose [Overload agent]'s action:", action_ovr + 1)
                else:
                    if overload_flag:
                        print("\tOverload!")
                    else:
                        print("\tNo OverLoad.")
                    final_action = final_actions[min_cost_index]
                    final_action_detail = final_actions_detail[min_cost_index]
                    if min_cost_index == 0:
                        print("\tChoose [Total agent]'s action:", action_tot + 1)
                    elif min_cost_index == 1:
                        print("\tChoose [Operation agent]'s action:", action_opt + 1)
                    elif min_cost_index == 2:
                        print("\tChoose [Overload agent]'s action:", action_ovr + 1)
                    else:
                        print("\tChoose [Never-Mig agent]'s action:", action_0 + 1)

                # --------------get state copy for DQN training--------------
                Threads = []
                for k in range(3):
                    t = MyThread(Algorithm.deepcopy_state, (state_now,))
                    t.start()
                    Threads.append(t)
                for thread in Threads:
                    thread.join()

                state_copy_1 = Threads[0].get_result()
                state_copy_2 = Threads[1].get_result()
                state_copy_3 = Threads[2].get_result()
                # --------------get state copy for DQN training--------------

                # calculate variance
                variance_now = calculate_variance(ob_ovr)
                # --------------Get reward for different DQN agent--------------
                state_now = Algorithm.migrate(state_now, final_action_detail)
                state_copy_1_next = Algorithm.migrate(state_copy_1, final_actions_detail[0])
                state_copy_2_next = Algorithm.migrate(state_copy_2, final_actions_detail[1])
                state_copy_3_next = Algorithm.migrate(state_copy_3, final_actions_detail[2])

                ob_next_1 = get_ob_next_for_tot(np.copy(ob_tot), action_tot, vnf_req_ratio_tot)
                ob_next_2 = get_ob_next_for_opt(np.copy(ob_opt), action_opt_ordered, vnf_req_ratio_opt)
                ob_next_3 = get_ob_next_for_ovr(np.copy(ob_ovr), action_ovr_ordered, vnf_req_ratio_ovr)
                # calculate variance
                variance_next = calculate_variance(ob_next_3)
                reward_for_overload = variance_now - variance_next
                # --------------Get reward for different DQN agent--------------

                # --------------------Study------------------------
                tot_RL.store_transition(ob_tot, action_tot, 1 / list_1[0], ob_next_1)
                opt_RL.store_transition(ob_opt, action_opt_ordered, 1 / list_2[1], ob_next_2)
                ovr_RL.store_transition(ob_ovr, action_ovr_ordered, reward_for_overload, ob_next_3)

                tot_RL.learn()
                opt_RL.learn()
                ovr_RL.learn()
                time_3 = datetime.datetime.now()
                total_train_time += (time_3 - time_2).microseconds / 3000000
                # --------------------Study------------------------
                print('\tob_tot:', *ob_tot)

                print_info = ['\tob_mig:']
                for PN in PN_list_now:
                    print_info.append(1 - PN.get_rest_cpu() / physical_node_cpu)
                print_info.append(', agent chose the')
                print_info.append(final_action + 1)
                print_info.append('th physical node.')
                print(*print_info)

            if Algorithm.overload_nodes_num(state_now) > 0:    # All actions can't handle overload.
                action_list_0 = [-1] * V
                true_action = list_to_action(action_list_0, state_now)
            else:
                true_action = Algorithm.get_action_from_change(state_cur, state_now)
            state_next = Algorithm.migrate(state_cur, true_action)
            print("TIME SLOT:", time_slot, "over", 'episode = ', eps)
            traffic_list = [SFC_list[0].get_traffic()[time_slot + 1],
                            SFC_list[1].get_traffic()[time_slot + 1],
                            SFC_list[2].get_traffic()[time_slot + 1]]
            # get traffic list, 获取当前时刻不同SFC流的瞬时流量列表
            state_next.set_traffic_list(traffic_list)

        # ------------------Training Over-------------------------------
        # -------------------Test Begin---------------------------------
        print("Beginning Test.")
        # 初始化物理节点、VNF、SFC列表
        PN_list_init, VNF_list_init, SFC_list, topology = Initialize().run(PN_num, VNF_num, cpu_limit, random_seed)

        traffic_list = [SFC_list[0].get_traffic()[800],
                        SFC_list[1].get_traffic()[800],
                        SFC_list[2].get_traffic()[800]]
        parameters = Parameters()
        # [alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3, down_time, penalty, basic_energy]
        # traffic_save_list_list = [[traffic_list[0]] * 4, [traffic_list[1]] * 4, [traffic_list[2]] * 4]
        final_list_list = []
        for k in range(9):
            final_list_list.append([])
        state_next = None
        test_time = 0
        for time_slot in range(800, 1200):

            print("时刻", time_slot, ":", end='')
            # print("\tSFC 1, 2, 3的瞬时流量分别为:", traffic_list)
            if time_slot == 800:
                state_cur = State(PN_list_init, VNF_list_init, traffic_list)
            else:
                state_cur = state_next

            state_cur = Algorithm.update_cpu_info(state_cur)
            # traffic_save_list_list = Algorithm.update_traffic_list_list(traffic_save_list_list, traffic_list)
            # state_predict = Algorithm.predictor(state_cur, traffic_save_list_list)
            Priority = change_VNF_priority(state_cur)
            state_now = Algorithm.deepcopy_state(state_cur)
            for i in Priority:
                action_list_tot = [-1] * V
                action_list_opt = [-1] * V
                action_list_ovr = [-1] * V
                action_list_0 = [-1] * V

                PN_list_now = state_now.get_PN_list()
                ordered_PN_list_now = sorted(PN_list_now, key=lambda x: x.get_rest_cpu(), reverse=True)
                VNF_list_now = state_now.get_VNF_list()
                overload_flag = False
                for PN in PN_list_now:
                    if (1 - PN.get_rest_cpu() / physical_node_cpu) >= cpu_limit ** 2:
                        overload_flag = True
                        break

                # -----------Parallel Execute------------
                time_0 = datetime.datetime.now()

                ob_tot, vnf_req_ratio_tot = get_ob_for_tot(state_now, i)
                ob_opt, vnf_req_ratio_opt = get_ob_for_opt(state_now, i)
                ob_ovr, vnf_req_ratio_ovr = get_ob_for_ovr(state_now, i)

                action_opt_ordered = opt_RL.choose_action(ob_opt)
                action_ovr_ordered = ovr_RL.choose_action(ob_ovr)
                opt_target_PN = ordered_PN_list_now[-1 * action_opt_ordered - 1]
                ovr_target_PN = ordered_PN_list_now[action_ovr_ordered]

                action_tot = tot_RL.choose_action(ob_tot)
                action_opt = opt_target_PN.get_PN_ID()
                action_ovr = ovr_target_PN.get_PN_ID()
                action_0 = VNF_list_now[i].get_PN_ID()

                time_1 = datetime.datetime.now()
                total_train_time += (time_1 - time_0).microseconds / 4000000

                action_list_tot[i] = action_tot
                action_list_opt[i] = action_opt
                action_list_ovr[i] = action_ovr

                action_tot_detail = list_to_action(action_list_tot, state_now)
                action_opt_detail = list_to_action(action_list_opt, state_now)
                action_ovr_detail = list_to_action(action_list_ovr, state_now)
                action_0_detail = list_to_action(action_list_0, state_now)

                final_actions = [action_tot, action_opt, action_ovr, action_0]
                final_actions_detail = [action_tot_detail, action_opt_detail, action_ovr_detail, action_0_detail]

                # ----并行执行不同动作-------
                time_2 = datetime.datetime.now()
                # calculate costs for comparing
                t1 = MyThread(Algorithm.total_cost, (state_now, action_tot_detail, SFC_list, topology, parameters))
                t1.start()
                t2 = MyThread(Algorithm.total_cost, (state_now, action_opt_detail, SFC_list, topology, parameters))
                t2.start()
                t3 = MyThread(Algorithm.total_cost, (state_now, action_ovr_detail, SFC_list, topology, parameters))
                t3.start()
                t4 = MyThread(Algorithm.total_cost, (state_now, action_0_detail, SFC_list, topology, parameters))
                t4.start()

                t1.join()
                t2.join()
                t3.join()
                t4.join()

                result_1 = t1.get_result()  # [tot_cost, R_cost, E_cost, O_cost, opt_cost, mig_cost]
                result_2 = t2.get_result()
                result_3 = t3.get_result()
                result_4 = t4.get_result()

                list_1 = [result_1[0], result_1[4]]
                list_2 = [result_2[0], result_2[4]]
                list_3 = [result_3[0], result_3[4]]
                list_4 = [result_4[0], result_4[4]]

                cost_list = [result_1[0], result_2[0], result_3[0], result_4[0]]
                min_cost_index = cost_list.index(min(cost_list))
                print('\tTotal cost list:', cost_list)

                if overload_flag and min(cost_list) > Parameters.penalty + Parameters.beta1 * VNF_num:
                    print("\tOverLoad and other agent can't work.")
                    final_action = action_ovr
                    final_action_detail = action_ovr_detail
                    print("\tChoose [Overload agent]'s action:", action_ovr + 1)
                else:
                    if overload_flag:
                        print("\tOverload!")
                    else:
                        print("\tNo OverLoad.")
                    final_action = final_actions[min_cost_index]
                    final_action_detail = final_actions_detail[min_cost_index]
                    if min_cost_index == 0:
                        print("\tChoose [Total agent]'s action:", action_tot + 1)
                    elif min_cost_index == 1:
                        print("\tChoose [Operation agent]'s action:", action_opt + 1)
                    elif min_cost_index == 2:
                        print("\tChoose [Overload agent]'s action:", action_ovr + 1)
                    else:
                        print("\tChoose [Never-Mig agent]'s action:", action_0 + 1)

                time_3 = datetime.datetime.now()
                test_time += (time_3 - time_2).microseconds / 1000000

                state_now = Algorithm.migrate(state_now, final_action_detail)
            if Algorithm.overload_nodes_num(state_now) > 0:    # All actions can't handle overload.
                action_list_0 = [-1] * V
                true_action = list_to_action(action_list_0, state_now)
            else:
                true_action = Algorithm.get_action_from_change(state_cur, state_now)

            output_list = Algorithm.total_cost(state_cur, true_action, SFC_list, topology, parameters)
            state_next = Algorithm.migrate(state_cur, true_action)

            print("\tTotal cost of the combined action:", output_list[0])
            for k in range(9):
                final_list_list[k].append(output_list[k])
            traffic_list = [SFC_list[0].get_traffic()[time_slot + 1],
                            SFC_list[1].get_traffic()[time_slot + 1],
                            SFC_list[2].get_traffic()[time_slot + 1]]
            # 获取当前时刻不同SFC流的瞬时流量列表
            # state_next = state_cur
            state_next.set_traffic_list(traffic_list)
        test_time = test_time / 400  # python的微秒是2022/4/11/21:01.561118，microsecond结果是561118这样的格式

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
        cost.to_csv('./result/PSDF_II_a_result_PN' + str(PN_num) + '_VN' +
                    str(VNF_num) + '_episode' + str(eps) + '.csv', index=False)

        ovr_RL.plot_cost('PSDF_PN' + str(PN_num) + '_VN' + str(VNF_num) + '_episode' + str(eps))
