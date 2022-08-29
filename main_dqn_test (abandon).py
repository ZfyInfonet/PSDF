from data_structure.state import State
from data_structure.Initialize import Initialize
from data_structure.action import Action
# from data_structure.VirtualNode import VNF
# from data_structure.PhysicalNode import PhysicalNode
# from data_structure.SFC import SFC
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
total_episode = Parameters.episode
episode = 1
random_seed = 1
PN_num = 10
VNF_num = Parameters.VNF_number
cpu_limit = Parameters.cpu_limit
training_stop_time = Parameters.training_stop_time
n_actions = PN_num


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


def get_ob(state: State):
    pn_List = state.get_PN_list()
    vnf_List = state.get_VNF_list()
    O_list = []
    for iii in range(len(pn_List)):
        O_list.append(pn_List[iii].get_rest_cpu() / pn_List[iii].get_total_cpu())
    for iii in range(len(vnf_List)):
        O_list.append(vnf_List[iii].get_req_cpu())
    O_ = np.array(O_list)
    return O_


if __name__ == '__main__':
    while episode <= total_episode:
        agent_list = []
        for i in range(VNF_num):
            eval_model = EvalNet(n_actions)
            target_model = EvalNet(n_actions)
            RL = DeepQNetwork(n_actions,  # 动作个数N^v个
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
            agent_list.append(RL)

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
            # ------------------------Training----------------------------
            for time_slot in range(training_stop_time):
                print("时刻", time_slot, ":", end='')
                # print("\tSFC 1, 2, 3的瞬时流量分别为:", traffic_list)
                if time_slot == 0:
                    state_cur = State(PN_list_init, VNF_list_init, traffic_list)
                else:
                    state_cur = state_next
                state_cur = Algorithm.update_cpu_info(state_cur)
                PN_list = state_cur.get_PN_list()
                N = len(PN_list)
                VNF_list = state_cur.get_VNF_list()
                V = len(VNF_list)
                ob = get_ob(state_cur)
                action_list = [-1] * V
                time_0 = datetime.datetime.now()
                for i in range(VNF_num):
                    action = agent_list[i].choose_action(ob)
                    action_list[i] = action
                action_detail = list_to_action(action_list, state_cur)
                tot_cost, R_cost, E_cost, P_cost = \
                    Algorithm.total_cost(state_cur, action_detail, SFC_list, topology, parameters)

                state_cur = Algorithm.migrate(state_cur, action_detail)
                ob_next = get_ob(state_cur)
                # 训练
                for i in range(V):
                    agent_list[i].store_transition(ob, action_list[i], 1/tot_cost, ob_next)
                    agent_list[i].learn()

                time_1 = datetime.datetime.now()
                total_train_time += (time_1 - time_0).microseconds / (V * 1000000)
                print("total_cost:", tot_cost)

                traffic_list = [SFC_list[0].get_traffic()[time_slot + 1],
                                SFC_list[1].get_traffic()[time_slot + 1],
                                SFC_list[2].get_traffic()[time_slot + 1]]
                # 获取当前时刻不同SFC流的瞬时流量列表
                state_next = state_cur
                state_next.set_traffic_list(traffic_list)
                print("over", 'episode = ', eps)
        # ------------------Training Over-------------------------------
        # -------------------Test Begin---------------------------------
        print("开始测试")

        final_total_cost_list = []
        final_reconfig_cost_list = []
        final_energy_cost_list = []
        final_penalty_cost_list = []
        # 初始化物理节点、VNF、SFC列表
        PN_list_init, VNF_list_init, SFC_list, topology = Initialize().run(PN_num, VNF_num, cpu_limit, random_seed)

        traffic_list = [SFC_list[0].get_traffic()[800],
                        SFC_list[1].get_traffic()[800],
                        SFC_list[2].get_traffic()[800]]
        parameters = Parameters()
        # [alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3, down_time, penalty, basic_energy]
        cost_list = []
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
            PN_list = state_cur.get_PN_list()
            N = len(PN_list)
            VNF_list = state_cur.get_VNF_list()
            V = len(VNF_list)

            ob = get_ob(state_cur)
            action_list = [-1] * V
            time_0 = datetime.datetime.now()
            for i in range(VNF_num):
                action = agent_list[i].choose_action(ob)
                action_list[i] = action
            time_1 = datetime.datetime.now()
            test_time = (time_1 - time_0).microseconds / (V * 1000000)
            action_detail = list_to_action(action_list, state_cur)
            tot_cost, R_cost, E_cost, P_cost = \
                Algorithm.total_cost(state_cur, action_detail, SFC_list, topology, parameters)
            state_cur = Algorithm.migrate(state_cur, action_detail)
            ob_next = get_ob(state_cur)

            state_next = state_cur

            # 测试集上是否在线训练？
            for i in range(V):
                agent_list[i].store_transition(ob, action_list[i], 1 / tot_cost, ob_next)
                # agent_list[i].learn()
            print("total_cost:", tot_cost)

            final_total_cost_list.append(tot_cost)
            final_reconfig_cost_list.append(R_cost)
            final_energy_cost_list.append(E_cost)
            final_penalty_cost_list.append(P_cost)

            traffic_list = [SFC_list[0].get_traffic()[time_slot + 1],
                            SFC_list[1].get_traffic()[time_slot + 1],
                            SFC_list[2].get_traffic()[time_slot + 1]]
            # 获取当前时刻不同SFC流的瞬时流量列表

            state_next.set_traffic_list(traffic_list)

        test_time = test_time / 400  # python的微秒是2022/4/11/21:01.561118，结果是561118这样的格式，而不是1s/1000

        dic = {
            'Total cost': final_total_cost_list,
            'R cost': final_reconfig_cost_list,
            'E cost': final_energy_cost_list,
            'P cost': final_penalty_cost_list
        }
        cost = pd.DataFrame(dic)
        cost.to_csv('./result/DQN_result_PN' + str(PN_num) + '_VN' + str(VNF_num) + '_episode' + str(episode) + '.csv',
                    index=False)

        runtime_list = {
            'Avr run time': [test_time],
            'total train time': [total_train_time]
        }

        time_list = pd.DataFrame(runtime_list)
        time_list.to_csv('./result/runtime_DQN_result_PN' + str(PN_num) + '_VN' + str(VNF_num) + '_episode' +
                         str(episode) + '.csv', index=False)
        if episode == Parameters.episode:
            agent_list[-1].plot_cost('DQN_PN' + str(PN_num) + '_VN' + str(VNF_num) + '_episode' + str(episode))
        episode += 1
