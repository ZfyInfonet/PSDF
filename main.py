from data_structure.Initialize import Initialize
from data_structure.state import State
from data_structure.DQN import EvalNet, DeepQNetwork
from algorithm import Algorithm
from parameters import Parameters
import pandas as pd
import datetime
from debug import Debug
# from debug import Debug

algorithm = 1
# ****************************************
# *      1. Greedy                       *
# *      2. TrV                          *
# *      3. RM                           *
# ****************************************
random_seed = 1
PN_num = 10
VNF_num = 15
cpu_limit = 0.5
init_time = 800
past_time = 400
n_actions = PN_num ** VNF_num
eval_model = EvalNet(n_actions)
target_model = EvalNet(n_actions)
RL = DeepQNetwork(n_actions,        # 动作个数N^v个
                  PN_num,           # 特征个数为N个, 特征为物理节点的剩余资源比率
                  eval_model=eval_model,
                  target_model=target_model,
                  learning_rate=0.01,
                  reward_decay=0.99,
                  e_greedy=0.9,
                  replace_target_iter=300,
                  memory_size=1000,
                  batch_size=30,
                  e_greedy_increment=None
                  )

if __name__ == '__main__':

    # 初始化物理节点、VNF、SFC列表
    PN_list, VNF_list, SFC_list, topology = Initialize().run(PN_num, VNF_num, cpu_limit, random_seed)
    traffic_list = [SFC_list[0].get_traffic()[init_time],
                    SFC_list[1].get_traffic()[init_time],
                    SFC_list[2].get_traffic()[init_time]]
    state_cur = State(PN_list, VNF_list, traffic_list)
    parameters = Parameters()
    # [alpha1, alpha2, beta1, beta2, gamma1, gamma2, gamma3, down_time, penalty, basic_energy]

    final_list_list = []
    for k in range(9):
        final_list_list.append([])

    state_next = None
    start_time = datetime.datetime.now()
    for time_slot in range(init_time, init_time + past_time):
        print("时刻", time_slot, ":", end ='')
        # print("\tSFC 1, 2, 3的瞬时流量分别为:", traffic_list)
        if time_slot == 800:
            state_cur = State(PN_list, VNF_list, traffic_list)
        else:
            state_cur = state_next
        state_cur = Algorithm.update_cpu_info(state_cur)
        # 获取当前时刻不同SFC流的瞬时流量列表
        # Debug.run(state_cur, SFC_list)
        state_next, output_list = \
            Algorithm.run(algorithm, state_cur, SFC_list, topology, parameters)

        for k in range(9):
            final_list_list[k].append(output_list[k])

        traffic_list = [SFC_list[0].get_traffic()[time_slot + 1],
                        SFC_list[1].get_traffic()[time_slot + 1],
                        SFC_list[2].get_traffic()[time_slot + 1]]
        # 获取当前时刻不同SFC流的瞬时流量列表
        # state_next = state_cur
        state_next.set_traffic_list(traffic_list)

    end_time = datetime.datetime.now()
    test_time = (end_time - start_time).microseconds / (1000000 * past_time) # 每个时隙平均运行时间

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

    if algorithm == 1:
        cost.to_csv('./result/Grd_result_VN' + str(VNF_num) + '.csv', index=False)
    elif algorithm == 2:
        cost.to_csv('./result/Trv_result_VN' + str(VNF_num) + '.csv', index=False)
    else:
        cost.to_csv('./result/RM_result_VN' + str(VNF_num) + '.csv', index=False)
#    Debug().run(PN_list, VNF_list, SFC_list)
