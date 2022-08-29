from data_structure.state import State
from data_structure.action import Action
from data_structure.VirtualNode import VNF
from data_structure.PhysicalNode import PhysicalNode
from data_structure.SFC import SFC
from parameters import Parameters
import numpy as np


class Algorithm:

    @staticmethod
    def run(algorithm, state: State, sfc_list, topology, parameters: Parameters):
        output_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if algorithm == 1:
            state_next, output_list = \
                Algorithm.greedy(state, sfc_list, topology, parameters)
        elif algorithm == 2:
            state_next, output_list = \
                Algorithm.Traverse(state, sfc_list, topology, parameters)
        elif algorithm == 3:
            state_next, output_list = \
                Algorithm.RM(state, sfc_list, topology, parameters)
        else:
            print("无效算法, state无变化")
            state_next = state
        return state_next, output_list

    @staticmethod
    def greedy(state, sfc_list, topology, parameters):
        state_mirror = Algorithm.deepcopy_state(state)
        PN_list = state_mirror.get_PN_list()
        VNF_list_all = state_mirror.get_VNF_list()
        traffic_list = state_mirror.get_traffic_list()
        PN_list_sorted = sorted(PN_list, key=lambda x: x.get_rest_cpu())  # 剩余资源从小到大排列
        # 存在过载物理机，且资源占用最少的物理机上存在剩余资源时，执行迁移:
        count = 0
        while PN_list_sorted[0].get_overload() and not PN_list_sorted[-1].get_overload():
            print(PN_list_sorted[0].get_PN_ID(), PN_list_sorted[0].get_rest_cpu(), PN_list_sorted[0].get_overload())
            print(PN_list_sorted[-1].get_PN_ID(), PN_list_sorted[-1].get_rest_cpu(), PN_list_sorted[-1].get_overload())
            VNF_list = PN_list_sorted[0].get_vnf_list()
            VNF_list_sorted = sorted(VNF_list, key=lambda x: x.get_req_cpu(), reverse=False)  # 升序排序
            VNF_smallest = VNF_list_sorted[0]
            if PN_list_sorted[-1].get_rest_cpu() < VNF_smallest.get_req_cpu():
                # print("物理节点溢出！最大剩余资源：", PN_list_sorted[-1].get_rest_cpu(), "VNF需求：", VNF_smallest.get_req_cpu())
                break
            PN_list_sorted[0].del_vnf(VNF_smallest)
            PN_list_sorted[-1].add_vnf(VNF_smallest)
            for vnf in VNF_list_all:  # 更新VNF总列表信息
                if vnf.get_vnf_ID() == VNF_smallest.get_vnf_ID():
                    vnf.set_PN_ID(PN_list_sorted[-1].get_PN_ID())
                    break
            PN_list_sorted = sorted(PN_list_sorted, key=lambda x: x.get_rest_cpu())
            count += 1
            if count == len(VNF_list_all):
                break
        print(count)
        PN_list_sorted = sorted(PN_list, key=lambda x: x.get_PN_ID())  # ID从小到大排列
        state_next = State(PN_list_sorted, VNF_list_all, traffic_list)
        action = Algorithm.get_action_from_change(state, state_next)
        print(action.get_mig_vnf_list())
        print(action.get_mig_des_list())
        output_list = Algorithm.total_cost(state, action, sfc_list, topology, parameters)
        return state_next, output_list

    @staticmethod
    def RM(state, sfc_list, topology, parameters):
        state_mirror = Algorithm.deepcopy_state(state)
        PN_list = state_mirror.get_PN_list()
        N = len(PN_list)
        VNF_list = state_mirror.get_VNF_list()
        V = len(VNF_list)
        for i in range(len(sfc_list)):
            sfc = sfc_list[i]
            VNF_ID_list = sfc.get_vnf_ID_list()
            for j in range(len(VNF_ID_list)):
                VNF_ID = VNF_ID_list[j]
                vnf = None
                for k in range(V):
                    if VNF_list[k].get_vnf_ID() == VNF_ID:
                        vnf = VNF_list[k]
                link_num_ori = Algorithm.calculate_link_num(sfc, vnf, topology, VNF_list)
                PN_ID_list = list(range(N))
                PN_ID_list.remove(vnf.get_PN_ID())  # 禁止原地TP
                for m in range(N - 1):
                    PN_ID = PN_ID_list[m]
                    save_ID = vnf.get_PN_ID()
                    vnf.set_PN_ID(PN_ID)
                    link_num_new = Algorithm.calculate_link_num(sfc, vnf, topology, VNF_list)
                    if link_num_new >= link_num_ori:
                        vnf.set_PN_ID(save_ID)  # 若链路数量增加了, 回档
                    else:
                        if PN_list[PN_ID].get_rest_cpu() < vnf.get_req_cpu():
                            vnf.set_PN_ID(save_ID)
                            continue
                            # print("物理节点溢出！")
                            # return state
                        PN_list[save_ID].del_vnf(vnf)
                        PN_list[PN_ID].add_vnf(vnf)
        state_next = State(PN_list, VNF_list, state.get_traffic_list())
        action = Algorithm.get_action_from_change(state, state_next)
        print(action.get_mig_vnf_list())
        print(action.get_mig_des_list())
        output_list = Algorithm.total_cost(state, action, sfc_list, topology, parameters)
        return state_next, output_list

    @staticmethod
    def Traverse(state, sfc_list, topology, parameters):
        PN_list = state.get_PN_list()
        VNF_list = state.get_VNF_list()
        N = len(PN_list)
        V = len(VNF_list)  # 需要V比特来表示每个VNF是否迁移
        PN_ID_list = list(range(N))
        PN_ID_list_list = []
        for i in range(V):  # 为每个VNF提供一个候选迁移节点列表
            PN_ID_list_temp = PN_ID_list.copy()
            PN_ID_list_temp.remove(VNF_list[i].get_PN_ID())  # 禁止原地TP
            PN_ID_list_list.append(PN_ID_list_temp)
        actions_detail = []  # 共有|N|^|V|个动作，遍历找出total_cost最小的action
        actions_cost = []
        action_cost_min = float('inf')
        action_cost_min_detail = None
        count = 0
        for i in range(2 ** V):  # 4个VNF的需要迁移的情况由2^4个二进制数表示
            action_index = format(i, 'b').zfill(V)  # 二进制字符串,例子: 0101, 第1,3号VNF迁移，2,4号不迁移
            k = bin(i).count('1')  # 需要迁移的VNF个数:k,例: 0101中1的个数:2
            methods = (N - 1) ** k  # 这k个需要迁移的VNF有(N-1)^k个迁移方案
            candidate_list_list = PN_ID_list_list.copy()
            Temp = []
            for p in range(V):
                if action_index[p] == '1':
                    Temp.append(candidate_list_list[p])
            string = [0] * k
            for j in range(methods):
                count += 1
                mig_vnf_list = list(map(int, list(action_index)))  # [0,1,0,1]
                mig_des_list = [-1] * V  # [-1,-1,-1,-1]
                result = []
                if len(string) == 0:  # 不进行任何迁移操作是一种特殊情况
                    # print("啥也不干")
                    action_cur = Action(mig_vnf_list, mig_des_list)
                    actions_detail.append(action_cur)
                    output_list = Algorithm.total_cost(state, action_cur, sfc_list, topology, parameters)
                    action_cost = output_list[0]
                    actions_cost.append(action_cost)
                    action_cost_min = action_cost
                    action_cost_min_detail = action_cur
                    # print(action_cost)
                    continue
                for s in range(k):
                    result.append(Temp[s][string[s]])
                string[0] += 1  # 此乃N-1进制的加法器,最大长度为k,如此即可遍历N-1的k次方
                for q in range(k):
                    if string[q] == N - 1:
                        string[q] = 0
                        if q == k - 1:
                            string = [0] * k
                        else:
                            string[q + 1] += 1
                for m in range(V):
                    if mig_vnf_list[m] == 1:
                        mig_des_list[m] = result[0]
                        result.remove(result[0])
                # print("VNF是否迁移列表:", mig_vnf_list)
                # print("迁移目的地列表:", mig_des_list)
                action_cur = Action(mig_vnf_list, mig_des_list)
                actions_detail.append(action_cur)
                output_list = Algorithm.total_cost(state, action_cur, sfc_list, topology, parameters)
                action_cost = output_list[0]
                actions_cost.append(action_cost)
                if action_cost < action_cost_min:
                    action_cost_min_detail = action_cur
                    action_cost_min = action_cost
                # print(action_cost)
        # 得到了所有可能的动作,智力提高了!
        # 执行了所有动作并选择最优解! CPU的寿命减少了！
        print("total_cost:{:.2f}".format(action_cost_min),
              "action:", action_cost_min_detail.get_mig_vnf_list(),
              action_cost_min_detail.get_mig_des_list())
        # 確定了最小值動作,开始执行迁移
        state_next = Algorithm.migrate(state, action_cost_min_detail)
        output_list = Algorithm.total_cost(state, action_cost_min_detail, sfc_list, topology, parameters)
        return state_next, output_list

    @staticmethod
    def migrate(state: State, action: Action):
        mig_vnf_list = action.get_mig_vnf_list()
        mig_des_list = action.get_mig_des_list()
        PN_list = state.get_PN_list()
        VNF_list = state.get_VNF_list()
        for i in range(len(VNF_list)):
            vnf = VNF_list[i]
            if mig_vnf_list[i] == 1:
                src_node_ID = VNF_list[i].get_PN_ID()
                des_node_ID = mig_des_list[i]
                if PN_list[des_node_ID].get_rest_cpu() < VNF_list[i].get_req_cpu():
                    # print("物理节点溢出！")
                    continue
                PN_list[src_node_ID].del_vnf(vnf)
                PN_list[des_node_ID].add_vnf(vnf)
                vnf.set_PN_ID(des_node_ID)

        return State(PN_list, VNF_list, state.get_traffic_list())

    @staticmethod
    def total_cost(state: State, action: Action, sfc_list, topology, parameters):  # 以下所有操作若改动变量必须恢复
        mig_vnf_list = action.get_mig_vnf_list()
        mig_des_list = action.get_mig_des_list()
        VNF_list = state.get_VNF_list()
        traffic_list = state.get_traffic_list()
        V = len(VNF_list)
        if not len(mig_vnf_list) == len(mig_des_list):
            raise Exception("Action elements do not match!")
        total_reconfiguration_cost = 0
        energy_cost_migration = 0  # 没乘系数前的Energy_cost_mig只是所有迁移vnf的流量和
        for i in range(V):
            if mig_vnf_list[i] == 1:  # 若此VNF需要迁移
                des_node_ID = mig_des_list[i]
                sfc_ID = VNF_list[i].get_sfc_ID()
                reconfiguration_cost = Algorithm.reconfiguration_cost(des_node_ID, VNF_list[i], VNF_list,
                                                                      sfc_list[sfc_ID], traffic_list[sfc_ID],
                                                                      topology, parameters.down_time,
                                                                      parameters.alpha1, parameters.alpha2)
                total_reconfiguration_cost += reconfiguration_cost
                energy_cost_migration += traffic_list[sfc_ID]
        # 得到了迁移相关的cost!下面需要进行迁移预演以确定energy cost & penalty cost
        state_mirror = Algorithm.deepcopy_state(state)
        PN_list_mirror = state_mirror.get_PN_list()
        VNF_list_mirror = state_mirror.get_VNF_list()
        migration_times = 0
        for i in range(V):
            vnf = VNF_list_mirror[i]
            if mig_vnf_list[i] == 1:
                src_node_ID = vnf.get_PN_ID()
                des_node_ID = mig_des_list[i]
                if vnf.get_req_cpu() > PN_list_mirror[des_node_ID].get_rest_cpu():
                    continue
                PN_list_mirror[src_node_ID].del_vnf(vnf)
                PN_list_mirror[des_node_ID].add_vnf(vnf)
                vnf.set_PN_ID(des_node_ID)
                migration_times += 1
        state_mirror = State(PN_list_mirror, VNF_list_mirror, traffic_list)
        phy_list = [1 - pn.get_rest_cpu() / Parameters.PN_total_cpu for pn in PN_list_mirror]
        total_energy_cost = Algorithm.energy_cost(state_mirror, parameters.beta1, parameters.beta2,
                                                  parameters.basic_energy, energy_cost_migration)
        opt_energy_cost = Algorithm.energy_cost(state_mirror, parameters.beta1, parameters.beta2,
                                                parameters.basic_energy, 0)
        total_penalty_cost = Algorithm.penalty_cost(state_mirror, parameters.penalty)

        total_reconfiguration_cost = parameters.gamma1 * total_reconfiguration_cost
        total_energy_cost = parameters.gamma2 * total_energy_cost
        total_penalty_cost = parameters.gamma3 * total_penalty_cost
        opt_energy_cost = parameters.gamma2 * opt_energy_cost
        total_cost = total_reconfiguration_cost + total_energy_cost + total_penalty_cost
        mig_cost = total_cost - total_penalty_cost - opt_energy_cost
        # print("R:", total_reconfiguration_cost, "E:", total_energy_cost, "P:", total_penalty_cost)
        return [total_cost, total_reconfiguration_cost, total_energy_cost, total_penalty_cost, opt_energy_cost,
                mig_cost, np.var(phy_list), Algorithm.sp_var(phy_list), migration_times]

    @staticmethod
    def sp_var(phy_list):
        new_list = []
        for i in phy_list:
            if i - 0 > 0.001:
                new_list.append(i)
        return np.var(new_list)

    @staticmethod
    def reconfiguration_cost(des_node_ID, vnf: VNF, VNF_list, sfc: SFC, vnf_traffic, topology, down_time, alpha1,
                             alpha2):
        src_node_ID = vnf.get_PN_ID()  # 记录vnf迁移源节点
        distance_init = Algorithm.calculate_link_num(sfc, vnf, topology, VNF_list)
        vnf.set_PN_ID(des_node_ID)
        distance_new = Algorithm.calculate_link_num(sfc, vnf, topology, VNF_list)
        vnf.set_PN_ID(src_node_ID)  # 回档
        distance_change = distance_new - distance_init
        return alpha1 * down_time * vnf_traffic + alpha2 * distance_change

    @staticmethod
    def energy_cost(state: State, beta1, beta2, basic_energy, energy_cost_migration):
        energy_cost_running = Algorithm.energy_running_cost(state, beta1, basic_energy)
        return energy_cost_running + beta2 * energy_cost_migration

    @staticmethod
    def penalty_cost(state: State, penalty):
        if Algorithm.overload_nodes_num(state):
            return penalty
        else:
            return 0

    @staticmethod
    def distance(PN_ID1, PN_ID2, topology):
        return topology[PN_ID1][PN_ID2]

    @staticmethod
    def calculate_link_num(sfc: SFC, vnf: VNF, topology, VNF_list):
        VNF_ID_list = sfc.get_vnf_ID_list()
        sfc_len = len(VNF_ID_list)
        vnf_src_ID = VNF_ID_list[0]
        vnf_des_ID = VNF_ID_list[sfc_len - 1]
        vnf_pos_ID = 0
        if len(VNF_ID_list) <= 1:
            return 0
        if vnf.get_vnf_ID() == vnf_src_ID:
            vnf_temp = None
            for i in range(len(VNF_list)):
                if VNF_list[i].get_vnf_ID() == VNF_ID_list[1]:
                    vnf_temp = VNF_list[i]
                    break
            return Algorithm.distance(vnf.get_PN_ID(), vnf_temp.get_PN_ID(), topology)
        elif vnf.get_vnf_ID() == vnf_des_ID:
            vnf_temp = None
            for i in range(len(VNF_list)):
                if VNF_list[i].get_vnf_ID() == VNF_ID_list[sfc_len - 2]:
                    vnf_temp = VNF_list[i]
                    break
            return Algorithm.distance(vnf_temp.get_PN_ID(), vnf.get_PN_ID(), topology)
        else:
            for i in range(sfc_len):
                if VNF_ID_list[i] == vnf.get_vnf_ID():
                    vnf_pos_ID = i
            vnf_temp1 = None
            vnf_temp2 = None
            for j in range(len(VNF_list)):
                if VNF_list[j].get_vnf_ID() == VNF_ID_list[vnf_pos_ID - 1]:
                    vnf_temp1 = VNF_list[j]
                if VNF_list[j].get_vnf_ID() == VNF_ID_list[vnf_pos_ID + 1]:
                    vnf_temp2 = VNF_list[j]
            return (Algorithm.distance(vnf_temp1.get_PN_ID(), vnf.get_PN_ID(), topology) +
                    Algorithm.distance(vnf.get_PN_ID(), vnf_temp2.get_PN_ID(), topology))

    @staticmethod
    def deepcopy_state(state: State):
        PN_list = state.get_PN_list()
        VNF_list = state.get_VNF_list()
        traffic_list = state.get_traffic_list()
        PN_list_new = []
        VNF_list_new = []
        traffic_list_new = traffic_list.copy()
        for i in range(len(VNF_list)):
            vnf = VNF_list[i]
            copy_vnf_ID = vnf.get_vnf_ID()
            copy_vnf_req_cpu = vnf.get_req_cpu()
            copy_PN_ID = vnf.get_PN_ID()
            copy_SFC_ID = vnf.get_sfc_ID()
            vnf_new = VNF(copy_vnf_ID, copy_vnf_req_cpu)
            vnf_new.set_PN_ID(copy_PN_ID)
            vnf_new.set_sfc_ID(copy_SFC_ID)
            VNF_list_new.append(vnf_new)
        for i in range(len(PN_list)):
            PN = PN_list[i]
            new_cpu_limit = PN.get_cpu_limit()
            new_PN_ID = PN.get_PN_ID()
            new_total_cpu = PN.get_total_cpu()
            new_rest_cpu = PN.get_rest_cpu()
            new_overload = PN.get_overload()
            new_MI = PN.get_MI()
            old_VNF_ID_list = PN.get_vnf_ID_list().copy()
            new_VNF_ID_list = []
            for j in range(len(old_VNF_ID_list)):
                new_VNF_ID_list.append(old_VNF_ID_list[j])
            new_VNF_list = []
            for k in range(len(new_VNF_ID_list)):
                vnf_ID = new_VNF_ID_list[k]
                for x in range(len(VNF_list_new)):
                    if VNF_list_new[x].get_vnf_ID() == vnf_ID:
                        new_VNF_list.append(VNF_list_new[x])
            PN_new = PhysicalNode(new_PN_ID, new_total_cpu, new_cpu_limit)
            PN_new.set_rest_cpu(new_rest_cpu)
            PN_new.set_overload(new_overload)
            PN_new.set_MI(new_MI)
            PN_new.set_vnf_list(new_VNF_list)
            PN_new.set_vnf_ID_list(new_VNF_ID_list)
            PN_list_new.append(PN_new)
        state_new = State(PN_list_new, VNF_list_new, traffic_list_new)
        return state_new

    @staticmethod
    def check_link_num(state: State, sfc_list, topology):
        VNF_list = state.get_VNF_list()
        link_num_list = []
        for i in range(len(sfc_list)):
            sfc = sfc_list[i]
            VNF_ID_list = sfc.get_vnf_ID_list()
            if len(VNF_ID_list) <= 1:
                link_num_list.append(0)
                continue
            link_num = 0
            for j in range(len(VNF_ID_list) - 1):
                vnf_src_ID = VNF_ID_list[j]
                vnf_des_ID = VNF_ID_list[j + 1]
                src_PN_ID = 0
                des_PN_ID = 0
                for k in range(len(VNF_list)):
                    if vnf_src_ID == VNF_list[k].get_vnf_ID():
                        src_PN_ID = VNF_list[k].get_PN_ID()
                    if vnf_des_ID == VNF_list[k].get_vnf_ID():
                        des_PN_ID = VNF_list[k].get_PN_ID()
                link_num += Algorithm.distance(src_PN_ID, des_PN_ID, topology)
            link_num_list.append(link_num)
        return link_num_list

    @staticmethod
    def predictor(state: State, traffic_save_list_list):  # traffic_save_list_list为一个3 * 4的矩阵列表
        state_copy = Algorithm.deepcopy_state(state)
        current_traffic_list = state_copy.get_traffic_list()
        new_traffic_list = []
        for index in range(len(traffic_save_list_list)):
            tra_list = traffic_save_list_list[index]
            del tra_list[0]
            tra_list.append(current_traffic_list[index])
            first_order_derivative_list = []
            for index_1 in range(len(tra_list) - 1):
                temp = tra_list[index_1 + 1] - tra_list[index_1]
                first_order_derivative_list.append(temp)
            second_order_derivative_list = []
            for index_2 in range(len(tra_list) - 2):
                temp = first_order_derivative_list[index_2 + 1] - first_order_derivative_list[index_2]
                second_order_derivative_list.append(temp)
            # 在只有两个2阶导的情况下
            new_second_order_derivative = 2 * second_order_derivative_list[1] - second_order_derivative_list[0]
            new_first_order_derivative = first_order_derivative_list[-1] + new_second_order_derivative
            new_traffic = tra_list[-1] + new_first_order_derivative
            if new_traffic < 0:
                new_traffic = 0.01
            new_traffic_list.append(new_traffic)
        state_predicted = State(state_copy.get_PN_list(), state_copy.get_VNF_list(), new_traffic_list)
        state_predicted = Algorithm.update_cpu_info(state_predicted)
        print("流量预测", current_traffic_list, new_traffic_list)
        return state_predicted

    @staticmethod
    def update_traffic_list_list(traffic_list_list, traffic_list_new):
        for i in range(len(traffic_list_new)):
            traffic_list_list[i].pop(0)
            traffic_list_list[i].append(traffic_list_new[i])
        return traffic_list_list

    @staticmethod
    def update_cpu_info(state: State):
        PN_list = state.get_PN_list()
        VNF_list = state.get_VNF_list()
        traffic_list = state.get_traffic_list()
        for vnf in VNF_list:
            sfc_ID = vnf.get_sfc_ID()
            req_cpu = Parameters.req_traffic_ratio * traffic_list[sfc_ID]
            vnf.set_req_cpu(req_cpu)
        for node in PN_list:
            PN_VNF_ID_list = node.get_vnf_ID_list()
            new_VNF_list = []
            for vnf_ID in PN_VNF_ID_list:
                for vnf in VNF_list:
                    if vnf_ID == vnf.get_vnf_ID():
                        new_VNF_list.append(vnf)
                        break
            node.set_vnf_list(new_VNF_list)
            node.update_vnf()
            if not node.check_cpu():
                print("物理节点验证出错", node.check_cpu())
        return State(PN_list, VNF_list, traffic_list)

    @staticmethod
    def get_action_from_change(state: State, state_next: State):
        VNF_list = state.get_VNF_list()
        V = len(VNF_list)
        VNF_list_new = state_next.get_VNF_list()
        mig_vnf_list = [0] * V
        mig_des_list = [-1] * V
        for i in range(V):
            for j in range(V):
                vnf_ID = VNF_list[i].get_vnf_ID()
                if vnf_ID != VNF_list_new[j].get_vnf_ID():
                    continue
                if VNF_list[i].get_PN_ID() != VNF_list_new[j].get_PN_ID():  # 会不会有bug呢
                    mig_vnf_list[i] = 1
                    mig_des_list[i] = VNF_list_new[j].get_PN_ID()

        return Action(mig_vnf_list, mig_des_list)

    @staticmethod
    def overload_nodes_num(state: State):
        PN_list = state.get_PN_list()
        overload_count = 0
        for i in range(len(PN_list)):
            if PN_list[i].get_overload():
                overload_count += 1
        return overload_count

    @staticmethod
    def energy_running_cost(state: State, beta_1, basic_energy):
        PN_list = state.get_PN_list()
        energy_running_list = []
        for i in range(len(PN_list)):
            node = PN_list[i]
            vnf_num = len(node.get_vnf_ID_list())  # 得到该节点上运行的VNF个数
            Energy_cost_run = beta_1 * vnf_num
            if vnf_num != 0:
                Energy_cost_run += basic_energy
            energy_running_list.append(Energy_cost_run)
        return sum(energy_running_list)
        # 返回总运行能耗
