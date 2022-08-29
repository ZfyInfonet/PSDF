from data_structure.PhysicalNode import PhysicalNode
from data_structure.VirtualNode import VNF
from data_structure.SFC import SFC
from parameters import Parameters
import numpy as np
import random
random.seed(1)      # 固定随机数


class Initialize:
    PN_list = []
    VNF_list = []
    SFC_list = []
    dis_matrix = [[0, 1, 2, 1, 1, 2, 2, 2, 3, 3],
                  [1, 0, 1, 2, 1, 1, 3, 2, 3, 2],
                  [2, 1, 0, 3, 2, 1, 4, 3, 4, 2],
                  [1, 2, 3, 0, 1, 2, 1, 2, 3, 3],
                  [1, 1, 2, 1, 0, 1, 2, 1, 2, 2],
                  [2, 1, 1, 2, 1, 0, 3, 2, 3, 1],
                  [2, 3, 4, 1, 2, 3, 0, 1, 2, 2],
                  [2, 2, 3, 2, 1, 2, 1, 0, 1, 1],
                  [3, 3, 4, 3, 2, 3, 2, 1, 0, 2],
                  [3, 2, 2, 3, 2, 1, 2, 1, 2, 0]]

    def create_PN_nodelist(self, node_num, cpu_limit):
        self.PN_list = []
        for i in range(node_num):
            self.PN_list.append(
                PhysicalNode(PN_ID = i, cpu = random.randint(Parameters.PN_total_cpu, Parameters.PN_total_cpu),
                             cpu_limit = cpu_limit)
            )

        return self.PN_list

    def get_PN_nodelist(self):
        return self.PN_list

    def create_VNF_list(self, VNF_num):
        self.VNF_list = []
        for i in range(VNF_num):
            self.VNF_list.append(
                VNF(i, Parameters.req_traffic_ratio * 0.01)
            )
        return self.VNF_list

    @staticmethod
    def init_network(PN_nodeList: [], VNF_list: []):
        VNF_num = len(VNF_list)
        PN_node_num = len(PN_nodeList)
        for i in range(VNF_num):
            VNF_list[i].set_PN_ID(i % PN_node_num)
            PN_nodeList[i % PN_node_num].add_vnf(VNF_list[i])
        return PN_nodeList, VNF_list

    @staticmethod
    def debug(PN_nodeList: []):
        PN_nodeNum = len(PN_nodeList)
        print("PN个数:", PN_nodeNum)
        for i in range(PN_nodeNum):  # debug 使用
            print("物理节点", PN_nodeList[i].get_PN_ID(), "中的VNF占用情况为:", PN_nodeList[i].get_vnf_ID_list(), end=', ')
            print("资源占用情况为:",
                  PN_nodeList[i].get_total_cpu() - PN_nodeList[i].get_rest_cpu(), "/", PN_nodeList[i].get_total_cpu(),
                  end='')
            if PN_nodeList[i].check_cpu() == 0:
                print(", 检查出错！")
            else:
                print()

    @staticmethod
    def load_traffic():
        traffic_1 = np.loadtxt("sfc1.csv", delimiter=",", skiprows=0)
        traffic_2 = np.loadtxt("sfc2.csv", delimiter=",", skiprows=0)
        traffic_3 = np.loadtxt("sfc3.csv", delimiter=",", skiprows=0)
        sfc_1 = SFC(SFC_ID = 0, traffic = traffic_1, VNF_ID_list = [])
        sfc_2 = SFC(SFC_ID = 1, traffic = traffic_2, VNF_ID_list = [])
        sfc_3 = SFC(SFC_ID = 2, traffic = traffic_3, VNF_ID_list = [])
        SFC_list = [sfc_1, sfc_2, sfc_3]
        return SFC_list

    @staticmethod
    def assign_VNF_to_SFC(SFC_list: [], VNF_list: []):
        VNF_num = len(VNF_list)
        SFC_num = len(SFC_list)
        for i in range(VNF_num):
            VNF_list[i].set_sfc_ID(i % SFC_num)
            SFC_list[i % SFC_num].add_vnf_ID(VNF_list[i].get_vnf_ID())
        return SFC_list, VNF_list

    def run(self, PN_nodeNum, VNF_num, cpu_limit, random_seed):
        print("初始化开始:")

        random.seed(random_seed)

        self.PN_list = Initialize().create_PN_nodelist(PN_nodeNum, cpu_limit)

        self.VNF_list = Initialize().create_VNF_list(VNF_num)

        self.PN_list, self.VNF_list = Initialize().init_network(self.PN_list, self.VNF_list)

        Initialize().debug(self.PN_list)

        self.SFC_list = Initialize().load_traffic()

        self.SFC_list, self.VNF_list = Initialize().assign_VNF_to_SFC(self.SFC_list, self.VNF_list)
        print("初始化结束.")

        return self.PN_list, self.VNF_list, self.SFC_list, self.dis_matrix
