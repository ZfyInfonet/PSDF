from data_structure.VirtualNode import VNF


class PhysicalNode:

    def __init__(self, PN_ID, cpu, cpu_limit):

        self.cpu_limit = cpu_limit
        # cpu使用阈值
        self.PN_ID = PN_ID
        # 物理节点ID
        self.total_cpu = cpu
        # 处理能力总容量
        self.rest_cpu = cpu
        # 剩余可用容量
        self.overload = False
        # 过载状态
        self.MI = 0
        # 迁移指数
        self.VNF_list = []
        # 当前映射列表
        self.VNF_ID_list = []
        # VNF ID 列表

    def add_vnf(self, vnf: VNF):
        self.VNF_list.append(vnf)
        self.VNF_ID_list.append(vnf.VNF_ID)
        self.rest_cpu -= vnf.get_req_cpu()
        if self.rest_cpu/self.total_cpu < 1 - self.cpu_limit:   # 假设cpu使用阈值为 0.9， 若剩余资源比例小于0.1，则过载
            self.overload = True
        else:
            self.overload = False

    def del_vnf(self, vnf: VNF):
        self.VNF_list.remove(vnf)
        self.VNF_ID_list.remove(vnf.VNF_ID)
        self.rest_cpu += vnf.get_req_cpu()
        if self.rest_cpu/self.total_cpu < 1 - self.cpu_limit:
            self.overload = True
        else:
            self.overload = False

    def get_cpu_limit(self):
        return self.cpu_limit

    def get_total_cpu(self):
        return self.total_cpu

    def get_overload(self):
        return self.overload

    def set_overload(self, if_overload):
        self.overload = if_overload

    def get_rest_cpu(self):
        return self.rest_cpu

    def set_rest_cpu(self, cpu):
        self.rest_cpu = cpu

    def get_MI(self):
        return self.MI

    def set_MI(self, MI):
        self.MI = MI

    def get_PN_ID(self):
        return self.PN_ID

    def get_vnf_list(self):
        return self.VNF_list

    def set_vnf_list(self, VNF_list):
        self.VNF_list = VNF_list

    def update_vnf(self):           # 只需在更新前修改VNF的req_cpu值即可
        self.rest_cpu = self.total_cpu
        for vnf in self.VNF_list:
            self.rest_cpu -= vnf.get_req_cpu()
        if self.rest_cpu/self.total_cpu < 1 - self.cpu_limit:   # 假设cpu使用阈值为 0.9， 若剩余资源比例小于0.1，则过载
            self.overload = True
        else:
            self.overload = False
        # print("物理节点", self.PN_ID, "资源使用情况更新完成!")

    def if_vnf_in(self, VNF_ID):
        for vnf in self.VNF_list:
            if vnf.get_vnf_ID() == VNF_ID:
                return True
        return False

    def get_vnf_ID_list(self):
        return self.VNF_ID_list

    def set_vnf_ID_list(self, VNF_ID_list):
        self.VNF_ID_list = VNF_ID_list

    def check_cpu(self):
        total_vnf_cpu = 0
        for vnf in self.VNF_list:
            total_vnf_cpu += vnf.get_req_cpu()

        # print(self.total_cpu, self.rest_cpu, total_vnf_cpu)
        if total_vnf_cpu - (self.total_cpu - self.rest_cpu) < 0.001:
            return True
        else:
            return False
