

class VNF:
    def __init__(self, VNF_ID, VNF_req_cpu):
        # VNF编号
        self.VNF_ID = VNF_ID
        # VNF所需求的CP资源总数
        self.VNF_req_cpu = VNF_req_cpu
        # VNF所处的物理节点的编号
        self.PN_ID = -1      # -1表示没有映射
        # VNF归属SFC流的编号
        self.SFC_ID = -1
        # VNF是否已被迁移
        self.flag = 0

    def set_vnf_ID(self, ID):
        self.VNF_ID = ID

    def get_vnf_ID(self):
        return self.VNF_ID

    def set_req_cpu(self, VNF_req_cpu):
        self.VNF_req_cpu = VNF_req_cpu

    def get_req_cpu(self):
        return self.VNF_req_cpu

    def set_PN_ID(self, PN_ID):
        self.PN_ID = PN_ID

    def get_PN_ID(self):
        return self.PN_ID

    def set_sfc_ID(self, SFC_ID):
        self.SFC_ID = SFC_ID

    def get_sfc_ID(self):
        return self.SFC_ID

    def set_flag(self, flag):
        self.flag = flag

    def get_flag(self):
        return self.flag