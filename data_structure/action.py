class Action:

    def __init__(self, mig_vnf_list, mig_des_list):
        self.mig_vnf_list = mig_vnf_list    # 共|VNF|个0或1
        self.mig_des_list = mig_des_list    # 共|VNF|个物理节点编号PN_ID

    def get_mig_vnf_list(self):
        return self.mig_vnf_list

    def get_mig_des_list(self):
        return self.mig_des_list
