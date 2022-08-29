

class SFC:

    def __init__(self, SFC_ID, traffic, VNF_ID_list):
        self.SFC_ID = SFC_ID
        self.traffic = traffic
        self.VNF_ID_list = VNF_ID_list

    def if_vnf_in_SFC(self, VNF_ID: int):
        for ID in self.VNF_ID_list:
            if ID == VNF_ID:
                return True
        return False

    def add_vnf_ID(self, VNF_ID):
        self.VNF_ID_list.append(VNF_ID)

    def get_sfc_ID(self):
        return self.SFC_ID

    def get_traffic(self):
        return self.traffic

    def get_vnf_ID_list(self):
        return self.VNF_ID_list
