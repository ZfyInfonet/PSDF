class State:

    def __init__(self, PN_list, VNF_list, traffic_list):
        self.PN_list = PN_list
        self.VNF_list = VNF_list
        self.traffic_list = traffic_list

    def get_PN_list(self):
        return self.PN_list

    def get_VNF_list(self):
        return self.VNF_list

    def get_traffic_list(self):
        return self.traffic_list

    def set_traffic_list(self, traffic_list):
        self.traffic_list = traffic_list
