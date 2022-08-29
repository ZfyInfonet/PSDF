# 使用说明:
# 在需要debug的地方插入debug().run(物理节点列表, VNF列表, SFC列表)
from data_structure.state import State

class Debug:

    @staticmethod
    def run(state: State, SFC_list):
        PN_list = state.get_PN_list()
        VNF_list = state.get_VNF_list()

        PN_num = len(PN_list)
        count = 1
        while True:
            print("*************************\n"
                  "*\t0: 查询物理节点信息\t*\n"
                  "*\t1: 查询全部VNF信息\t*\n"
                  "*\t2: 查询全部SFC信息\t*\n"
                  "*\te: 退出\t\t\t\t*\n"
                  "*************************")
            choose = input("请选择: ")
            print("----------------------第", count, "次查询--------------------------")
            if choose == '0':
                for i in range(PN_num):
                    temp_node = PN_list[i]
                    temp_node_vnflist = temp_node.get_vnf_list()
                    print("物理节点", temp_node.get_PN_ID(), "状态:")
                    print("cpu资源占用情况:", temp_node.get_total_cpu() - temp_node.get_rest_cpu(),
                          "/", temp_node.get_total_cpu(), "; VNF列表:", temp_node.get_vnf_ID_list(),
                          "; 过载状态:", temp_node.get_overload())
                    print("\tVNF详情:")
                    for j in range(len(temp_node_vnflist)):
                        temp_vnf = temp_node_vnflist[j]
                        print("\t VNF", temp_vnf.get_vnf_ID(), ": 资源需求:", temp_vnf.get_req_cpu(),
                              "; PN_ID: ", temp_vnf.get_PN_ID(), "; SFC_ID: ", temp_vnf.get_sfc_ID())
            elif choose == '1':
                for i in range(len(VNF_list)):
                    print("VNF", i, "信息: ID:", VNF_list[i].get_vnf_ID(), "; req_cpu:", VNF_list[i].get_req_cpu(),
                          "; PN_ID:", VNF_list[i].get_PN_ID(), "; SFC_ID:", VNF_list[i].get_sfc_ID())
            elif choose == '2':
                for i in range(len(SFC_list)):
                    print("SFC", SFC_list[i].get_sfc_ID(), "信息: VNF列表:", SFC_list[i].get_vnf_ID_list())
            elif choose == 'e':
                print("退出")
                print("-------------------------exit----------------------------")
                break
            else:
                print("无效查询")
            print("----------------------------------------------------------")
            count += 1
            count = count % 1000
