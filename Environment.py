"""
Environment.py
"""
__author__ = "giorgio@ac.upc.edu"

import numpy as np
from scipy import stats
import subprocess
import networkx as nx

from helper import pretty, softmax
from Traffic import Traffic


OMTRAFFIC = 'Traffic.txt'
OMBALANCING = 'Balancing.txt'
OMROUTING = 'Routing.txt'
OMDELAY = 'Delay.txt'
LINKSTATE = 'Linkstate.txt'
LOSSLOG = 'PacketLossRate.txt'
TRAFFICLOG = 'TrafficLog.csv'
BALANCINGLOG = 'BalancingLog.csv'
REWARDLOG = 'rewardLog.csv'
WHOLELOG = 'Log.csv'
OMLOG = 'omnetLog.csv'


# FROM MATRIX
def matrix_to_rl(matrix):
    return matrix[(matrix!=-1)]

matrix_to_log_v = matrix_to_rl

def matrix_to_omnet_v(matrix):
    return matrix.flatten()

def vector_to_file(vector, file_name, action):
    string = ','.join(pretty(_) for _ in vector)
    with open(file_name, action) as file:
        return file.write(string + '\n')


# FROM FILE
def file_to_csv(file_name):
    # reads file, outputs csv
    with open(file_name, 'r') as file:
        return file.readline().strip().strip(',')

def csv_to_matrix(string, nodes_num):
    # reads text, outputs matrix
    v = np.asarray(tuple(float(x) for x in string.split(',')[:nodes_num**2]))
    M = np.split(v, nodes_num)
    return np.vstack(M)

def csv_to_lost(string):
    return float(string.split(',')[-1])


# FROM RL
def rl_to_matrix(vector, nodes_num):
    M = np.split(vector, nodes_num)
    for _ in range(nodes_num):
        M[_] = np.insert(M[_], _, -1)
    return np.vstack(M)

def csv_to_vector(string, start, vector_num):
    v = np.asarray(string.split(',')[start:(start+vector_num)])
    M = v.astype(np.float)
    return M

# TO RL
def rl_state(env):
    return np.concatenate((matrix_to_rl(env.env_Linkstate), matrix_to_rl(env.env_T)))
  
def rl_reward(env):

    # 当存在不通的链路，则reward = -L- sum(不通链路数目), L为链路数目
    reward = 0
    # 当不存在不通链路时， reward = sum(utilization)

    return reward


# WRAPPER ITSELF
def omnet_wrapper(env):
    if env.ENV == 'label':
        sim = 'router'
    elif env.ENV == 'balancing':
        sim = 'balancer'

    prefix = ''
    if env.CLUSTER == 'arvei':
        prefix = '/scratch/nas/1/giorgio/rlnet/'

    simexe = prefix + 'omnet/' + sim + '/networkRL'
    simfolder = prefix + 'omnet/' + sim + '/'
    simini = prefix + 'omnet/' + sim + '/' + 'omnetpp.ini'

    try:
        omnet_output = subprocess.check_output([simexe, '-n', simfolder, simini, env.folder + 'folder.ini']).decode()
    except Exception as e:
        omnet_output = e.stdout.decode()

    if 'Error' in omnet_output:
        omnet_output = omnet_output.replace(',', '')
        o_u_l = [_.strip() for _ in omnet_output.split('\n') if _ is not '']
        omnet_output = ','.join(o_u_l[4:])
    else:
        omnet_output = 'ok'

    vector_to_file([omnet_output], env.folder + OMLOG, 'a')

# label environment
class OmnetLinkweightEnv():

    def __init__(self, DDPG_config, folder):
        self.ENV = 'label'
        self.ROUTING = 'Linkweight'

        self.folder = folder

        self.ACTIVE_NODES = DDPG_config['ACTIVE_NODES']

        self.ACTUM = DDPG_config['ACTUM']

        topology = 'omnet/router/NetworkAll.matrix'
        self.graph = nx.Graph(np.loadtxt(topology, dtype=int))
        self.topo = np.loadtxt(topology, dtype=int)
        self.origin_graph = nx.Graph(self.topo)
        if self.ACTIVE_NODES != self.graph.number_of_nodes():
            return False
        ports = 'omnet/router/NetworkAll.ports'
        self.ports = np.loadtxt(ports, dtype=int)

        self.a_dim = self.graph.number_of_edges()

        self.s_dim = self.ACTIVE_NODES**2 *2- self.ACTIVE_NODES    # traffic minus diagonal
        # self.s_dim = self.ACTIVE_NODES**2 * 2

        self.STATUM = DDPG_config['STATUM']
        if self.STATUM == 'RT':
            self.s_dim *= 2    # traffic + routing table minus diagonals

        self.PRAEMIUM = DDPG_config['PRAEMIUM']

        capacity = self.ACTIVE_NODES * (self.ACTIVE_NODES -1)

        self.TRAFFIC = DDPG_config['TRAFFIC']
        self.tgen = Traffic(self.ACTIVE_NODES, self.TRAFFIC, capacity)

        self.CLUSTER = DDPG_config['CLUSTER'] if 'CLUSTER' in DDPG_config.keys() else False

        self.env_T = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # traffic
        self.env_W = np.full([self.a_dim], -1.0, dtype=float)           # weights
        self.env_R = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)    # routing
        self.env_Rn = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)   # routing (nodes)
        self.env_L = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # link loss
        self.env_Linkstate = np.full([self.ACTIVE_NODES]*2, 1.0, dtype=float) # linkstate
        self.env_all_shortest = [] # 记录所有最短路径
        self.env_Bandwidth = np.full([self.ACTIVE_NODES]*2, 9.0 ,dtype=float) # 链路的带宽
        self.linkfailure = [] # 记录每次断掉的是哪一条链路
        self.linkset = [] # 记录50次断掉链路的情况
        self.counter = 0
        


    def upd_env_T(self, matrix):
        self.env_T = np.asarray(matrix)
        np.fill_diagonal(self.env_T, -1)

    
    
    def upd_env_W(self, vector):
        self.env_W = np.asarray(softmax(vector))

    def check_if_link_failure(self):
        for s in range(self.ACTIVE_NODES):
            for d in range(self.ACTIVE_NODES):
                if s!=d:
                    path = self.env_all_shortest[s][d]
                    for i in range(len(path)-1):
                        node = path[i]
                        next = path[i+1]

                        if self.env_T[s][d]!=0 and self.env_Linkstate[node][next]==0 :
                            # sp = nx.all_simple_paths(self.graph, source=s, target=d)
                            # for p in sp:
                            #     print(p)
                            print(path)
                            print(node, next)
                            # print(self.graph.edges())
                            # print('link:  ',self.env_Linkstate[node][next])
                            return False

        return True

    def upd_env_R(self):
        weights = {}

        for e, w in zip(self.graph.edges(), self.env_W):
            weights[e] = w
        
        nx.set_edge_attributes(self.graph, 'weight', weights)
        # print(self.env_W)

        routing_nodes = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)
        routing_ports = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)

        all_shortest = nx.all_pairs_dijkstra_path(self.graph)
        self.env_all_shortest = all_shortest.copy()
        sucess = self.check_if_link_failure()
        if sucess == False:
            return sucess
        for s in range(self.ACTIVE_NODES):
            for d in range(self.ACTIVE_NODES):
                if s != d:
                    next = all_shortest[s][d][1]
                    port = self.ports[s][next]
                    routing_nodes[s][d] = next
                    routing_ports[s][d] = port
                else:
                    routing_nodes[s][d] = -1
                    routing_ports[s][d] = -1

        self.env_R = np.asarray(routing_ports)
        self.env_Rn = np.asarray(routing_nodes)
        return sucess

    

    def upd_env_L(self, matrix):
        '''
        更新loss，并计算链路利用率
        '''
        self.env_L = np.asarray(matrix)
        linktraffic = np.full([self.ACTIVE_NODES]*2,0,dtype=float)
        linkutilization = np.full([self.ACTIVE_NODES]*2,0,dtype=float)
        # 计算每条链路traffic总和
        for s in range(self.ACTIVE_NODES):
            for d in range(self.ACTIVE_NODES):
                if s!=d:
                    path = self.env_all_shortest[s][d]
                    for i in range(len(path)-1):
                        next = path[i+1]
                        linktraffic[path[i]][next] += self.env_T[s][d]
        
        for i in range(self.ACTIVE_NODES):
            for j in range(self.ACTIVE_NODES):
                
                linkutilization[i][j] = ( 1-self.env_L[i][j] ) * linktraffic[i][j] / self.env_Bandwidth[i][j]
               
        linkutilization = linkutilization.clip(min=0,max=1)
        self.env_Linkstate = (1-linkutilization).copy() # linkstate= 1- utilization
       
    def generate_set_link_failure(self):
        '''
        生成50条链路断掉的情况
        '''
        linkset = []
        edges = self.graph.edges()
        graph = self.graph.copy()
        for i in range(50):
            while True:
                number = np.random.choice(len(edges))
                link = edges[number]
                graph = self.graph.copy()
                graph.remove_edge(link[0],link[1])
                if nx.is_connected(graph)==True:
                    break
            linkset.append(link)
        self.linkset = linkset.copy()

    def choose_one_link_failure_from_set(self, step):
        link = self.linkset[step]
        print('断掉的链路是:', link)
        self.linkfailure = link
        # self.graph.remove_edge(link[0],link[1])
        self.env_Linkstate = self.topo.copy()
        self.env_Linkstate[link[0]][link[1]] = 0
        self.env_Linkstate[link[1]][link[0]] = 0



    def generate_link_failure(self):
        '''
        选择有一条链路断掉了，topo发生了变化
        '''
        # self.graph = self.origin_graph.copy()
        edges = self.graph.edges()
        graph = self.graph.copy()
        while True:
            number = np.random.choice(len(edges))
            link = edges[number]
            graph = self.graph.copy()
            graph.remove_edge(link[0],link[1])
            if nx.is_connected(graph)==True:
                break

            
            
        print('断掉的链路是:', link)
        self.linkfailure = link
        # self.graph.remove_edge(link[0],link[1])
        self.env_Linkstate = self.topo.copy()
        self.env_Linkstate[link[0]][link[1]] = 0
        self.env_Linkstate[link[1]][link[0]] = 0
        
        # print(self.graph.edges())
        # print(self.origin_graph.edges())
        # print(self.topo-self.env_Linkstate)
        # input('---')

        # self.env_Linkstate = self.topo.copy()
        # np.fill_diagonal(self.env_Linkstate,-1)
        # new_link = self.env_Linkstate.reshape(1, self.ACTIVE_NODES**2)
        
        # while True:
        #     number = np.random.choice(self.ACTIVE_NODES**2)
        #     if new_link[0][number] !=0 and new_link[0][number]!=-1:
        #         new_link[0][number]=0
        #         break
        # self.env_Linkstate = new_link.reshape(self.ACTIVE_NODES, self.ACTIVE_NODES)
        # np.fill_diagonal(self.env_Linkstate, 0)



   
    def upd_env_Linkstate(self, matrix):
        self.env_Linkstate = np.asarray(matrix)
        np.fill_diagonal(self.env_Linkstate, 0)
        
            

    def render(self):
        return


    def reset(self, easy=False):
        if self.counter != 0:
            return None

        # routing
        self.upd_env_W(np.full([self.a_dim], 0.50, dtype=float))
        self.upd_env_R()
        vector_to_file(matrix_to_omnet_v(self.env_R), self.folder + OMROUTING, 'w')
        
        # link state
        self.upd_env_Linkstate(np.full([self.ACTIVE_NODES]*2, 1, dtype=float)) # 
        vector_to_file(matrix_to_omnet_v(self.env_Linkstate), self.folder + LINKSTATE, 'w')
        

        # traffic
        self.upd_env_T(self.tgen.generate())

        vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        # 生成50条链路断掉的情况
        self.generate_set_link_failure()

        return rl_state(self)


    def step(self, action, step_cnt):
        self.counter += 1

        self.upd_env_W(action)
        
        sucess = self.upd_env_R()
        if sucess == False:
            # 存在不通的路径
            # self.graph = self.origin_graph # 回到初始没有链路断掉的情况
            reward = -1
            self.graph = self.origin_graph.copy()
            self.upd_env_T(self.tgen.generate())
            self.upd_env_Linkstate(self.topo.copy())
            new_state = rl_state(self)
            # input('--------')
            vector_to_file([reward], self.folder + REWARDLOG, 'a')
            return new_state, reward, 1



        # 判断traffic 和 routing 是否冲突


        # write to file input for Omnet: Routing
        vector_to_file(matrix_to_omnet_v(self.env_R), self.folder + OMROUTING, 'w')
        # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")

        # execute omnet
        omnet_wrapper(self)

        # read Omnet's output: Lost
        om_output = file_to_csv(self.folder + LOSSLOG)
        
        self.upd_env_L(csv_to_matrix(om_output, self.ACTIVE_NODES))
        # print(self.env_L)
        # input('---')
        

        # reward = rl_reward(self)
        reward = np.sum(1-self.env_Linkstate)

        # log reward to file
        vector_to_file([reward], self.folder + REWARDLOG, 'a')
        cur_state = rl_state(self)
        
        # generate traffic for next iteration
        # self.generate_link_failure()
        self.choose_one_link_failure_from_set(step_cnt)
        self.upd_env_T(self.tgen.generate())
        
        # write to file input for Omnet: Traffic, or do nothing if static
        if self.TRAFFIC.split(':')[0] not in ('STAT', 'STATEQ', 'FILE'):
            vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        new_state = rl_state(self)
        # return new status and reward
        return new_state, reward, 0


    def end(self):
        return
