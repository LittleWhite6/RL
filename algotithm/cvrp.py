import numpy as np
import random
import math


from hyper_parameter import *


depot_positionings = {0: 'R', 1: 'E', 2: 'C'}
customer_positionings = {0: 'R', 1: 'C', 2: 'RC'}
# New benchmark instances for the Capacitated Vehicle Routing Problem 2017 EJOR


def calculate_distance(point0, point1):
    dx = point1[0] - point0[0]
    dy = point1[1] - point0[1]
    return math.sqrt(dx * dx + dy * dy)


def generate_problem():
    global problem_seed
    #depot_positioning = depot_positionings.get(random.randint(0, 2))
    depot_positioning = 'R'
    #customer_positioning = customer_positionings.get(random.randint(0, 2))
    customer_positioning = 'R'
    np.random.seed(problem_seed)
    random.seed(problem_seed)
    problem_seed += 1
    locations = np.random.uniform(size=(num_train_points+1, 2))
    # depot+customers random generation
    if depot_positioning == 'C':  # Central depot
        locations[0][0] = 0.5
        locations[0][1] = 0.5
    elif depot_positioning == 'E':  # Eccentric depot
        locations[0][0] = 0.0
        locations[0][1] = 0.0

    if customer_positioning != 'R':
        S = np.random.randint(6) + 3
        centers = locations[1: (S + 1)]
        grid_centers, probabilities = [], []
        for x in range(0, 1000):
            for y in range(0, 1000):
                grid_center = [(x + 0.5) / 1000.0, (y + 0.5) / 1000.0]
                p = 0.0
                for center in centers:
                    distance = calculate_distance(grid_center, center)
                    # 乘1000是因为2017EJOR中的grid是1000*1000，而np.random生成的grid是1*1
                    p += math.exp(-distance * 1000.0 / 40.0)
                grid_centers.append(grid_center)
                probabilities.append(p)
        probabilities = np.asarray(probabilities) / np.sum(probabilities)
        if customer_positioning in 'C':
            num_clustered_locations = num_train_points - S
        else:
            num_clustered_locations = num_train_points // 2 - S
        grid_indices = np.random.choice(
            range(len(grid_centers)), num_clustered_locations, p=probabilities)
        for index in range(num_clustered_locations):
            grid_index = grid_indices[index]  # 将customer分布在choice选中的格点附近
            locations[index + S + 1][0] = grid_centers[grid_index][0] + \
                (np.random.uniform() - 0.5) / 1000.0
            locations[index + S + 1][1] = grid_centers[grid_index][1] + \
                (np.random.uniform() - 0.5) / 1000.0

    capacities = [0] * (num_train_points+1)
    depot_capacity_map = {
        10: 20,
        20: 30,
        50: 40,
        100: 50,
        200: 50
    }
    # dict.get(key,default) 第一个参数为键值，第二个参数为默认值，返回值为键值对应的值，不存在则为默认值
    capacities[0] = depot_capacity_map.get(num_train_points, 50)
    for i in range(1, num_train_points+1):
        capacities[i] = np.random.randint(9) + 1

    init_problem = Problem(locations, capacities)
    return init_problem


def find_minimal_index(problem, nodes_exist, load, last_node):
    min_dist = float("inf")
    index = 0
    # 初始距离设为无穷大
    for i in range(1, num_train_points + 1):
        if problem.dist_matrix[last_node][i] < min_dist \
        and nodes_exist[i-1] == 0 \
        and load - problem.capacities[i] > 0:
            min_dist = problem.dist_matrix[last_node][i]
            index = i
    return index


def random_construct(problem, solution):
    node_list = [i+1 for i in range(num_train_points)]
    #标记已服务的节点
    for i in range(len(solution.path)):
        for j in range(1, len(solution.path[i]) - 1):
            node_list[solution.path[i][j] - 1] = -1
    #去除solution中已经存在的节点
    while node_list.__contains__(-1):
        node_list.remove(-1)
    while node_list:
        load = problem.capacities[0]
        path = [0]
        while load > 0 and node_list:
            node = random.choice(node_list)
            if load > problem.capacities[node]:
                path.append(node)
                node_list.remove(node)
                load -= problem.capacities[node]
            else:
                #如果检查到超负荷就直接构建新的路径
                path.append(0)
                solution.path.append(path)
                break
        if not node_list:
            path.append(0)
            solution.path.append(path)
    #是否需要？
    problem.reset_change_at_and_no_improvement_at()
    return solution


def construct_solution(problem, default_random = True):
    #random construct
    if default_random:
        node_list = [i+1 for i in range(num_train_points)]
        paths = []
        while node_list:
            load = problem.capacities[0]
            path = [0]
            while load > 0 and node_list:
                node = random.choice(node_list)
                if load > problem.capacities[node]:
                    path.append(node)
                    node_list.remove(node)
                    load -= problem.capacities[node]
                else:
                    #如果检查到超负荷就直接构建新的路径
                    path.append(0)
                    paths.append(path)
                    break
            if not node_list:
                path.append(0)
                paths.append(path)
        init_solution = Solution(problem, paths)
        return init_solution

    #min_dist heuristic construct
    else:
        nodes_exist = np.zeros(num_train_points)
        paths = []
        while nodes_exist.__contains__(0):
            path = [0]
            load = problem.capacities[0]
            while load > 0:
                next_node = find_minimal_index(
                    problem, nodes_exist, load, path[-1])
                if next_node != 0:
                    nodes_exist[next_node - 1] = 1
                path.append(next_node)
                load -= problem.capacities[next_node]
            paths.append(path)
        init_solution = Solution(problem, paths)
        return init_solution


class Problem:
    def __init__(self, locations, capacities):
        self.locations = locations
        self.capacities = capacities
        self.dist_matrix = []
        dist_matrix_row = []
        for i in range(len(self.locations)):
            for j in range(len(self.locations)):
                dist_matrix_row.append(calculate_distance(
                    self.locations[i], self.locations[j]))
            self.dist_matrix.append(dist_matrix_row)
            dist_matrix_row = []
        self.change_at = [0] * (len(self.locations) + 1)
        self.no_improvement_at = {}

    def reset_change_at_and_no_improvement_at(self):
        self.change_at = [0] * (len(self.locations) + 1)
        self.no_improvement_at = {}
    
    def mark_change_at(self, step, path_indices):
        for path_index in path_indices:
            self.change_at[path_index] = step
    
    def mark_no_improvement(self, step, action, index_first, index_second=-1, index_third=-1):
        key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
        self.no_improvement_at[key] = step

    def should_try(self, action, index_first, index_second=-1, index_third=-1):
        key = '{}_{}_{}_{}'.format(action, index_first, index_second, index_third)
        no_improvement_at = self.no_improvement_at.get(key, -1)
        return self.change_at[index_first] >= no_improvement_at or \
            self.change_at[index_second] >= no_improvement_at or \
            self.change_at[index_third] >= no_improvement_at

class Solution:
    def __init__(self, problem, paths):
        self.path = paths
        self.cost = self.get_cost(problem)  
        self.path_load = [] #测试算子用列表


    #获得车辆在节点i处的负载
    def get_vehicle_load(self, problem, path, node):
        load = 0
        if node == 0:
            return problem.capacities[0]
        for i in range(1, len(path)):
            load += problem.capacities[path[i]]


    #获得解决方案的cost
    def get_cost(self, problem):
        dist = 0.0
        for path_num in range(len(self.path)):
            for i in range(1, len(self.path[path_num])):
                dist += problem.dist_matrix[self.path[path_num][i-1]][self.path[path_num][i]]
        return dist


    #展示解决方案的路径
    def show_path(self):
        for path_num in range(len(self.path)):
            for i in range(len(self.path[path_num])-1):
                print(self.path[path_num][i], end="->")
            print(0)