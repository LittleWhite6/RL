import numpy as np
import random
import math


problem_seed = 0
# Global seed
num_train_points = 20
# Global 训练样本点数


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


def construct_solution(problem):
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
    init_solution = Solution(paths)
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


class Solution:
    def __init__(self, paths):
        self.path = paths
        self.cost = 0
        self.path_load = [] #测试算子用列表
            
    def get_cost(self, problem):
        dist = 0
        for path_num in range(len(self.path)):
            for i in range(1, len(self.path[path_num])):
                dist += problem.dist_matrix[self.path[path_num][i-1]][self.path[path_num][i]]
        return dist

    def show_path(self):
        for path_num in range(len(self.path)):
            for i in range(len(self.path[path_num])-1):
                print(self.path[path_num][i], end="->")
            print(0)