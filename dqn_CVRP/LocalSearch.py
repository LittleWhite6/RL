from cvrp import *

import copy

EPSILON = 1e-6


def calculate_travel_node_dist(problem, node_i, node_j):
    return problem.dist_matrix[node_i][node_j]


def intra_2_opt(problem, path):
    n = len(path) - 1
    max_delta = EPSILON
    label = None
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            before = calculate_travel_node_dist(
                problem, path[i - 1], path[i]) + calculate_travel_node_dist(problem, path[j], path[j + 1])
            after = calculate_travel_node_dist(
                problem, path[i - 1], path[j]) + calculate_travel_node_dist(problem, path[i], path[j + 1])
            delta = before - after
            if before-after > max_delta:
                max_delta = delta
                label = i, j
    if label != None:
        first = label[0]
        second = label[1]
        while first < second:
            path[first], path[second] = path[second], path[first]
            first+=1
            second-=1
    return path, label




def main():
    problem = generate_problem()
    solution = construct_solution(problem)
    solution.show_path()
    ls = True
    path_prob = [0.25]*len(solution.path)   #只能给4条路径的solu分布0.25概率，如何真正实现均匀分布概率
    path_num = np.zeros(len(solution.path))
    for i in range(len(solution.path)):
        path_num[i] = i
    count = 0
    last_route_prob = None
    while ls == True:
        count += 1
        print("当前为第{}次迭代".format(count),solution.get_cost(problem))
        i = np.random.choice(path_num, p=path_prob).astype(int)
        solution.path[i], label = intra_2_opt(problem, solution.path[i])
        if label == None:
            last_route_prob = path_prob[i]
            path_prob[i] = 0
            for j in range(len(path_prob)):
                if path_prob[j] == 0:
                    pass
                else:
                    path_prob[j] += last_route_prob
                    break
        cunzai = 0
        for z in range(len(path_prob)):
            if path_prob[z] != 0:
                cunzai += 1
                break
        if cunzai == 0:
            ls = False
    solution.show_path()
    print(solution.get_cost(problem))


if __name__ == "__main__":
    main()