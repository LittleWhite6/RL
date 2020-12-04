from cvrp import *

import copy


def calculate_travel_node_dist(problem, node_i, node_j):
    return problem.dist_matrix[node_i][node_j]


# intra


# operator1: 2_opt
def intra_two_Opt(problem, path):
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
            first += 1
            second -= 1
    return path, label


# operator2: Symmetric-exchange
def intra_Symmetric_exchange(problem, path):
    n = len(path) - 1
    max_delta = EPSILON
    label = None
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            if j == i + 1:
                before = calculate_travel_node_dist(
                    problem, path[i - 1], path[i]) + calculate_travel_node_dist(problem, path[j], path[j + 1])
                after = calculate_travel_node_dist(
                    problem, path[i - 1], path[j]) + calculate_travel_node_dist(problem, path[i], path[j + 1])
            else:
                before = calculate_travel_node_dist(problem, path[i-1], path[i]) + calculate_travel_node_dist(
                    problem, path[i], path[i + 1]) + calculate_travel_node_dist(problem, path[j - 1], path[j]) + calculate_travel_node_dist(problem, path[j], path[j + 1])
                after = calculate_travel_node_dist(problem, path[i-1], path[j]) + calculate_travel_node_dist(
                    problem, path[j], path[i + 1]) + calculate_travel_node_dist(problem, path[j - 1], path[i]) + calculate_travel_node_dist(problem, path[i], path[j + 1])
            delta = before - after
            if delta > max_delta:
                max_delta = delta
                label = i, j
    if label != None:
        path[label[0]], path[label[1]] = path[label[1]], path[label[0]]
    return path, label


# operator3: Relocate
def intra_Relocate(problem, path):
    n = len(path) - 1
    max_delta = EPSILON
    label = None
    for i in range(1, n):
        for j in range(n):
            before = calculate_travel_node_dist(problem, path[i - 1], path[i]) + calculate_travel_node_dist(
                problem, path[i], path[i + 1]) + calculate_travel_node_dist(problem, path[j], path[j + 1])
            after = calculate_travel_node_dist(problem, path[j], path[i]) + calculate_travel_node_dist(
                problem, path[i], path[j + 1]) + calculate_travel_node_dist(problem, path[i - 1], path[i + 1])
            delta = before - after
            if delta > max_delta:
                max_delta = delta
                label = i, j
    if label != None:
        path.insert(label[1] + 1, path[label[0]])
        path.pop(label[0])
    return path, label


# inter (multi-segment 未测试)


def get_path_load(problem, path):
    n = len(path) - 1
    # 不包含最后的depot
    path_load = [0] * n
    for i in range(1, n):
        path_load[i] = path_load[i - 1] + problem.capacities[path[i]]
    return path_load


# operator3: Cross
def inter_Cross(problem, path_first, path_second):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    max_delta = EPSILON
    label = None
    load_first = get_path_load(problem, path_first)
    load_second = get_path_load(problem, path_second)
    start = 1
    for first in range(1, n_first):
        unserved_customer_first = load_first[n_first - 2] - load_first[first - 1]
        for second in range(start, n_second):
            unserved_customer_second = load_second[n_second - 2] - load_second[second - 1]
            # 检查第二个车辆服务第一个路径剩余的节点是否会超载
            if unserved_customer_first + load_second[second - 1] > problem.capacities[0]:
                break
            elif unserved_customer_second + load_first[first - 1] > problem.capacities[0]:
                start = second
                continue
            before = calculate_travel_node_dist(problem, path_first[first - 1], path_first[first]) + calculate_travel_node_dist(
                problem, path_second[second - 1], path_second[second])
            after = calculate_travel_node_dist(problem, path_first[first - 1], path_second[second]) + calculate_travel_node_dist(
                problem, path_second[second - 1], path_first[first])
            delta = before - after
            if delta > max_delta:
                max_delta = delta
                label = first, second
    if label != None:
        path_temp = path_first[:]
        path_first = path_temp[:label[0]] + path_second[label[1]:]
        path_second = path_second[:label[1]] + path_temp[label[0]:]
    return path_first, path_second, label


# operator4: Reverse-cross
def inter_Reverse_cross(problem, path_first, path_second):
    #Reverse one of two route
    reverse_index = random.randint(0, 1)
    '''
    如何实现完全随机选取翻转的路径?
    random.seed(problem_seed)
    problem_seed *= 10
    '''
    if reverse_index == 0:
        path_first = path_first[::-1]
    else:
        path_second = path_second[::-1]
    return inter_Cross(problem, path_first, path_second)


# operator5: Symmetric-exchange(segments = [m, n], definition before passing, m 代表第一个切片长度，n代表第二个切片长度, 0,1,2 = len: 1,2,3)
def Symmetric_exchange(problem, path_first, path_second, segments):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    max_delta = EPSILON
    label = None
    load_first = get_path_load(problem, path_first)
    load_second = get_path_load(problem, path_second)

    for first in range(n_first - segments[0]):
        head_first = first + 1
        tail_first = first + segments[0]
        segments_load_first = load_first[tail_first] - load_first[first]
        remains_capacity_first = problem.capacities[0] - load_first[-1] + segments_load_first
        for second in range(n_second - segments[1]):
            head_second = second + 1
            tail_second = second + segments[1]
            segments_load_second = load_second[tail_second] - load_second[second]
            remains_capacity_second = problem.capacities[0] - load_second[-1] + segments_load_second
            if segments_load_first > remains_capacity_second or segments_load_second > remains_capacity_first:
                continue
            else:
                before = calculate_travel_node_dist(problem, path_first[first], path_first[head_first]) \
                    + calculate_travel_node_dist(problem, path_first[tail_first], path_first[tail_first + 1]) \
                    + calculate_travel_node_dist(problem, path_second[second], path_second[head_second])\
                    + calculate_travel_node_dist(problem, path_second[tail_second], path_second[tail_second + 1])
                after = calculate_travel_node_dist(problem, path_first[first], path_second[head_second]) \
                    + calculate_travel_node_dist(problem, path_second[tail_second], path_first[tail_first + 1]) \
                    + calculate_travel_node_dist(problem, path_second[second], path_first[head_first])\
                    + calculate_travel_node_dist(problem, path_first[tail_first], path_second[tail_second + 1])
                delta = before - after
                if delta > max_delta:
                    max_delta = delta
                    label = head_first, tail_first, head_second, tail_second
    if label != None:
        path_temp = path_first[:]
        path_first = path_first[:label[0]] + path_second[label[2]:label[3]] + path_first[label[1] + 1:]
        path_second = path_second[:label[2]] + path_temp[label[0]:label[1]] + path_second[label[3] + 1:]
    return path_first, path_second, label

            
# operator6: Asymmetric-exchange
def Asymmetric_exchange(problem, path_first, path_second, segments):
    return Symmetric_exchange(problem, path_first, path_second, segments)


# operator7: Relocate(segments = m: 1, 2, 3)
def Relocate(problem, path_first, path_second, segment):
    n_first = len(path_first) - 1
    n_second = len(path_second) - 1
    max_delta = EPSILON
    label = None
    load_first = get_path_load(problem, path_first)
    load_second = get_path_load(problem, path_second)
    remains_capacity_second = problem.capacities[0] - load_second[-1]

    for first in range(n_first - segment):
        head_first = first + 1
        tail_first = first + segment
        segment_load_first = load_first[tail_first] - load_first[first]        
        for second in range(n_second):
            if remains_capacity_second < segment_load_first:
                break
            before = calculate_travel_node_dist(problem, path_first[first], path_first[head_first]) \
                + calculate_travel_node_dist(problem, path_first[tail_first], path_first[tail_first + 1])
            after = calculate_travel_node_dist(problem, path_second[second], path_first[head_first]) \
                + calculate_travel_node_dist(problem, path_first[tail_first], path_second[second + 1])
            delta = before - after
            if delta > max_delta:
                max_delta = delta
                label = first, head_first, tail_first, second
    if label != None:
        if segment == 1:
            node = path_first[label[1]]
            path_first.pop(label[1])
            path_second.insert(label[3] + 1, node)
        else:
            segment = path_first[label[1]:label[2]]
            path_first = path_first[:label[1]] + path_first[label[2]:]
            path_second = path_second[:label[3] + 1] + segment + path_second[label[3] + 2:]
    return path_first, path_second, label


# operator8: Cyclic_exchange(每次循环交换一个顾客)
def Cyclic_exchange(problem, solution):
    path_num = len(solution.path)
    for i in range(path_num - 1):
        solution.path[i], solution.path[i + 1], label = Relocate(problem, solution.path[i], solution.path[i + 1], 1)
    solution.path[path_num], solution.path[1], label = Relocate(problem, solution.path[path_num], solution.path[1], 1)
    return solution


# Perturbation operators:


def random_construct(problem, solution):
    node_list = [i+1 for i in range(num_train_points)]
    #标记已服务的节点
    for i in range(len(solution.path)):
        for j in range(len(solution.path[i])):
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
    return solution
                

# operator9: Random-permute (先全部删除路径再随机构建)
def Random_permute(problem, solution):
    n = len(solution.path)
    #每次选几个路径，当前为随机
    m = random.randint(1, n)
    #每次选择哪几个路径，当前随机
    path_index = [i for i in range(n)]
    destory_paths = np.random.choice(path_index, m, replace=False).tolist()
    destory_paths.sort(reverse=True)
    for i in range(len(destory_paths)):
        solution.path.pop(destory_paths[i])
    solution = random_construct(problem, solution)


'''
# operator10: Random-exchang
def Random_exchange(problem, solution):
    min_len = float("inf")
    for i in range(len(solution.path)):
        curr_len = len(solution.path[i])
        if curr_len < min_len:
            min_len = curr_len
    m = random.randint(1, min_len)
'''