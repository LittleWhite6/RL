from LocalSearch import *

def select_path(path_non_optimal):
    #每次随机选择路径
    paths = list(path_non_optimal.keys()) #返回全部待优化路径
    n = len(paths)
    path_index = random.randint(0, n - 1)
    path = paths[path_index]
    return path


def select_2_paths(path_non_optimal):
    #完全随机选择俩条路径
    paths = list(path_non_optimal.keys())
    paths = np.random.choice(paths, 2, False)
    return paths[0], paths[1]


def env_step(problem, solution, action):
    # 随机选择优化路径
    next_solution = copy.deepcopy(solution)
    if action in range(0, 3):
        #生成待intra优化的路径
        paths_intra_ls = {}
        for i in range(len(solution.path)):
            paths_intra_ls[i] = 0
        path = select_path(paths_intra_ls)
        if action == 0:
            next_solution.path[path], label = intra_two_Opt(problem, solution.path[path])
        elif action == 1:
            next_solution.path[path], label = intra_Symmetric_exchange(problem, solution.path[path])
        elif action == 2:
            next_solution.path[path], label = intra_Relocate(problem, solution.path[path])
        if label:
            paths_intra_ls[path] += 1   #可以优先选择改进少的路径优化
        else:
            paths_intra_ls.pop(path)
    elif action in range(3, 9):
        paths_inter_ls = {}
        n = len(solution.path)
        '''
        if n < 2:
            return solution #待优化路径总数小于2时返回原解决方案
        '''
        for i in range(n):
            paths_inter_ls[i] = 0
        path_first, path_second = select_2_paths(paths_inter_ls)
        #生成inter优化的路径
        if action == 3:
            next_solution.path[path_first], solution.path[path_second], label = inter_Cross(problem, solution.path[path_first], solution.path[path_second])
        elif action == 4:
            next_solution.path[path_first], solution.path[path_second], label = inter_Reverse_cross(problem, solution.path[path_first], solution.path[path_second])
        elif action == 5:
            next_solution.path[path_first], solution.path[path_second], label = Symmetric_exchange(problem, solution.path[path_first], solution.path[path_second], [random.randint(0, 2), random.randint(0, 2)])
        elif action == 6: 
            next_solution.path[path_first], solution.path[path_second], label = Asymmetric_exchange(problem, solution.path[path_first], solution.path[path_second], [random.randint(0, 2), random.randint(0, 2)])
        elif action ==7:
            next_solution.path[path_first], solution.path[path_second], label = Relocate(problem, solution.path[path_first], solution.path[path_second], random.randint(1, 3))
        elif action ==8:
            next_solution == Cyclic_exchange(problem, solution)
        #此次优化成功
        if label:
            paths_inter_ls[path_first] += 1
            paths_inter_ls[path_second] += 1
        else:
            paths_inter_ls.pop(path_first)
            paths_inter_ls.pop(path_second)
    else:
        next_solution = Random_permute(problem, solution)
        label = None
    return next_solution, label
