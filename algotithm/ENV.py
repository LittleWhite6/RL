from LocalSearch import *

def select_2_paths(path_non_optimal):
    #完全随机选择俩条路径
    paths = list(path_non_optimal.keys())
    paths = np.random.choice(paths, 2, False)
    return paths[0], paths[1]


def env_step(step, problem, solution, action, no_change):
    # !随机选择优化路径!
    num_paths = len(solution.path)
    next_solution = copy.deepcopy(solution)
    delta_sum = 0.0
    label = None
    if action in range(0, 3):
        for path_index in range(num_paths):
            modified = problem.should_try(action, path_index)
            while modified:
                if action == 0:
                    next_solution.path[path_index], label, delta = intra_two_Opt(problem, next_solution.path[path_index])
                elif action == 1:
                    next_solution.path[path_index], label, delta = intra_Symmetric_exchange(problem, next_solution.path[path_index])
                elif action == 2:
                    next_solution.path[path_index], label, delta = intra_Relocate(problem, next_solution.path[path_index])
                if label:
                    problem.mark_change_at(step, [path_index])
                    delta_sum += delta
                else:
                    modified = False
                    problem.mark_no_improvement(step, action, path_index)
        return next_solution, delta_sum

    if action in range(3, 21):
        for path_first in range(num_paths - 1):
            for path_second in range(path_first + 1, num_paths):
                modified = problem.should_try(action, path_first, path_second)
                while modified:
                    #生成inter优化的路径
                    if action == 3:
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = inter_Cross(problem, next_solution.path[path_first], next_solution.path[path_second])
                    elif action == 4:
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = inter_Reverse_cross(problem, next_solution.path[path_first], next_solution.path[path_second])
                    elif action == 5:
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Symmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [1, 1])
                    elif action == 6:
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Symmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [2, 2])
                    elif action == 7:
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Symmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [3, 3])

                    elif action == 8: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [1, 1])
                    elif action == 9: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [1, 2])
                    elif action == 10: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [1, 3])
                    elif action == 11: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [2, 1])
                    elif action == 12: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [2, 2])
                    elif action == 13: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [2, 3])
                    elif action == 14: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [3, 1])
                    elif action == 15: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [3, 2])
                    elif action == 16: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [3, 3])

                    elif action == 17:
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Relocate(problem, next_solution.path[path_first], next_solution.path[path_second], 1)
                    elif action == 18:
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Relocate(problem, next_solution.path[path_first], next_solution.path[path_second], 2)
                    elif action == 19:
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Relocate(problem, next_solution.path[path_first], next_solution.path[path_second], 3)
                    elif action == 20:
                        #delta < 0???
                        next_solution, label, delta = Cyclic_exchange(problem, next_solution)
                    #此次优化成功
                    if label:
                        problem.mark_change_at(step, [path_first, path_second])
                        delta_sum += delta
                    else:
                        modified = False
                        problem.mark_no_improvement(step, action, path_first, path_second)
        return next_solution, delta_sum

    if action == 21 and no_change >= 5:
        next_solution = Random_permute(problem, next_solution)
        return next_solution, delta_sum
    else:
        return next_solution, delta_sum