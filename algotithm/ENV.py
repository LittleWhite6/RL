from LocalSearch import *

def select_2_paths(path_non_optimal):
    #完全随机选择俩条路径
    paths = list(path_non_optimal.keys())
    paths = np.random.choice(paths, 2, False)
    return paths[0], paths[1]


def reconstruct_solution_by_exchange(problem, existing_solution, paths_ruined):
    path0 = copy.deepcopy(existing_solution[paths_ruined[0]])
    path1 = copy.deepcopy(existing_solution[paths_ruined[1]])
    num_exchanged = 0
    for i in range(1, len(path0) - 1):
        for j in range(1, len(path1) - 1):
            if problem.capacities[path0[i]] == problem.capacities[path1[j]]:
                if calculate_travel_node_dist(problem, path0[i], path1[j]) < 0.2:
                    path0[i], path1[j] = path1[j], path0[i]
                    num_exchanged += 1
                    break
    if num_exchanged >= 0:
        return [path0, path1]
    else:
        return []


def calculate_adjusted_distance_between_indices(problem, from_index, to_index):
    distance = calculate_travel_node_dist(problem, from_index, to_index)
    frequency = problem.get_frequency(from_index, to_index)
    return distance * (1.0 - frequency)


def sample_next_index(to_indices, adjusted_distances):
    if len(to_indices) == 0:
        return 0
    adjusted_probabilities = np.asarray([1.0 / max(d, EPSILON) for d in adjusted_distances])
    adjusted_probabilities /= np.sum(adjusted_probabilities)
    return np.random.choice(to_indices, p=adjusted_probabilities)


def pertubation_operator(problem, existing_solution, step):
    solution = []
    n = num_train_points
    customer_indices = list(range(n + 1))
    candidate_indices = []
    for path_index in range(len(existing_solution)):
        if len(existing_solution) > 2:
            candidate_indices.append(path_index)
    paths_ruined = np.random.choice(candidate_indices, NUM_PATHS_TO_RUIN, replace=False)
    start_customer_index = n + 1
    for path_index in paths_ruined:
        path = existing_solution[path_index]
        for customer_index in path:
            if customer_index == 0:
                continue
            start_customer_index -= 1
            customer_indices[start_customer_index] = customer_index
    
    if np.random.uniform() < 0.5:
        while len(solution) == 0:
            paths_ruined = np.random.choice(candidate_indices, NUM_PATHS_TO_RUIN, replace=False)
            solution = reconstruct_solution_by_exchange(problem, existing_solution, paths_ruined)
    else:
        trip = [0]
        capacit_left = problem.capacities[0]
        i = start_customer_index
        while i <= n:
            to_indices = []
            adjusted_distances = []
            for j in range(i, n + 1):
                if problem.capacities[customer_indices[j]] > capacit_left:
                    continue
                to_indices.append(j)
                adjusted_distances.append(
                    calculate_adjusted_distance_between_indices(problem, trip[-1], customer_indices[j]))
            random_index = sample_next_index(to_indices, adjusted_distances)

            if random_index == 0:
                trip.append(0)
                solution.append(trip)
                trip = [0]
                capacit_left = problem.capacities[0]
                continue
            customer_indices[i], customer_indices[random_index] = customer_indices[random_index], customer_indices[i]
            trip.append(customer_indices[i])
            capacit_left -= problem.capacities[customer_indices[i]]
            i += 1
        if len(trip) > 1:
            trip.append(0)
            solution.append(trip)
    
    while len(solution) < len(paths_ruined):
        solution.append([0, 0])
    improved_solution = copy.deepcopy(existing_solution)
    solution_index = 0
    for path_index in sorted(paths_ruined):
        improved_solution[path_index] = copy.deepcopy(solution[solution_index])
        solution_index += 1
    problem.mark_change_at(step, paths_ruined)
    for solution_index in range(len(paths_ruined), len(solution)):
        improved_solution.append(copy.deepcopy(solution[solution_index]))
        problem.mark_change_at(step, [len(improved_solution) - 1])
    
    has_seen_empty_path = False
    for path_index in range(len(improved_solution)):
        if len(improved_solution[path_index]) == 2:
            if has_seen_empty_path:
                empty_slot_index = path_index
                for next_path_index in range(path_index + 1, len(improved_solution)):
                    if len(improved_solution[next_path_index]) > 2:
                        improved_solution[empty_slot_index] = copy.deepcopy(improved_solution[next_path_index])
                        empty_slot_index += 1
                improved_solution = improved_solution[:empty_slot_index]
                problem.mark_change_at(step, range(path_index, empty_slot_index))
                break
            else:
                has_seen_empty_path = True
    return improved_solution


def env_step(step, problem, solution, action):
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
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [1, 2])
                    elif action == 9: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [1, 3])
                    elif action == 10: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [2, 1])
                    elif action == 11: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [2, 3])
                    elif action == 12: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [3, 1])
                    elif action == 13: 
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Asymmetric_exchange(problem, next_solution.path[path_first], next_solution.path[path_second], [3, 2])

                    elif action == 14:
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Relocate(problem, next_solution.path[path_first], next_solution.path[path_second], 1)
                    elif action == 15:
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Relocate(problem, next_solution.path[path_first], next_solution.path[path_second], 2)
                    elif action == 16:
                        next_solution.path[path_first], next_solution.path[path_second], label, delta = Relocate(problem, next_solution.path[path_first], next_solution.path[path_second], 3)
                    
                    elif action == 17:
                        next_solution, label, delta = Cyclic_exchange(problem, next_solution)
                    if label:
                        problem.mark_change_at(step, [path_first, path_second])
                        delta_sum += delta
                    else:
                        modified = False
                        problem.mark_no_improvement(step, action, path_first, path_second)
        return next_solution, delta_sum
    

