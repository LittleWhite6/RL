def get_color(curr_color):
    if curr_color == 0:
        return 'b'
    if curr_color == 1:
        return 'g'
    if curr_color == 2:
        return 'r'
    if curr_color == 3:
        return 'c'
    if curr_color == 4:
        return 'm'
    if curr_color == 5:
        return 'y'
    if curr_color == 6:
        return 'k'
    return 'r'


def paint(problem, solution):
    for i in range(len(problem.locations)):  # 坐标放大
        problem.locations[i][0] *= 1000
        problem.locations[i][1] *= 1000
    x = []
    y = []
    curr_color = 0
    for i in range(len(solution)):
        for j in range(len(solution[i])):
            x.append(problem.locations[solution[i][j]][0])
            y.append(problem.locations[solution[i][j]][1])
            # one route exist
        for index in range(1, len(x)):
            l_x = []
            l_y = []
            l_x.append(x[index-1])
            l_x.append(x[index])
            l_y.append(y[index-1])
            l_y.append(y[index])
            plt.plot(l_x, l_y, color=get_color(curr_color))
            plt.scatter(l_x, l_y, c='black')
        x = []
        y = []
        curr_color += 1
        if curr_color > 6:
            curr_color = 0
    plt.show()
    plt.cla()