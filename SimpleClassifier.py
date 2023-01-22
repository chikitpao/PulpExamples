"""  
    Implemented data classifier (without and with penalties) from the video 
    'Machine Learning Classification Using Linear Programming?' by OptWhiz.
    Url: https://www.youtube.com/watch?v=QV_fG1UPQ50

    Author: Chi-Kit Pao
    REMARK: Requires library PuLP and Matplotlib to run this program.
    USAGE:
    Step 1: Put data into the list 'data' in function 'main'.
    Step 2: Change 'x_label', 'y_label', and 'forced_buffer' to proper values.
    Step 3: Run the program.
"""


from matplotlib import pyplot as plt
import pulp as plp


def calculate_result_wo_penalty(data: list[tuple[float, float, float]]) \
        -> tuple[float, float, float]:
    """ Returns a, b, delta. Can be None if no result is found. """

    # Objective: find maximum delta
    problem = plp.LpProblem(f'FindClassifier', plp.LpMaximize)
    lp_delta = plp.LpVariable('lp_delta', 0, None)
    problem += lp_delta
    # Other decision variables: a and b as in y = ax + b
    lp_a = plp.LpVariable('lp_a', None, None)
    lp_b = plp.LpVariable('lp_b', None, None)
    # Add Constraints:
    for d in data:
        if d[2] == 0:
            problem += lp_a * d[0] + lp_b - lp_delta >= d[1]
        else:
            problem += lp_a * d[0] + lp_b + lp_delta <= d[1]
    status = problem.solve(plp.PULP_CBC_CMD(msg=False))
    print(f'status: {status}')
    if status == 1:
        return lp_a.value(), lp_b.value(), lp_delta.value()
    else:
        return None, None, None

def calculate_result_with_penalty(data: list[tuple[float, float, float]], 
        forced_buffer: float) -> tuple[float, float, float]:
    """ Returns a, b, penalty. Can be None if no result is found. """

    # Objective: find minimum penalty (sum of single penalties)
    problem = plp.LpProblem(f'FindClassifier', plp.LpMinimize)    
    penalties = []
    for i in range(len(data)):
        penalties.append(plp.LpVariable(f'penalty{i}', 0, None))
    problem += sum(penalties)
    # Other decision variables: a and b as in y = ax + b
    lp_a = plp.LpVariable('lp_a', None, None)
    lp_b = plp.LpVariable('lp_b', None, None)
    # Add Constraints:
    for i, d in enumerate(data):
        if d[2] == 0:
            problem += d[1] - (lp_a * d[0] + lp_b - forced_buffer) <= penalties[i]
        else:
            problem += (lp_a * d[0] + lp_b + forced_buffer) - d[1] <= penalties[i]
    status = problem.solve(plp.PULP_CBC_CMD(msg=False))
    print(f'status: {status}')
    if status == 1:
        return lp_a.value(), lp_b.value(), problem.objective.value()
    else:
        return None, None, None


def plot_result(data: list[tuple[float, float]], labels: tuple[str, str], 
    values: tuple[float, float], delta: float, penalty: float = None):
    fig, axes = plt.subplots()

    # Unpack all x and y values
    all_x_passed, all_y_passed, passed = zip(*[x for x in data if x[2] != 0])
    all_x_failed, all_y_failed, passed = zip(*[x for x in data if x[2] == 0])
    max_x = max([x[0] for x in data])
    max_y = max([x[1] for x in data])

    # add grid lines
    axes.grid(True)
    axes.set_xlabel(labels[0])
    axes.set_ylabel(labels[1])
    axes.set_xlim(0, max_x + 1)
    axes.set_ylim(0, max_y + 1)
    # add lines at x=0, y=0 and labels
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    print(f'data: {data}')
    if penalty is None:
        result = f'a = {values[0]}, b = {values[1]}, delta = {delta}'
    else:
        result = f'a = {values[0]}, b = {values[1]}, delta = {delta}, penalty = {penalty}'
    print(f'result: {result}')
    plt.title(result)

    a, b = values
    y_end = a * (max_x + 1) + b
    plt.plot([0, max_x + 1], [b, y_end], '-', color='b')
    plt.plot([0, max_x + 1], [b-delta, y_end-delta], '--', color='b')
    plt.plot([0, max_x + 1], [b+delta, y_end+delta], '--', color='b')
    axes.scatter(all_x_passed, all_y_passed, s=20, facecolor='g', edgecolor='g')
    axes.scatter(all_x_failed, all_y_failed, s=20, facecolor='r', edgecolor='r')

    # Draw penalties:
    if penalty is not None:
        for i, d in enumerate(data):
            if d[2] == 0:
                set_value = values[0] * d[0] + values[1] - delta
                if d[1] > set_value:
                    plt.plot([d[0], d[0]], [d[1], set_value], '--', color='r')
            else:
                set_value = values[0] * d[0] + values[1] + delta
                if d[1] < set_value:
                    plt.plot([d[0], d[0]], [d[1], set_value], '--', color='g')

    fig.canvas.manager.set_window_title('Simple Classifier')
    plt.show()


def main():
    # data is a list of tuples with (x, y, b)
    # b: 1 if passed, 0 if failed
    data = [ (94.5, 7.5, 1),
        (70.0, 6.9, 1),
        (83.0, 9.4, 1),
        (82.7, 10.3, 1),
        (81.5, 6.9, 1),
        (66.0, 2.0, 0),
        (79.6, 1.8, 0),
        (54.7, 3.4, 0),
        (61.3, 5.1, 0),
        (43.9, 7.5, 0),
        # points which prevent clear classification
        (30.0, 9.5, 1),
        (90.0, 1.0, 1),
        ]
    x_label = 'Grade Before Exam'
    y_label = 'Hours Studied'
    # buffer used when penalty is expected
    forced_buffer: float = 1.0

    a, b, delta = calculate_result_wo_penalty(data)
    if delta is not None:
        plot_result(data, (x_label, y_label), (a, b), delta)
    else:
        a, b, penalty = calculate_result_with_penalty(data, forced_buffer)
        plot_result(data, (x_label, y_label), (a, b), forced_buffer, penalty)


if __name__ == '__main__':
    main()
