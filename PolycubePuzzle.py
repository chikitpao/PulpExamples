"""  
    Solves polycube puzzle either with identical pieces or different pieces.

    Author: Chi-Kit Pao
    REMARK: Requires library PuLP, Matplotlib and NumPy to run this program.
    USAGE:
    # 1: Solving puzzle with identical pieces
    # - Change the puzzle of variable "raw_polycube" in function "main", if necessary.
    # - Run the program either without option or with option "--single".
    # 2: Solving puzzle with different pieces
    # - Change the puzzle of variable "raw_polycubes" in function "main", if necessary.
    # - Run the program with option "--multiple".
"""

import itertools
from matplotlib import pyplot as plt
import numpy as np
import pulp as plp
import sys
import time


def build_polycube(raw_polycube:list) -> np.ndarray:
    result = np.zeros((3, 3, 3), int)
    # Output: type(result)=<class 'numpy.ndarray'>
    # print(f"{type(result)=}")
    for row, line in enumerate(raw_polycube):
        for col, v in enumerate(line):
            for layer in range(3):
                result[layer, row, col] = int((v & (1 << layer)) != 0)
    return result

def get_rotations(polycube:np.ndarray) -> list[np.ndarray]:
    # REMARK: 'numpy.ndarray' is not hashable, so put it into a
    # set will cause error:
    # TypeError: unhashable type: 'numpy.ndarray'
    result = list()
    pc = polycube
    # "Up" position (= z-axis)
    result.append(polycube)
    # Other 3 orientations of "up" position: Rotate around z-axis.
    result.append(np.rot90(pc, k=1, axes=(1,2)))
    result.append(np.rot90(pc, k=2, axes=(1,2)))
    result.append(np.rot90(pc, k=-1, axes=(1,2)))

    # "Left" position: Rotate left around y-axis. Original up-axis (= z-axis) has rotated to (negative) x-axis.
    left_pc = np.rot90(pc, k=1, axes=(1,0))
    result.append(left_pc)
    # Other 3 orientations: Rotate around x-axis.
    result.append(np.rot90(left_pc, k=1, axes=(2,0)))
    result.append(np.rot90(left_pc, k=2, axes=(2,0)))
    result.append(np.rot90(left_pc, k=-1, axes=(2,0)))

    # "Right" position: Rotate right around y-axis. Original up-axis (= z-axis) has rotated to x-axis.
    right_pc = np.rot90(pc, k=-1, axes=(1,0))
    result.append(right_pc)
    # Other 3 orientations: Rotate around x-axis.
    result.append(np.rot90(right_pc, k=1, axes=(2,0)))
    result.append(np.rot90(right_pc, k=2, axes=(2,0)))
    result.append(np.rot90(right_pc, k=-1, axes=(2,0)))

    # "Down" position: Rotate 180° around y-axis. Original up-axis (= z-axis) has rotated to (negative) z-axis.
    down_pc = np.rot90(pc, k=2, axes=(1,0))
    result.append(down_pc)
    # Other 3 orientations: Rotate around z-axis.
    result.append(np.rot90(down_pc, k=1, axes=(1,2)))
    result.append(np.rot90(down_pc, k=2, axes=(1,2)))
    result.append(np.rot90(down_pc, k=-1, axes=(1,2)))

    # "Forward position": Rotate right around x-axis. Original up-axis (= z-axis) has rotated to y-axis.
    forward_pc = np.rot90(pc, k=-1, axes=(2,0))
    result.append(forward_pc)
    # Other 3 orientations: Rotate around y-axis.
    result.append(np.rot90(forward_pc, k=1, axes=(1,0)))
    result.append(np.rot90(forward_pc, k=2, axes=(1,0)))
    result.append(np.rot90(forward_pc, k=-1, axes=(1,0)))

    # "Backward position": Rotate left around x-axis. Original up-axis (= z-axis) has rotated to (negative) y-axis.
    backward_pc = np.rot90(pc, k=1, axes=(2,0))
    result.append(backward_pc)
    # Other 3 orientations: Rotate around y-axis.
    result.append(np.rot90(backward_pc, k=1, axes=(1,0)))
    result.append(np.rot90(backward_pc, k=2, axes=(1,0)))
    result.append(np.rot90(backward_pc, k=-1, axes=(1,0)))

    return result

def get_transformations(polycube:np.ndarray) -> list[np.ndarray]:
    # REMARK: 'numpy.ndarray' is not hashable, so put it into a
    # set will cause error:
    # TypeError: unhashable type: 'numpy.ndarray'
    temp_result = []
    rotations = get_rotations(polycube)
    for rotation in rotations:
        x_sums = [rotation[:, i, :].sum() for i in range(3)]
        x_back = len(list(itertools.takewhile(lambda x: x == 0, x_sums)))
        x_zero_count = len(list(filter(lambda x: x == 0, x_sums)))

        y_sums = [rotation[:, :, i].sum() for i in range(3)]
        y_back = len(list(itertools.takewhile(lambda x: x == 0, y_sums)))
        y_zero_count = len(list(filter(lambda x: x == 0, y_sums)))

        z_sums = [rotation[i, :, :].sum() for i in range(3)]
        z_back = len(list(itertools.takewhile(lambda x: x == 0, z_sums)))
        z_zero_count = len(list(filter(lambda x: x == 0, z_sums)))

        # Shift to the lowest possible position
        rotation_mod = np.roll(rotation, (-z_back, -x_back, -y_back), (0, 1, 2))

        # Try all possible translations
        # REMARK: Translations will not cause duplicates here.
        for dx in range(x_zero_count+1):
            for dy in range(y_zero_count+1):
                for dz in range(z_zero_count+1):
                    temp_result.append(np.roll(rotation_mod, (dz, dx, dy), (0, 1, 2)))

    # Remove duplicates
    result = []
    for candidate in temp_result:
        if all(not np.array_equal(M, candidate) for M in result):
            result.append(candidate)

    return result

def plot_polycube(fig, ax, i, pc):
    r = [0,1]
    alpha_ = 0.1
    wire_color = 'k'
    linewidth_ = 1
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'pink', 'w']
    color_ = colors[i]

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    for posx, posy, posz in itertools.product(list(range(3)), repeat=3):
        if pc[posz, posx, posy] != 1:
            continue

        X, Y = np.meshgrid(r, r)
        one = np.array([[1, 1]])
        X_ = X+posx
        Y_ = Y+posy
        Z1_ = X+posz
        Z2_ = Y+posz

        # REMARK: Don't draw inner surfaces.
        # bottom and top
        if posz == 0 or pc[posz-1][posx][posy] != 1:
            ax.plot_surface(X_, Y_, one*posz, alpha=alpha_, color=color_)
        ax.plot_wireframe(X_, Y_, one*posz, color=wire_color, linewidth=linewidth_)
        if posz == 2 or pc[posz+1][posx][posy] != 1:
            ax.plot_surface(X_, Y_, one*(posz+1), alpha=alpha_, color=color_)
        ax.plot_wireframe(X_, Y_, one*(posz+1), color=wire_color, linewidth=linewidth_)
        # left and right
        if posx == 0 or pc[posz][posx-1][posy] != 1:
            ax.plot_surface(one*posx, Y_, Z1_, alpha=alpha_, color=color_)
        ax.plot_wireframe(one*posx, Y_, Z1_, color=wire_color, linewidth=linewidth_)
        if posx == 2 or pc[posz][posx+1][posy] != 1:
            ax.plot_surface(one*(posx+1), Y_, Z1_, alpha=alpha_, color=color_)
        ax.plot_wireframe(one*(posx+1), Y_, Z1_, color=wire_color, linewidth=linewidth_)
        # front and back
        if posy == 0 or pc[posz][posx][posy-1] != 1:
            ax.plot_surface(X_, one*posy, Z2_, alpha=alpha_, color=color_)
        ax.plot_wireframe(X_, one*posy, Z2_, color=wire_color, linewidth=linewidth_)
        if posy == 2 or pc[posz][posx][posy+1] != 1:
            ax.plot_surface(X_, one*(posy+1), Z2_, alpha=alpha_, color=color_)
        ax.plot_wireframe(X_, one*(posy+1), Z2_, color=wire_color, linewidth=linewidth_)

def plot_result(result):
    if not result:
        return
    length = len(result)

    d, m = divmod(length, 3)
    row_count = d if m == 0 else (d + 1)
    fig = plt.figure()
    #fig, axes = plt.subplots(row_count, 3)
    #print(f"{fig=}")
    #print(f"{axes=}")
    for i, pc in enumerate(result):
        ax = fig.add_subplot(row_count, 3, i + 1, projection='3d')
        # ax.set_aspect('equal', adjustable='box')
        ax.set_box_aspect([1,1,1])
        ax.set_xlim3d([0.0, 3.25])
        ax.set_ylim3d([0.0, 3.25])
        ax.set_zlim3d([0.0, 3.25])
        plot_polycube(fig, ax, i, pc)

    title = "Polycube Puzzle"
    fig.suptitle(title)
    fig.canvas.manager.set_window_title(title)
    plt.show()

def solve_single(raw_polycube):
    polycube = build_polycube(raw_polycube)
    transformations = get_transformations(polycube)

    print(f"transformation count: {len(transformations)}")
    polycube_count, r = divmod(27, np.count_nonzero(polycube))
    assert r == 0

    # Objective: Find exactly 9 transformations of a polycube piece to form a 3x3 cube.
    problem = plp.LpProblem(f'FindSolution', plp.LpMaximize)
    lp_sum = plp.LpVariable('lp_sum', 0, polycube_count, cat = "Integer")
    problem += lp_sum

    # Other decision variables: transformation used
    lp_transform = [plp.LpVariable(f'lp_tr{i}', 0, 1, cat = "Binary") for i in range(len(transformations))]

    # Constraints:
    # 1: Sum is the number of transformations used and must be <= polycube_count.
    problem += lp_sum == sum(lp_transform)
    problem += lp_sum <= polycube_count
    # 2: Examine each sub-cube of the 3x3 cube separately. Express its value as the sum of transformations
    # which occupies that sub-cube. The sum must be exactly 1 since each sub-cube can be occupied by only
    # one transformation.
    for i in range(27):
        used = []
        for tr_no, tr in enumerate(transformations):
            if tr[i // 9, (i // 3) % 3, i % 3] == 1:
                used.append(lp_transform[tr_no])
        problem += 1 == sum(used)

    status = problem.solve(plp.PULP_CBC_CMD(msg=False))
    print(f'status: {status}')

    # Remark: Use either problem.objective.value() or lp_sum.value() since they 
    # are the same.
    if status != 1:
        print(f"No solution found!")
        return []
    else:  # (1 = OPTIMAL)
        print(f"{lp_sum.name=}")
        print(f"{lp_sum.value()=}")
        print(f"{problem.objective.name=}")
        print(f"{problem.objective.value()=}")
        assert lp_sum.value() == polycube_count
        
        result = []
        for i, tr in enumerate(lp_transform):
            if lp_transform[i].value() > 0:
                print(i, lp_transform[i].value())
                result.append(transformations[i])

        print("Result:")
        print(result)
        print(sum([i * r for i, r in enumerate(result)]))
        return result

def solve_multiple(raw_polycubes):
    polycubes = [build_polycube(rp) for rp in raw_polycubes]

    transformations = []
    for i, pc in enumerate(polycubes):
        transformations.append(get_transformations(pc))

    print(f"transformation count: {sum(map(len, transformations))}")
    print(f"{list(map(len, transformations))}")

    polycube_count = len(polycubes)

    # Objective: Find transformation for every polycube so they form a 3x3 cube.
    problem = plp.LpProblem(f'FindSolution', plp.LpMaximize)
    lp_sum = plp.LpVariable('lp_sum', 0, polycube_count, cat = "Integer")
    problem += lp_sum

    # Other decision variables:
    lp_transform_used = []
    for i, transformation_list in enumerate(transformations):
        # index of transformation used by every polycube, i = polycube, j = index of transformation
        lp_transform_used.append([plp.LpVariable(f'used{i}_{j}', 0, 1, cat = "Binary") for j in range(len(transformation_list))])

    # Constraints:
    # 1: Sum is the number of transformations used and must be <= polycube_count.
    problem += lp_sum == sum([b for a in lp_transform_used for b in a])
    problem += lp_sum <= polycube_count
    # 2: We can only use one transformation of every polycube.
    for pc_no, pc_used_transform_list in enumerate(lp_transform_used):
        problem += 1 == sum(pc_used_transform_list)
    # 3: Examine each sub-cube of the 3x3 cube separately. Express its value as the sum of polycubes
    # which occupies that sub-cube. The sum must be exactly 1 since each sub-cube can be occupied by only
    # one polycube.
    for i in range(27):
        used = []
        for pc_no, tr_list in enumerate(transformations):
            for tr_no, tr in enumerate(tr_list):
                if tr[i // 9, (i // 3) % 3, i % 3] == 1:
                    used.append(lp_transform_used[pc_no][tr_no])
        problem += 1 == sum(used)
    
    status = problem.solve(plp.PULP_CBC_CMD(msg=False))
    print(f'status: {status}')

    # Remark: Use either problem.objective.value() or lp_sum.value() since they 
    # are the same.
    if status != 1:
        print(f"No solution found!")
        return []
    else:  # (1 = OPTIMAL)
        print(f"{lp_sum.name=}")
        print(f"{lp_sum.value()=}")
        print(f"{problem.objective.name=}")
        print(f"{problem.objective.value()=}")
        assert lp_sum.value() == polycube_count
        
        result = []
        for i, used in enumerate(lp_transform_used):
            for j, transform in enumerate(used):
                if transform.value() > 0:
                    print(i, j, transform.value())
                    result.append(transformations[i][j])

        print("Result:")
        print(result)
        print(sum([i * r for i, r in enumerate(result)]))
        return result

def main():
    start_time = time.time()
    single = True

    for arg in sys.argv[1:]:
        if arg == "--single":
            single = True
        elif arg == "--multiple":
            single = False
        else:
            print("Invalid argument!")
            return

    result = []
    if single:
        # Polycubes are described in top view.
        # Bottom -> Bit 0
        # Middle -> Bit 1
        # Top -> Bit 2
        raw_polycube = [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0]]
        result = solve_single(raw_polycube)
    else:
        # Polycubes are described in top view.
        # Bottom -> Bit 0
        # Middle -> Bit 1
        # Top -> Bit 2
        raw_polycubes = [
            [[1, 1, 0],
            [3, 0, 0],
            [1, 0, 0]],
            [[1, 0, 0],
            [1, 1, 0],
            [1, 0, 0]],
            [[1, 3, 0],
            [0, 1, 1],
            [0, 0, 0]],
            [[1, 0, 0],
            [1, 3, 0],
            [1, 0, 0]],
            [[1, 3, 0],
            [1, 0, 0],
            [0, 0, 0]],
            [[1, 1, 0],
            [0, 1, 0],
            [0, 1, 0]]
        ]
        result = solve_multiple(raw_polycubes)
    
    print(f"Time elapsed: {time.time() - start_time} s")
    
    plot_result(result)

if __name__ == '__main__':
    main()

##### Output (option "--single"):
# transformation count: 144
# status: 1
# 9.0
# 3 1.0
# 18 1.0
# 57 1.0
# 75 1.0
# 84 1.0
# 105 1.0
# 118 1.0
# 133 1.0
# 143 1.0
# Result:
# [array([[[0, 1, 1],
#         [0, 1, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [1, 0, 0],
#         [1, 1, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [1, 1, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [1, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 1],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 1, 1],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[1, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[1, 1, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 1, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 1, 0],
#         [0, 1, 0]]]), array([[[0, 0, 0],
#         [0, 0, 1],
#         [0, 0, 1]],

#        [[0, 0, 0],
#         [0, 0, 1],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [1, 0, 0],
#         [0, 0, 0]],

#        [[1, 0, 0],
#         [1, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 1]],

#        [[0, 0, 0],
#         [0, 0, 1],
#         [0, 0, 1]]])]
# [[[4 0 0]
#   [1 0 6]
#   [1 1 6]]

#  [[4 4 3]
#   [7 5 6]
#   [2 2 8]]

#  [[7 3 3]
#   [7 5 8]
#   [2 5 8]]]
# Time elapsed: 0.18631386756896973 s

##### Output (option --multiple):
# transformation count: 600
# [96, 72, 96, 96, 96, 144]
# status: 1
# 9.0
# 0 72 1.0
# 1 42 1.0
# 2 95 1.0
# 3 31 1.0
# 4 89 1.0
# 5 75 1.0
# Result:
# [array([[[1, 0, 0],
#         [1, 1, 0],
#         [1, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [1, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 1, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[1, 1, 1],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 1],
#         [0, 0, 1]],

#        [[0, 0, 1],
#         [0, 1, 1],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 1, 0],
#         [0, 1, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [1, 1, 1]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [1, 0, 0],
#         [0, 0, 0]],

#        [[1, 1, 0],
#         [1, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 1],
#         [0, 0, 1],
#         [0, 1, 1]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]])]
# [[[0 1 5]
#   [0 0 5]
#   [0 5 5]]

#  [[1 1 1]
#   [4 3 2]
#   [0 3 2]]

#  [[4 4 2]
#   [4 2 2]
#   [3 3 3]]]
# Time elapsed: 0.45742034912109375 s
