"""  
    Solves polycube puzzle either with identical pieces or different pieces.

    Author: Chi-Kit Pao
    REMARK: Requires library PuLP and NumPy to run this program.
    USAGE:
    # 1: Solving puzzle with identical pieces
    # - Change the puzzle of variable "raw_polycube" in function "main", if necessary.
    # - Run the program either without option or with option "--single".
    # 2: Solving puzzle with different pieces
    # - Change the puzzle of variable "raw_polycubes" in function "main", if necessary.
    # - Run the program with option "--multiple".
"""

import itertools
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

def solve_single(raw_polycube):
    polycube = build_polycube(raw_polycube)
    transformations = get_transformations(polycube)

    print(f"transformation count: {len(transformations)}")

    # Objective: Find exactly 9 transformations of a polycube piece to form a 3x3 cube.
    problem = plp.LpProblem(f'FindSolution', plp.LpMaximize)
    lp_sum = plp.LpVariable('lp_sum', 0, 9, cat = "Integer")
    problem += lp_sum

    # Other decision variables: transformation used
    lp_transform = [plp.LpVariable(f'lp_tr{i}', 0, 1, cat = "Binary") for i in range(len(transformations))]

    # Constraints:
    # 1: Sum is the number of transformations used and must be <= 9.
    lp_sum = sum(lp_transform)
    problem += lp_sum <= 9
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
    else:  # (1 = OPTIMAL)
        print(f"{lp_sum.value()}")
        assert lp_sum.value() == 9.0
        
        result = []
        for i, tr in enumerate(lp_transform):
            if lp_transform[i].value() > 0:
                print(i, lp_transform[i].value())
                result.append(transformations[i])

        print("Result:")
        print(result)
        print(sum([i * r for i, r in enumerate(result)]))

def solve_multiple(raw_polycubes):
    polycubes = [build_polycube(rp) for rp in raw_polycubes]

    transformations = []
    for i, pc in enumerate(polycubes):
        transformations.append(get_transformations(pc))

    print(f"transformation count: {sum(map(len, transformations))}")
    print(f"{list(map(len, transformations))}")

    # Objective: Find transformation for every polycube so they form a 3x3 cube.
    problem = plp.LpProblem(f'FindSolution', plp.LpMaximize)
    lp_sum = plp.LpVariable('lp_sum', 0, 9, cat = "Integer")
    problem += lp_sum

    # Other decision variables:
    lp_transform_used = []
    # lp_transform = []
    for i, transformation_list in enumerate(transformations):
        # index of transformation used by every polycube, i = polycube, j = index of transformation
        lp_transform_used.append([plp.LpVariable(f'used{i}_{j}', 0, 1, cat = "Binary") for j in range(len(transformation_list))])
        # # possible transformations for every polycube
        # lp_transform.append([plp.LpVariable(f'tr{i}_{j}', 0, 1, cat = "Binary") for j in range(len(transformation_list))])

    # Constraints:
    # 1: Sum is the number of transformations used and must be <= 9.
    for used in lp_transform_used:
        lp_sum += sum(used)
    problem += lp_sum <= 9
    # 2: We can only use one transformation of every polycube.
    for pc_no, pc_used_transform_list in enumerate(lp_transform_used):
        problem += 1 == sum(pc_used_transform_list)
    # 2: Examine each sub-cube of the 3x3 cube separately. Express its value as the sum of polycubes
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
    else:  # (1 = OPTIMAL)
        print(f"{lp_sum.value()}")
        assert lp_sum.value() == 9.0
        
        result = []
        for i, used in enumerate(lp_transform_used):
            for j, transform in enumerate(used):
                if transform.value() > 0:
                    print(i, j, transform.value())
                    result.append(transformations[i][j])

        print("Result:")
        print(result)
        print(sum([i * r for i, r in enumerate(result)]))

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

    if single:
        # Polycubes are described in top view.
        # Bottom -> Bit 0
        # Middle -> Bit 1
        # Top -> Bit 2
        raw_polycube = [
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 0]]
        solve_single(raw_polycube)
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
        solve_multiple(raw_polycubes)

    print(f"Time elapsed: {time.time() - start_time} s")

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
