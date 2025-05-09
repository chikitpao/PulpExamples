"""  
    Solves polycube puzzle with identical pieces.

    Author: Chi-Kit Pao
    REMARK: Requires library PuLP and NumPy to run this program.
"""

import itertools
import numpy as np
import pulp as plp
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

def main():
    start_time = time.time()

    # Polycubes are described in top view.
    # Bottom -> Bit 0
    # Middle -> Bit 1
    # Top -> Bit 2
    raw_polycube = [
         [1, 1, 0],
         [1, 0, 0],
         [0, 0, 0]]

    polycube = build_polycube(raw_polycube)
    transformations = get_transformations(polycube)

    print(f"transformation count: {len(transformations)}")

    # Objective: Find exactly 9 transformations of a polycube piece to form a 3x3 cube.
    problem = plp.LpProblem(f'FindSolution', plp.LpMaximize)
    lp_sum = plp.LpVariable('lp_sum', 9, 9)
    problem += lp_sum

    # Other decision variables: transformation used
    lp_transfrom = [plp.LpVariable(f'lp_tr{i}', 0, 1, cat = "Binary") for i in range(len(transformations))]

    # Constraints:
    # 1: Sum is the number of transformations used.
    lp_sum = sum(lp_transfrom)
    # 2: Examine each sub-cube of the 3x3 cube separately. Express its value as the sum of transformations
    # which occupies that sub-cube. The sum must be exactly 1 since each sub-cube can be occupied by only
    # one transformation.
    for i in range(27):
        used = []
        for tr_no, tr in enumerate(transformations):
            if tr[i // 9, (i // 3) % 3, i % 3] == 1:
                used.append(lp_transfrom[tr_no])
        problem += 1 == sum(used)
    status = problem.solve(plp.PULP_CBC_CMD(msg=False))
    
    print(f'status: {status}')
    # Remark: Use either problem.objective.value() or lp_sum.value() since they 
    # are the same.
    if status != 1:
        print("No solution found!")
    else:  # (1 = OPTIMAL)
        print(f"{lp_sum.value()}")
        assert lp_sum.value() == 9.0
        
        result = []
        for i, tr in enumerate(lp_transfrom):
            if lp_transfrom[i].value() > 0:
                print(i, lp_transfrom[i].value())
                result.append(transformations[i])

        print("Result:")
        print(result)
        print(sum([i * r for i, r in enumerate(result)]))
    
    print(f"Time elapsed: {time.time() - start_time} s")

if __name__ == '__main__':
    main()

# Output:
# transformation count: 144
# status: 1
# 9.0
# 20 1.0
# 27 1.0
# 47 1.0
# 48 1.0
# 56 1.0
# 73 1.0
# 82 1.0
# 88 1.0
# 113 1.0
# Result:
# [array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [1, 0, 0],
#         [1, 1, 0]]]), array([[[0, 0, 1],
#         [0, 1, 1],
#         [0, 0, 0]],

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
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 1, 1],
#         [0, 0, 1]]]), array([[[1, 1, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[1, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [1, 1, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [1, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 1, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[1, 1, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 1]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 1, 1]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [1, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [1, 1, 0],
#         [0, 0, 0]],

#        [[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]]]), array([[[0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0]],

#        [[0, 0, 1],
#         [0, 0, 1],
#         [0, 0, 0]],

#        [[0, 0, 1],
#         [0, 0, 0],
#         [0, 0, 0]]])]
# [[[3 3 1]
#   [7 1 1]
#   [4 4 6]]

#  [[3 5 8]
#   [7 7 8]
#   [4 6 6]]

#  [[5 5 8]
#   [0 2 2]
#   [0 0 2]]]
# Time elapsed: 0.23205232620239258 s
