"""  
    Solves Sudoku puzzle.

    Author: Chi-Kit Pao
    REMARK: Requires library PuLP and Matplotlib to run this program.
    USAGE:  Either change the variable "sudoku" or specify the input 
            file name as program parameter. Put 0 for unknown numbers.
"""

import itertools
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
import numpy as np
import sys
import pulp as plp


sudoku = ["030000000",
          "000195000",
          "008000060",
          "800060000",
          "400800001",
          "000020000",
          "060000280",
          "000419005",
          "000000070"]

class Sudoku:
    def __init__(self, lines):
        self.lines = []
        self.input_coordinates = set()
        for line in lines:
            self.lines.append(list(map(lambda c: ord(c) - ord('0'), line)))

    def solve(self):
        # Use linear programming to find a possible solution of a given Sudoku puzzle.
        # The implementation here is loosely based on the consideration from the website
        # https://www.coin-or.org/PuLP/CaseStudies/a_sudoku_problem.html

        # Objective: Not really needed. Can use default sense (=const.LpMinimize).
        problem = plp.LpProblem(f'FindSolution')

        # Objective function not really needed.

        # Decision variables: row, column and value
        lp_var_cells = []
        for i in range(0, 9):
            row_list = []
            for j in range(0, 9):
                cell_list = [plp.LpVariable(f'x{i}_{j}_{v}', 0, 1, cat=plp.const.LpBinary) for v in range(1, 10)]
                row_list.append(cell_list)
            lp_var_cells.append(row_list)

        for i in range(0, 9):
            for j in range(0, 9):
                input_value = self.lines[i][j]
                if input_value != 0:
                    self.input_coordinates.add((i, j))
                    # Add constraint: Set lp variables of known values
                    for v in range(1, 10):
                        problem += lp_var_cells[i][j][v-1] == (1 if v == input_value else 0)

                # Add constraint: Only one variable of a cell is one and others are zero.
                problem += plp.lpSum([lp_var_cells[i][j][v-1] for v in range(1, 10)]) == 1

        # Add further constraints
        # Row Constraint: Every number only occurs once in a row.
        for i in range(0, 9):
            for v in range(1, 10):
                problem += plp.lpSum([lp_var_cells[i][j][v-1] for j in range(0, 9)]) == 1
        # Column Constraint: Every number only occurs once in a column.
        for j in range(0, 9):
            for v in range(1, 10):
                problem += plp.lpSum([lp_var_cells[i][j][v-1] for i in range(0, 9)]) == 1
        # Constraint for subsquare: Every number only occurs once in a sub-square.
        for i, j in itertools.product((0, 3, 6), repeat=2):
            for v in range(1, 10):
                l = [lp_var_cells[i+di][j+dj][v-1] for di, dj in itertools.product(list(range(3)), repeat=2)]
                problem += plp.lpSum(l) == 1

        status = problem.solve(plp.PULP_CBC_CMD(msg=False))
        if status != 1:
            raise RuntimeError(f"No solution found for Sudoku! Pulp problem status: {status}!")

        self.__write_back(lp_var_cells)

    def __write_back(self, lp_var_cells):
        for i, j in itertools.product(range(0, 9), repeat=2):
            v = len(list(itertools.takewhile(lambda x: x.value() == 0, lp_var_cells[i][j]))) + 1
            if self.lines[i][j] == 0:
                assert 1 <= v <= 9, f"{i=}, {j=}, {v=}"
                self.lines[i][j] = v
            else:
                assert self.lines[i][j] == v
                pass

def plot_sudoku(s):
    fig, ax = plt.subplots()
    max_num = 9
    for i, j in itertools.product(list(range(0,max_num)), repeat=2):
        color_ = 'k' if (i, j) in s.input_coordinates else 'r'
        ax.text(i+0.5,j+0.5,str(s.lines[8-j][i]), ha='center', va='center', color=color_)

    ax.axis([0, max_num, 0, max_num])
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_visible(False)
        axis.set_ticks(np.arange(max_num) + 0.5)
        axis.set_ticklabels("")

    for i in range(max_num):
        if i % 3 != 0:
            ax.axhline(y=(i), color='tab:blue')
            ax.axvline(x=(i), color='tab:blue')
    for i in (3, 6):
        ax.axhline(y=(i), color='k', linewidth=3)
        ax.axvline(x=(i), color='k', linewidth=3)

    title = "Sudoku"
    fig.suptitle(title)
    fig.canvas.manager.set_window_title(title)
    plt.show()

def main():
    global sudoku
    if len(sys.argv) == 2:
        with open(sys.argv[1], 'r') as f:
            sudoku = list(map(lambda line: line.replace('\n', ''), f.readlines()))
    # else keep internal variable the same
    
    s = Sudoku(sudoku)
    s.solve()
    for line in s.lines:
        print(line)

    plot_sudoku(s)

if __name__ == '__main__':
    main()
