"""  
    Solves problem in the video 'What in the world is a linear program?' by 
    OptWhiz.
    Url: https://www.youtube.com/watch?v=ceE96L4XWVE

    Author: Chi-Kit Pao
    REMARK: Requires library PuLP and Matplotlib to run this program.
    REMARK: Might not work for concave polygons at the moment.
    USAGE: Put coordinates of the polygon into the list 'points' in function 
        'main' and then run the program.
"""

import math
from matplotlib import pyplot as plt
import pulp as plp


def is_polygon_ccw(points: list[float]) -> bool:
    # Uses algorithm from "How to determine if a list of polygon points are in clockwise order?"
    # https://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order
    sum = 0
    for i in range(len(points) - 1):
        sum += (points[i + 1][0] - points[i][0]) * (points[i + 1][1] 
            + points[i][1])
    return sum < 0


def get_constraint_from_edge(point1: tuple[float, float], 
                            point2: tuple[float, float], 
                            ccw: bool,
                            vars: list[plp.pulp.LpVariable]) \
                            -> 'plp.pulp.LpConstraint':
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    if ccw:
        n = (dx + dy * 1j) * (1j)
    else:
        n = (dx + dy * 1j) * (-1j)
    n_norm = math.sqrt(n.real**2 + n.imag**2)
    # Constraint: (C-P1) dot n / n_norm >= r
    # C: Center of circle (u, v)
    # P1: Point 1 (x1, y1)
    # => (n.real * (u - x1) + n.imag * (v - y1)) / n_norm >= r
    # => (n.real * u - n.real * x1 + n.imag * v - n.imag * y1) / n_norm >= r
    # => n.real * u + n.imag * v - n_norm * r >= n.real * x1 + n.imag * y1
    return n.real * vars[0] + n.imag * vars[1] - n_norm * vars[2] >= n.real \
        * point1[0] + n.imag * point1[1]

def calculate_result(points: list[float]) -> tuple[float, float, float]:
    """ Returns u, v, r. Can be None if no result is found. """
    # Close the polygon
    points.append(points[0])

    # Objective: find maximum radius
    problem = plp.LpProblem(f'FindRadius', plp.LpMaximize)
    lp_r = plp.LpVariable('lp_r', 0, None)
    problem += lp_r
    # Other decision variables: x- and y-coordinates of center
    lp_u = plp.LpVariable('lp_u', 0, None)
    lp_v = plp.LpVariable('lp_v', 0, None)
    # Constraints:
    # #1 Center is inside the polygon AND distances from the center to 
    #   lines containing each edge is at least r.
    ccw: bool = is_polygon_ccw(points)
    vars = [lp_u, lp_v, lp_r]
    for i in range(len(points) - 1):
        problem += get_constraint_from_edge(points[i], points[i + 1], ccw, 
            vars)
    # #2 Radius is larger than or equal to 0.
    problem += lp_r >= 0
    status = problem.solve(plp.PULP_CBC_CMD(msg=False))
    print(f'status: {status}')
    # Remark: Use either problem.objective.value() or lp_r.value() since they 
    # are the same.
    if status == 1:
        # REMARK: In spite of status == 1 (OPTIMAL), these values might be 
        # None (e.g. when points construct a line) 
        return lp_u.value(), lp_v.value(), lp_r.value()
    else:
        return None, None, None


def plot_result(points: list[float], u: float = None, v: float = None, r: float = None):
    fig, axes = plt.subplots()

    # Unpack all x and y values
    all_x, all_y = zip(*points)

    # add grid lines
    axes.grid(True)
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.set_xlim(min(all_x)-1, max(all_x)+1)
    axes.set_ylim(min(all_y)-1, max(all_y)+1)
    axes.set_aspect('equal')
    # add lines at x=0, y=0 and labels
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')

    # Example output
    # points: [(0, 0), (5, 0), (6, 2), (3, 5), (-1, 3), (0, 0)]
    # u = 3.0, v = 2.0710678, r = 2.0710678
    print(f'points: {points}')
    if u is None:
        result = f'No result found!'
    else:
        result = f'u = {u}, v = {v}, r = {r}'
        circle = plt.Circle((u, v), r, color='g')
        axes.add_patch(circle)
    print(result)
    plt.title(result)

    plt.plot(all_x, all_y)
    fig.canvas.set_window_title('Circle in Polygon')
    plt.show()


def main():
    # points of polygon
    points = [ (0, 0),
        (5, 0),
        (6, 2),
        (3, 5),
        (1, 3)
        ]

    u, v, r = calculate_result(points)
    plot_result(points, u, v, r)


if __name__ == '__main__':
    main()
