#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
python解方程
"""

from scipy.optimize import fsolve
from math import sin,cos

part=4

if part==1:
    # 求解非线性方程组
    def solve_function(unsolved_value):
        x=unsolved_value[0]
        return [
            sin(x)-0.5
        ]

    solved=fsolve(solve_function,[3.14])
    print(solved)
    solved=fsolve(solve_function,[0])
    print(solved)
    print("Program done!")

if part==2:
    # 求解三元二次方程组
    def solve_function(unsolved_value):
        x, y, z = unsolved_value[0], unsolved_value[1], unsolved_value[2]
        return [
            x ** 2 + y ** 2 - 10,
            y ** 2 + z ** 2 - 34,
            x ** 2 + z ** 2 - 26,
        ]

    solved = fsolve(solve_function, [0, 0, 0])
    print(solved)
    print("Program done!")


if part==3:
    #解的非完备性
    def solve_function(unsolved_value):
        x = unsolved_value[0]
        return [
            x ** 2 - 9,
        ]

    solved = fsolve(solve_function, [0])
    print(solved)
    print("Program done!")

if part == 4:
    # 较难无法求解
    def solve_function(unsolved_value):
        x, y = unsolved_value[0], unsolved_value[1]
        return [
            x * x + 2 * x * y,
            2 * x * y - 2 * y * y
        ]

    solved = fsolve(solve_function, [6, -3])
    print(solved)

    print("Program done!")
