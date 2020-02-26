#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
python解方程
created by xingxinagrui on 2020.2.24
"""

from scipy.optimize import fsolve
from math import sin,cos
from sympy import *

# 1-4 scipy
# 5-7 sympy
part=8

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


if part==3:
    #解的非完备性
    def solve_function(unsolved_value):
        x = unsolved_value[0]
        return [
            x ** 2 - 9,
        ]

    solved = fsolve(solve_function, [0])
    print(solved)


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

if part == 5:
    # 二元一次方程
    x = Symbol('x')
    y = Symbol('y')
    solved_value=solve([2*x+y-1, x-2*y], [x, y])
    print(solved_value)


if part == 6:
    # 多解情况
    x = Symbol('x')
    solved_value=solve([x**2-9], [x])
    print(solved_value)

    # 复数解
    solved_value = solve([x ** 2 + 9], [x])
    print(solved_value)
    solved_value = solve([x ** 4 - 9], [x])
    print(solved_value)


    # 非线性解
    solved_value = solve([sin(x) - 0.5], [x])
    print(solved_value)
    solved_value = solve([sin(x) - 1], [x])
    print(solved_value)

if part == 7:
    # 二元二次方程组
    x = Symbol('x')
    y=  Symbol('y')
    solved_value=solve([x**2+2*x*y-6,2*x*y-2*y**2+3], [x,y])
    print(solved_value)

if part==8:
    x=-(-3 + sqrt(13))*sqrt(sqrt(13)/2 + 2)
    y=-sqrt(sqrt(13)/2 + 2)
    z=x*x-2*y*y
    print(z)


print("Program done!")
