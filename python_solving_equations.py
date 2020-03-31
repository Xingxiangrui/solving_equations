#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
python解方程
created by xingxinagrui on 2020.2.24
"""

from scipy.optimize import fsolve
from math import sin, cos
from sympy import *
import math

# 1-4 scipy
# 5-7 sympy
part = 19

if part == 1:
    # 求解非线性方程组,scipy
    def solve_function(unsolved_value):
        x = unsolved_value[0]
        return [
            sin(x) - 0.5
        ]


    solved = fsolve(solve_function, [3.14])
    print(solved)
    solved = fsolve(solve_function, [0])
    print(solved)

if part == 2:
    # 求解三元二次方程组,scipy
    def solve_function(unsolved_value):
        x, y, z = unsolved_value[0], unsolved_value[1], unsolved_value[2]
        return [
            x ** 2 + y ** 2 - 10,
            y ** 2 + z ** 2 - 34,
            x ** 2 + z ** 2 - 26,
        ]


    solved = fsolve(solve_function, [0, 0, 0])
    print(solved)

if part == 3:
    # 解的非完备性，scipy，不完备
    def solve_function(unsolved_value):
        x = unsolved_value[0]
        return [
            x ** 2 - 9,
        ]


    solved = fsolve(solve_function, [0])
    print(solved)

if part == 4:
    # scipy,较难无法求解
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
    solved_value = solve([2 * x + y - 1, x - 2 * y], [x, y])
    print(solved_value)

if part == 6:
    # 多解情况
    x = Symbol('x')
    solved_value = solve([x ** 2 - 9], [x])
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
    y = Symbol('y')
    solved_value = solve([x ** 2 + 2 * x * y - 6, 2 * x * y - 2 * y ** 2 + 3], [x, y])
    print(solved_value)

if part == 8:
    # 二元多次方程组
    x = -(-3 + sqrt(13)) * sqrt(sqrt(13) / 2 + 2)
    y = -sqrt(sqrt(13) / 2 + 2)
    z = x * x - 2 * y * y
    print(z)

if part == 9:
    # 求解微分方程，RLC震荡电路
    u = Function('u')
    t = Symbol('t', real=True)
    L = Symbol('L', real=True)
    C = Symbol('C', real=True)
    R = Symbol('R', real=True)
    w = Symbol('w', real=True)
    M = Symbol('M', real=True)
    eq = L * C * u(t).diff(t, 2) + R * C * u(t).diff(t, 1) + u(t) - M * cos(w * t)

    # eq2= u(t).diff(t,2)+2*u(t).diff(t,1)-u(t)
    print(dsolve(Eq(eq, 0), u(t)))

if part == 10:
    # RLC震荡电路的验算
    t = Symbol('t', real=True)
    L = Symbol('L', real=True)
    C = Symbol('C', real=True)
    R = Symbol('R', real=True)
    w = Symbol('w', real=True)
    M = Symbol('M', real=True)
    C1 = Symbol('C1', real=True)
    C2 = Symbol('C2', real=True)
    u = Function('u')
    Solve = Symbol('Slolve')
    u = -C * L * M * w ** 2 * cos(t * w) / (C ** 2 * R ** 2 * w ** 2 + (C * L * w ** 2 - 1) ** 2) + \
        C * M * R * w * sin(t * w) / (C ** 2 * R ** 2 * w ** 2 + (C * L * w ** 2 - 1) ** 2) + \
        C1 * exp(t * (-R - sqrt(C * (C * R ** 2 - 4 * L)) / C) / (2 * L)) + \
        C2 * exp(t * (-R + sqrt(C * (C * R ** 2 - 4 * L)) / C) / (2 * L)) + \
        M * cos(t * w) / (C ** 2 * R ** 2 * w ** 2 + (C * L * w ** 2 - 1) ** 2)

    Result = Symbol('Result', real=True)
    Result = simplify(L * C * u(t).diff(t, 2) + R * C * u(t).diff(t, 1) + u(t) - M * cos(w * t))
    print(Result)

if part == 11:
    # 较为复杂的方程，RLC震荡电路
    u = Function('u')
    t = Symbol('t', real=True)
    L = Symbol('L', real=True)
    C = Symbol('C', real=True)
    R = Symbol('R', real=True)
    w = Symbol('w', real=True)
    M = Symbol('M', real=True)
    eq = L * C * u(t).diff(t, 2) + R * C * u(t).diff(t, 1) + u(t) - M * cos(w * t)

    eq2 = u(t).diff(t, 2) + 2 * u(t).diff(t, 1) - u(t)
    print(dsolve(Eq(eq, 0), u(t)))

if part == 12:
    # 三角函数与化简
    x = Symbol('x')
    print(cos(3 * pi / 2 - x))
    print(cos(3 * pi / 2 + x))

if part == 13:
    # 求导与微积分，偏导
    x = Symbol('x')
    y = atan(x)
    print("一阶导数:", end=' ')
    print(diff(y, x, 1))  # 一阶导数
    print("二阶导数:", end=' ')
    print(diff(y, x, 2))  # 二阶导数
    print("积分:", end=' ')
    print(integrate(y, x))  # 积分
    print("定积分:", end=' ')
    print(integrate(y, (x, 0, 3 * pi / 2)))  # 定积分
    print("广义积分:", end=' ')
    print(integrate(y, (x, 0, +oo)))  # 广义积分
    print('以下为多元方程：')
    a = Symbol('a')
    y2 = atan(x + a)
    print("偏导一阶:", end=' ')
    print(diff(y2, x, 1))  # 偏导一阶
    print("偏导二阶:", end=' ')
    print(diff(y2, x, 2))  # 偏导二阶
    print("积分:", end=' ')
    print(integrate(y2, x))  # 积分
    print("定积分:", end=' ')
    y3 = exp(-(x + a + 1))
    print(integrate(y3, (x, 0, 3 * pi / 2)))  # 定积分
    print("广义积分:", end=' ')
    print(integrate(y3, (x, 0, +oo)))  # 广义积分

if part == 14:
    # 求解客户方程,（1）
    t = Symbol('t')
    C = Function('C')
    eq = log(C(0) / C(t)) - 0.0379 * t + 0.0489
    print(dsolve(Eq(eq, 0), C(t)))
    '''
    1.05011533395588*C(0)*exp(-0.0379*t)
    '''
if part == 15:
    # 求解客户方程一般形式
    t = Symbol('t')
    C = Function('C')
    b = Symbol('b')
    print('第一组解的形式：')
    eq1 = log(C(0) / C(t)) - 0.0379 * t + b
    print(dsolve(Eq(eq1, 0), C(t)))
    '''
    C(0)*exp(b - 0.0379*t)
    '''
    print('第二组解的形式：')
    eq2 = 0.001 * (1 / C(t) - 1 / C(0)) - 0.0022 * t + 0.0133
    print(dsolve(Eq(eq2, 0), C(t)))
    '''
    Eq(C(t), 10.0*C(0)/(22.0*t*C(0) - 133.0*C(0) + 10.0))
    '''
    print('第三组方程的解：')
    eq3 = (1 / (C(t) * C(t)) - 1 / (C(0) * C(0))) / 2 * 0.000001 - 0.002 * t + 0.0148
    print(dsolve(Eq(eq3, 0), C(t)))
    print(dsolve(eq3, C(t)))

    '''
    [Eq(C(t), -sqrt(1/(4000.0*t*C(0)**2 - 29600.0*C(0)**2 + 1.0))*C(0)), 
      Eq(C(t), sqrt(1/(4000.0*t*C(0)**2 - 29600.0*C(0)**2 + 1.0))*C(0))]
    '''

if part == 16:
    # 求解客户方程，验算
    t = 0
    C = 1.05011533395588 * 1300 * exp(-0.0379 * t)
    print(C)
    t = 0
    b = 0.0489
    print(1300 * exp(0.0489 - 0.0379 * t))
    '''
    1365.14993414264
    '''
    print(log(1300 / 1365.14993414264))

if part == 17:
    # 较难的微分方程组不可解
    P = Symbol('P')
    a = Symbol('a')
    b = Symbol('b')
    w = Symbol('w')
    C = Symbol('C')
    r = Symbol('r')
    As = Symbol('As')
    Ds = Symbol('Ds')
    Av = Symbol('Av')
    k = Symbol('k')
    B = Symbol('B')
    Dv = Symbol('Dv')
    t = Symbol('t')
    L0 = Symbol('L0')
    exp_value = Symbol('exp_value')  # exp_value=exp(-b*s)

    L1 = Function('L1')
    L2 = Function('L2')
    I = Function('I')
    S = Function('S')
    V = Function('V')

    eq1 = S(t).diff(t, 1) - As * I(t) + Ds * S(t)
    eq2 = V(t).diff(t, 1) - Av * I(t) + k * V(t) * (S(t) / (1 + S(t))) * B + Dv * V(t)
    eq3 = L1(t) * As + L2(t) * Av - C
    eq4 = L1(t).diff(t, 1) - (r + Ds) * L1(t) - P * V(t) ** a * b * L0 * exp_value - k * L2(t) * V(t) * B / (
            (1 + S(t)) * (1 + S(t)))
    eq5 = L2(t).diff(t, 1) - (r + k * (B * S(t)) / (1 + S(t)) + Dv) * L2(t) + P * a * V(t) ** (a - 1) * (
            1 + L0 * exp_value) - w

    eq=(Eq(S(t).diff(t, 1),As * I(t) - Ds * S(t)),\
        Eq(V(t).diff(t, 1),-(- Av * I(t) + k * V(t) * (S(t) / (1 + S(t))) * B + Dv * V(t))),\
        Eq(L1(t).diff(t, 1),-( - (r + Ds) * L1(t) - P * V(t) ** a * b * L0 * exp_value - k * L2(t) * V(t) * B / (
            (1 + S(t)) * (1 + S(t))))),\
        Eq(L2(t).diff(t, 1),-(- (r + k * (B * S(t)) / (1 + S(t)) + Dv) * L2(t) + P * a * V(t) ** (a - 1) * (
            1 + L0 * exp_value) - w)),\
        Eq(eq3,0)
        )
    print(dsolve(eq))
    #print(dsolve(Eq(eq1 + eq2 + eq3 + eq4 + eq5, 0), V(t)))
    # print(solve(Eq(eq2, eq1,0,0), B))
    # print(solve(Eq(eq1,0),Eq(eq2, 0), B))

if part == 18:
    # sympy中的次方
    x = Symbol('x')
    a = Symbol('a')
    V = Function('V')
    eq1 = V(x) ** a - x ** 2
    print(solve((eq1, 0), V(x)))

if part == 19:
    # 求解微分方程组
    t = Symbol('t')
    V = Function('V')
    U = Function('U')
    eq1 = V(t).diff(t, 1) + U(t).diff(t, 1) - 3
    eq2 = V(t).diff(t, 1) - U(t).diff(t, 1)
    eq=(Eq(eq1,0),Eq(eq2,0))
    # print(dsolve([Eq(eq1,0),Eq(eq2,0)],[V(t),U(t)]))
    #print(dsolve(Eq(eq1, 0), U(t)))
    print(dsolve(eq, U(t)))
    #print(dsolve([eq1], U(t)))
    # print(dsolve([Eq(eq1, 0), Eq(eq2, 0)], U(t)))
    #print(dsolve(Eq(eq1, 0), U(t)))

if part == 20:
    t = Symbol('t')  # 自变量
    u1 = Symbol('u1')  # 已知量
    u2 = Symbol('u2')
    w1 = Symbol('w1')
    w2 = Symbol('w2')
    B1 = Symbol('B1')
    B2 = Symbol('B2')
    Lambda1 = Symbol('Lambda1')
    Lambda2 = Symbol('Lambda2')
    r1 = Symbol('r1')
    r2 = Symbol('r2')
    P = Symbol('P')
    #n = Symbol('n')
    n=3
    a1 = Symbol('a1')  # 待求量
    b1 = Symbol('b1')
    c1 = Symbol('c1')
    d1 = Symbol('d1')
    a2 = Symbol('a2')
    b2 = Symbol('b2')
    c2 = Symbol('c2')
    d2 = Symbol('d2')
    #X1 = Function('X1')  # 函数表达式
    #X2 = Function('X2')

    X1 = a1 * cos(n * t) + b1 * sin(n * t) + c1 * cos(3 * n * t) + d1 * sin(3 * n * t)
    X2 = a2 * cos(n * t) + b2 * sin(n * t) + c2 * cos(3 * n * t) + d2 * sin(3 * n * t)

    eq1 = X1.diff(t, 2) + \
          2 * u1 * X1.diff(t, 1) + \
          w1**2*X1 + b1 * X1 ** 3 + \
          Lambda1 * (X1-X2) + \
          r1 * ((X1-X2) ** 3) - P * cos(n * t)
    eq2 = X2.diff(t, 2) + \
          2 * u2 * X2.diff(t, 1) + \
          w2**2*X2 + b2 * X2 ** 3 + \
          Lambda2 * (X2-X1) + \
          r2 * ((X2-X1) ** 3)

    print('eq1/sin(n*t):')
    print(simplify(eq1/sin(n*t)))
    print('eq1/cos(3*n*t):')
    print(simplify(eq1/cos(3*n*t)))
    print('eq1/sin(3*n*t)')
    print(simplify(eq1/sin(3*n*t)))


    '''
    eq1/sin(n*t):
    (Lambda1*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t)) - P*cos(n*t) + b1*(a1*cos(n*t) + b1*sin(n*t) + c1*cos(3*n*t) + d1*sin(3*n*t))**3 - n**2*(a1*cos(n*t) + b1*sin(n*t) + 9*c1*cos(3*n*t) + 9*d1*sin(3*n*t)) + r1*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t))**3 + 2*u1*(-a1*n*sin(n*t) + b1*n*cos(n*t) - 3*c1*n*sin(3*n*t) + 3*d1*n*cos(3*n*t)) + w1**2*(a1*cos(n*t) + b1*sin(n*t) + c1*cos(3*n*t) + d1*sin(3*n*t)))/sin(n*t)
    (Lambda1*(a1*cos(3*t) - a2*cos(3*t) + b1*sin(3*t) - b2*sin(3*t) + c1*cos(9*t) - c2*cos(9*t) + d1*sin(9*t) - d2*sin(9*t)) - P*cos(3*t) - 9*a1*cos(3*t) + b1*(a1*cos(3*t) + b1*sin(3*t) + c1*cos(9*t) + d1*sin(9*t))**3 - 9*b1*sin(3*t) - 81*c1*cos(9*t) - 81*d1*sin(9*t) + r1*(a1*cos(3*t) - a2*cos(3*t) + b1*sin(3*t) - b2*sin(3*t) + c1*cos(9*t) - c2*cos(9*t) + d1*sin(9*t) - d2*sin(9*t))**3 - 6*u1*(a1*sin(3*t) - b1*cos(3*t) + 3*c1*sin(9*t) - 3*d1*cos(9*t)) + w1**2*(a1*cos(3*t) + b1*sin(3*t) + c1*cos(9*t) + d1*sin(9*t)))/sin(3*t)
    eq1/cos(3*n*t):
    (Lambda1*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t)) - P*cos(n*t) + b1*(a1*cos(n*t) + b1*sin(n*t) + c1*cos(3*n*t) + d1*sin(3*n*t))**3 - n**2*(a1*cos(n*t) + b1*sin(n*t) + 9*c1*cos(3*n*t) + 9*d1*sin(3*n*t)) + r1*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t))**3 + 2*u1*(-a1*n*sin(n*t) + b1*n*cos(n*t) - 3*c1*n*sin(3*n*t) + 3*d1*n*cos(3*n*t)) + w1**2*(a1*cos(n*t) + b1*sin(n*t) + c1*cos(3*n*t) + d1*sin(3*n*t)))/cos(3*n*t)
    eq1/sin(3*n*t)
    (Lambda1*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t)) - P*cos(n*t) + b1*(a1*cos(n*t) + b1*sin(n*t) + c1*cos(3*n*t) + d1*sin(3*n*t))**3 - n**2*(a1*cos(n*t) + b1*sin(n*t) + 9*c1*cos(3*n*t) + 9*d1*sin(3*n*t)) + r1*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t))**3 + 2*u1*(-a1*n*sin(n*t) + b1*n*cos(n*t) - 3*c1*n*sin(3*n*t) + 3*d1*n*cos(3*n*t)) + w1**2*(a1*cos(n*t) + b1*sin(n*t) + c1*cos(3*n*t) + d1*sin(3*n*t)))/sin(3*n*t)
    '''
    # print('eq1 is:' )
    # print(eq1)
    # print(simplify(eq1))
    # print('eq2 is:' )
    # print(eq2)
    # print(simplify(eq2))

    '''
    eq1 is:
    Lambda1*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t)) - P*cos(n*t) + b1*(a1*cos(n*t) + b1*sin(n*t) + c1*cos(3*n*t) + d1*sin(3*n*t))**3 - n**2*(a1*cos(n*t) + b1*sin(n*t) + 9*c1*cos(3*n*t) + 9*d1*sin(3*n*t)) + r1*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t))**3 + 2*u1*(-a1*n*sin(n*t) + b1*n*cos(n*t) - 3*c1*n*sin(3*n*t) + 3*d1*n*cos(3*n*t)) + w1**2*(a1*cos(n*t) + b1*sin(n*t) + c1*cos(3*n*t) + d1*sin(3*n*t))
    Lambda1*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t)) - P*cos(n*t) + b1*(a1*cos(n*t) + b1*sin(n*t) + c1*cos(3*n*t) + d1*sin(3*n*t))**3 - n**2*(a1*cos(n*t) + b1*sin(n*t) + 9*c1*cos(3*n*t) + 9*d1*sin(3*n*t)) - 2*n*u1*(a1*sin(n*t) - b1*cos(n*t) + 3*c1*sin(3*n*t) - 3*d1*cos(3*n*t)) + r1*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t))**3 + w1**2*(a1*cos(n*t) + b1*sin(n*t) + c1*cos(3*n*t) + d1*sin(3*n*t))
    eq2 is:
    Lambda2*(-a1*cos(n*t) + a2*cos(n*t) - b1*sin(n*t) + b2*sin(n*t) - c1*cos(3*n*t) + c2*cos(3*n*t) - d1*sin(3*n*t) + d2*sin(3*n*t)) + b2*(a2*cos(n*t) + b2*sin(n*t) + c2*cos(3*n*t) + d2*sin(3*n*t))**3 - n**2*(a2*cos(n*t) + b2*sin(n*t) + 9*c2*cos(3*n*t) + 9*d2*sin(3*n*t)) + r2*(-a1*cos(n*t) + a2*cos(n*t) - b1*sin(n*t) + b2*sin(n*t) - c1*cos(3*n*t) + c2*cos(3*n*t) - d1*sin(3*n*t) + d2*sin(3*n*t))**3 + 2*u2*(-a2*n*sin(n*t) + b2*n*cos(n*t) - 3*c2*n*sin(3*n*t) + 3*d2*n*cos(3*n*t)) + w2**2*(a2*cos(n*t) + b2*sin(n*t) + c2*cos(3*n*t) + d2*sin(3*n*t))
    -Lambda2*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t)) + b2*(a2*cos(n*t) + b2*sin(n*t) + c2*cos(3*n*t) + d2*sin(3*n*t))**3 - n**2*(a2*cos(n*t) + b2*sin(n*t) + 9*c2*cos(3*n*t) + 9*d2*sin(3*n*t)) - 2*n*u2*(a2*sin(n*t) - b2*cos(n*t) + 3*c2*sin(3*n*t) - 3*d2*cos(3*n*t)) - r2*(a1*cos(n*t) - a2*cos(n*t) + b1*sin(n*t) - b2*sin(n*t) + c1*cos(3*n*t) - c2*cos(3*n*t) + d1*sin(3*n*t) - d2*sin(3*n*t))**3 + w2**2*(a2*cos(n*t) + b2*sin(n*t) + c2*cos(3*n*t) + d2*sin(3*n*t))
    '''
    solved_value=solve([eq1,eq2],[a1,b1,c1,d1,a2,b2,c2,d2])
    print('Solved_value:')
    print(solved_value)


print("Program done!")
