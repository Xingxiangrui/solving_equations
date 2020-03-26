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
part=12

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

if part==9:
    #求解微分方程
    u = Function('u')
    t = Symbol('t',real=True)
    L = Symbol('L',real=True)
    C = Symbol('C', real=True)
    R = Symbol('R', real=True)
    w = Symbol('w', real=True)
    M = Symbol('M',real=True)
    eq=L*C*u(t).diff(t,2)+R*C*u(t).diff(t,1)+u(t)-M*cos(w*t)

    #eq2= u(t).diff(t,2)+2*u(t).diff(t,1)-u(t)
    print(dsolve(Eq(eq,0),u(t)))

if part==10:
    t = Symbol('t', real=True)
    L = Symbol('L', real=True)
    C = Symbol('C', real=True)
    R = Symbol('R', real=True)
    w = Symbol('w', real=True)
    M = Symbol('M', real=True)
    C1 = Symbol('C1', real=True)
    C2 = Symbol('C2', real=True)
    u = Function('u')
    Solve=Symbol('Slolve')
    u=-C*L*M*w**2*cos(t*w)/(C**2*R**2*w**2 + (C*L*w**2 - 1)**2) +\
    C*M*R*w*sin(t*w)/(C**2*R**2*w**2 + (C*L*w**2 - 1)**2) +\
    C1*exp(t*(-R - sqrt(C*(C*R**2 - 4*L))/C)/(2*L)) +\
    C2*exp(t*(-R + sqrt(C*(C*R**2 - 4*L))/C)/(2*L)) +\
    M*cos(t*w)/(C**2*R**2*w**2 + (C*L*w**2 - 1)**2)

    Result=Symbol('Result',real=True)
    Result=simplify(L*C*u(t).diff(t,2)+R*C*u(t).diff(t,1)+u(t)-M*cos(w*t))
    print(Result)

if part==11:
    #较为复杂的方程
    u = Function('u')
    t = Symbol('t',real=True)
    L = Symbol('L',real=True)
    C = Symbol('C', real=True)
    R = Symbol('R', real=True)
    w = Symbol('w', real=True)
    M = Symbol('M',real=True)
    eq=L*C*u(t).diff(t,2)+R*C*u(t).diff(t,1)+u(t)-M*cos(w*t)

    eq2= u(t).diff(t,2)+2*u(t).diff(t,1)-u(t)
    print(dsolve(Eq(eq,0),u(t)))

if part==12:
    #三角函数
    x=Symbol('x')
    print(cos(3*pi/2-x))
    print(cos(3 * pi / 2 + x))

if part==13:
    x=Symbol('x')
    y=atan(x)
    print("一阶导数:",end=' ')
    print(diff(y,x,1)) #一阶导数
    print("二阶导数:",end=' ')
    print(diff(y,x,2)) #二阶导数
    print("积分:",end=' ')
    print(integrate(y,x))  #积分
    print("定积分:", end=' ')
    print(integrate(y, (x,0,3*pi/2)))   #定积分
    print("广义积分:", end=' ')
    print(integrate(y, (x, 0, +oo)))  # 广义积分
    print('以下为多元方程：')
    a = Symbol('a')
    y2=atan(x+a)
    print("偏导一阶:", end=' ')
    print(diff(y2,x,1)) #偏导一阶
    print("偏导二阶:", end=' ')
    print(diff(y2,x,2)) #偏导二阶
    print("积分:",end=' ')
    print(integrate(y2,x))  #积分
    print("定积分:", end=' ')
    y3=exp(-(x+a+1))
    print(integrate(y3, (x,0,3*pi/2)))   #定积分
    print("广义积分:", end=' ')
    print(integrate(y3, (x, 0, +oo)))  # 广义积分
print("Program done!")
