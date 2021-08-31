import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

from sympy import *

def line_func(x,A,B):
    return A*x+B

'''方式1            方式2  y
|————————————> x          ↑
|                         |
|                     --> |  
↓                         |————————————> x
y
'''
'''将旧坐标系转为新坐标系'''
def oldxy2newxy(x,y,H):
    y = [H - sub_y for sub_y in  y]
    return x, y

'''将新坐标系转为旧坐标系，同上'''
def newxy2oldxy(x,y,H):
    y = [H - sub_y for sub_y in  y]
    return x, y

'''拟合单条线，返回对应的x值'''
def get_x_y(x:list, y:list, H, y_stead):
    '''
    :param x: list格式，一条车道线对应所有的x坐标
    :param y: 同上
    :param H: 图像的高
    :return:
    '''
    # 直线拟合
    x,y = oldxy2newxy(x, y, H)
    A1, B1 = optimize.curve_fit(line_func, x, y)[0]     # 拟合好了线
    get_x1 = -B1*1./A1    #y为0时,x坐标
    max_y = max(y)
    if max_y < y_stead:
        y_stead = max_y
    get_x2 = 1.0*(y_stead-B1)/A1 # y为固定值时，x坐标
    return get_x1, get_x2, H, y_stead

def get_triangle_point(x_final_left_down, y_final_left_down, x_final_left_up, y_final_left_up, x_final_right_down, y_final_right_down, x_final_right_up, y_final_right_up,H):
    # 转换坐标系
    y_final_left_up = (H-y_final_right_up)*1.0
    y_final_left_down = (H-y_final_left_down)*1.0
    y_final_right_down = (H-y_final_right_down)*1.0
    y_final_right_up = (H-y_final_right_up)*1.0
    x11 = x_final_left_down
    x12 = x_final_left_up
    y11 = y_final_left_down
    y12 = y_final_left_up

    x21 = x_final_right_down
    x22 = x_final_right_up
    y21 = y_final_right_down
    y22 = y_final_right_up
    # 得到第三角形的三个点th
    x = Symbol('x')
    ans = solve([(x-x11)/(x12-x11)*(y12-y11)+y11-(x-x21)/(x22-x21)*(y22-y21)+y21], [x])
    x = ans[x]
    y = (x-x11)/(x12-x11)*(y12-y11)+y11
    # 转换坐标系
    y = H-y
    return x,y

'''获取最终的四个点'''
def get_final_x(all_line, W, H, y_stead):
    '''
    :param all_line: [[A_x,A_y],[B],...,[N]]   A_x,A_y = X_set:list,Y_set:list, A,B,C...代表不同的车道线

    [[[x1,x2],[y1,y2]],[[],[]]]
    :param W:
    :param H:
    :param y_stead: 固定好的y值,这个y值是相对于方式2的坐标系，见line8
    :return:
    '''
    xy_candidate = []
    for line in all_line:
        x1, x2, y1, y2 = get_x_y(line[0], line[1], H, y_stead)  # x1,y1存放底部坐标，x2,y2 存放 y_stead 对应的坐标
        xy_candidate.append([x1, x2, y1, y2])
    x_steay_dowm = W/2.0
    x_left_theo = W*2
    x_right_theo = W*2
    x_final_left_down, y_final_left_down, x_final_left_up, y_final_left_up, x_final_right_down, y_final_right_down, x_final_right_up, y_final_right_up= None, None, None, None,None, None, None, None
    for index in range(len(xy_candidate)):
        if xy_candidate[index][0] - x_steay_dowm < 0 and abs(xy_candidate[index][0]-x_steay_dowm) < x_left_theo:
            # 左边
            x_left_theo = abs(xy_candidate[index][0]-x_steay_dowm)
            x_final_left_down = xy_candidate[index][0]
            x_final_left_up = xy_candidate[index][1]
            y_final_left_down = xy_candidate[index][2]  # 可以不用管它
            y_final_left_up = H-xy_candidate[index][3]
        if xy_candidate[index][0] - x_steay_dowm > 0 and abs(xy_candidate[index][0]-x_steay_dowm) < x_right_theo:
            # 右边
            x_right_theo = abs(xy_candidate[index][0]-x_steay_dowm)
            x_final_right_down = xy_candidate[index][0]
            x_final_right_up = xy_candidate[index][1]
            y_final_right_down = xy_candidate[index][2]
            y_final_right_up = H-xy_candidate[index][3]

    x,y = get_triangle_point(x_final_left_down, y_final_left_down, x_final_left_up, y_final_left_up, x_final_right_down, y_final_right_down, x_final_right_up, y_final_right_up, H)
    return (x_final_left_down, y_final_left_down), (x, y), (x_final_right_down, y_final_right_down), (x, y)

