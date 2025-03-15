#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试按题号拆分子问题功能
"""

import os
from backend.core.solver import problem_solver

def test_subproblem_by_question_number():
    """测试按题号拆分子问题功能"""
    # 测试问题1：明确的数字题号
    test_problem1 = """
    一质量为m的小球从高度h处由静止释放，沿着光滑的轨道下滑，轨道的形状是一个半径为R的圆的1/4弧，
    然后小球离开轨道做自由抛体运动。求：
    1. 小球离开轨道时的速度大小和方向
    2. 小球落地时距离轨道末端的水平距离
    """
    
    # 测试问题2：带括号的题号
    test_problem2 = """
    考虑一个质量为m的物体在竖直平面内的运动。物体受到重力和空气阻力的作用，空气阻力与速度成正比，比例系数为k。
    物体从高度h处以初速度v0水平抛出。求：
    (1) 建立物体运动的微分方程
    (2) 求解物体的运动轨迹方程
    (3) 计算物体落地时的速度大小和方向
    """
    
    # 创建输出目录
    os.makedirs("test_results", exist_ok=True)
    
    # 测试问题1
    print("\n=== 测试按数字题号拆分子问题 ===")
    print("原题目：", test_problem1.strip())
    
    with open("test_results/test_problem1_results.md", "w", encoding="utf-8") as f:
        f.write("# 测试按数字题号拆分子问题\n\n")
        f.write(f"## 原题目\n\n{test_problem1.strip()}\n\n")
        f.write("## 求解过程\n\n")
        
        for i, step in enumerate(problem_solver.solve_problem(test_problem1, None, False)):
            if isinstance(step, tuple):
                solution, _ = step
                if solution:
                    print(f"\n步骤 {i}: 已保存到文件")
                    f.write(f"### 步骤 {i}\n\n{solution}\n\n")
    
    # 测试问题2
    print("\n\n=== 测试按括号题号拆分子问题 ===")
    print("原题目：", test_problem2.strip())
    
    with open("test_results/test_problem2_results.md", "w", encoding="utf-8") as f:
        f.write("# 测试按括号题号拆分子问题\n\n")
        f.write(f"## 原题目\n\n{test_problem2.strip()}\n\n")
        f.write("## 求解过程\n\n")
        
        for i, step in enumerate(problem_solver.solve_problem(test_problem2, None, False)):
            if isinstance(step, tuple):
                solution, _ = step
                if solution:
                    print(f"\n步骤 {i}: 已保存到文件")
                    f.write(f"### 步骤 {i}\n\n{solution}\n\n")
    
    print("\n测试完成，结果已保存到 test_results 目录")

if __name__ == "__main__":
    test_subproblem_by_question_number()