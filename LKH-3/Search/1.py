import matplotlib.pyplot as plt
import numpy as np
import matplotlib
# 强制使用 TkAgg 后端，这样会弹出一个独立窗口，绕过 PyCharm 的 bug
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np



def generate_tsp_plot(num_cities=30, seed=42):
    """
    生成并绘制 TSP 路径图
    :param num_cities: 城市数量
    :param seed: 随机种子，保证每次生成的图一样
    """
    # 1. 设置随机种子并生成随机坐标 (x, y)
    np.random.seed(seed)
    cities = np.random.rand(num_cities, 2)

    # 2. 使用“最近邻算法” (Nearest Neighbor) 构建一条近似路径
    # 注意：这不是绝对最优解，但画出来的图比较整洁，适合展示
    unvisited = set(range(num_cities))
    current_city = 0
    path_indices = [0]
    unvisited.remove(0)

    while unvisited:
        # 找到距离当前城市最近的未访问城市
        nearest_city = None
        min_dist = float('inf')

        for city_idx in unvisited:
            # 计算欧几里得距离
            dist = np.linalg.norm(cities[current_city] - cities[city_idx])
            if dist < min_dist:
                min_dist = dist
                nearest_city = city_idx

        # 移动到下一个城市
        current_city = nearest_city
        path_indices.append(current_city)
        unvisited.remove(current_city)

    # 最后回到起点，形成闭环
    path_indices.append(path_indices[0])

    # 获取路径的坐标点
    path_points = cities[path_indices]

    # 3. 开始绘图
    plt.figure(figsize=(10, 8), dpi=100)

    # 绘制路径连线
    plt.plot(path_points[:, 0], path_points[:, 1],
             c='royalblue', linewidth=1.5, zorder=1, label='Path')

    # 绘制所有城市节点
    plt.scatter(cities[:, 0], cities[:, 1],
                c='black', s=50, zorder=2, label='Cities')

    # 标记起点/终点 (红色)
    plt.scatter(cities[0, 0], cities[0, 1],
                c='red', s=150, marker='*', zorder=3, label='Start/End')

    # 添加箭头表示方向 (可选，每隔几个点画一个箭头)
    for i in range(len(path_points) - 1):
        # 仅在部分线段绘制箭头以防太乱
        if i % 1 == 0:
            start = path_points[i]
            end = path_points[i + 1]
            plt.annotate('', xy=end, xytext=start,
                         arrowprops=dict(arrowstyle="->", color='royalblue', lw=1.5))

    # 图表装饰
    plt.title(f'TSP Path Visualization ({num_cities} Cities)', fontsize=15)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 显示图片
    plt.show()


# 运行函数生成图片
if __name__ == "__main__":
    generate_tsp_plot(num_cities=25)