import math
import sys
import time
import threading
import random
import numpy as np
import pandas as pd
import matplotlib
import csv

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import io
import scipy.stats as stats


# matplotlib.use('Agg')


class Car:
    def __init__(self, initial_position, color, number):
        self.angle = 0.00
        self.position = initial_position
        self.velocity = 0.00
        self.acceleration = 2.00
        self.max_velocity = random.uniform(13, 15) * k  # 最大速度限制
        self.min_velocity = 0.00  # 最小速度限制
        self.color = color
        self.number = number
        self.num_events_in_10s = 0
        self.last_event_update = 0
        self.event_triggered = False
        self.speed_history = []
        self.front_car = None
        self.cumulative_distance = 0.0
        self.car_patch = patches.Rectangle((initial_position, 0), 4, 2, color=self.color)
        self.T = random.uniform(0.5, 1.5)
        self.delta = random.uniform(1.5, 2)
        self.current_station = None
        self.order = 0
        self.station_arrival_time = []


k = 1
road_length = 230
road_radius = road_length / (2 * math.pi)
num_cars = int(38 * 0.8)
car_length = 4

# start_num_cars = int(38 * 0.2)
# end_num_cars = int(38 * 0.8)
deceleration = 4
acceleration = 3 * k

num_stations = 8
station_positions = np.linspace(0, road_length, num_stations, endpoint=False)
station_length = station_positions[1] - station_positions[0]
station_data = [{'position': pos, 'car_count': 0, 'total_speed': 0.00, 'average_speed': 0.00, 'index': i} for i, pos in
                enumerate(station_positions)]

interval = 100
simulation_times = 0
warm_up_phase = True
average_velocities = []
car_initial_positions = np.linspace(0, road_length, num_cars, endpoint=False)
cars = [Car(position, f"C{i + 1}", number=i) for i, position in enumerate(car_initial_positions)]

update_count = 0
info_printed = False
car_positions = car_initial_positions

initial_spacing = car_positions[2] - car_positions[1]  # 初始间距
desired_spacing = initial_spacing
s_0 = 1

warm_up_duration = 60
interval_duration = 1000  # 初始值为300秒（5分钟）
all_car_velocities_sets = []
average_velocities_per_cars = []
average_speeds_by_density = {0: []}
intervals = []

car_labels = []
buf = io.BytesIO()


def play_animation(event):
    ani.event_source.start()


def pause_animation(event):
    ani.event_source.stop()


def polar_to_cartesian(angle, radius):
    x = radius * np.cos(np.deg2rad(angle))
    y = radius * np.sin(np.deg2rad(angle))
    return x, y


# 添加一个函数来初始化模拟
def initialize_simulation():
    global cars, interval_duration, average_velocities, update_count, car_initial_positions, simulation_times, warm_up_phase, desired_spacing, station_data

    # 初始化车辆
    car_initial_positions = np.linspace(0, road_length, num_cars, endpoint=False)
    cars = [Car(position, f"C{i + 1}", number=i) for i, position in enumerate(car_initial_positions * 0.7)]

    for i, car in enumerate(cars):
        car.front_car = cars[(i + 1) % num_cars]
        # 判断每辆车的初始位置属于哪个站点
        for station in station_data:
            if station['position'] <= car.position < station['position'] + station_length:
                car.current_station = station
                station['car_count'] += 1
                station['total_speed'] += car.velocity
                print(f"车辆{car.number + 1}初始在站点{station['index'] + 1}")
                break
    disturbance = cars[1].position - cars[0].position
    desired_spacing = disturbance
    cars[0].position += disturbance * 0.3

    # 初始化其他相关变量
    update_count = 0
    # interval_duration = 10  # 重置interval_duration
    simulation_times += 1
    warm_up_phase = True
    # current_simulation_time = 0

    # 初始化统计数据
    average_velocities = []


def update(frame):
    global update_count, info_printed, car_positions, cars, interval_duration, warm_up_phase, desired_spacing
    if interval == 1:
        ani.event_source.stop()
        plt.close()

    patches = []

    current_simulation_time = (update_count - 1) * (interval / 1000)
    update_count += 1

    # print(f'第{int(simulation_times)}次模拟')
    # print(f"模拟时长: {(update_count - 1) * (interval / 1000):.2f} 秒")
    # print("车辆信息：")
    # 在每次更新前重置站点内车辆数量和速度总和
    for i, station in enumerate(station_data):
        station['car_count'] = 0  # 重置站点内车辆数量
        station['total_speed'] = 0
        station['order_list'] = []  # 新增一个列表用于存储在站点内的车辆顺序

    for car in cars:
        for station in station_data:
            if station['position'] <= car.position < station['position'] + station_length:
                station['car_count'] += 1
                station['total_speed'] += car.velocity
                station['order_list'].append(car)  # 将车辆添加到站点的顺序列表

        # 在每次更新前，计算站点内的车辆顺序
    for station in station_data:
        order_list = station['order_list']
        order_list.sort(key=lambda car: car.position)
        for order, car in enumerate(order_list):
            car.order = order

    # 在每次更新时记录站点的平均速度
    if not warm_up_phase:
        for car in cars:
            current_station = car.current_station
            next_station = station_data[(current_station['index'] + 1) % len(station_data)]  # 下一个站点是下一个站点或第一个站点

            # 获取当前站点和下一个站点的密度
            order = car.order
            current_density = car.current_station['car_count']
            next_density = next_station['car_count']

            # 将密度组合编码为唯一的整数键
            density_combination = order * 10 + next_density

            # 如果密度组合不存在于字典中，创建一个新的键值对
            if density_combination not in average_speeds_by_density:
                average_speeds_by_density[density_combination] = []

            # 计算当前站点的平均速度
            # current_station['average_speed'] = current_station['total_speed'] / current_density
            # 将平均速度添加到相应的密度组合中
            average_speeds_by_density[density_combination].append(car.velocity)

    for car in cars:
        for station in station_data:
            if station['position'] <= car.position < station['position'] + station_length and car.current_station != station:
                # 车辆进入了站点
                station['car_count'] += 1
                # 寻找当前车辆所在站点
                current_station = car.current_station
                # 如果找到当前站点，减少其车辆数量
                if current_station is not None:
                    current_station['car_count'] -= 1

                # 更新车辆的当前站点
                car.current_station = station
                car.station_arrival_time.append((station['index'] + 1, current_simulation_time))
                print(f"车辆{car.number + 1}进入站点{station['index'] + 1}时间: {current_simulation_time:.2f} 秒")
                break
            # 更新站点的平均速度
    # 在每次更新前重置站点内车辆数量

    for i, station in enumerate(station_data):

        x, y = polar_to_cartesian(station['position'] * 360 / road_length, road_radius)
        # 获取站点内车辆数量
        car_count = station['car_count']
        car_count_text = f'Car Count: {car_count}'

        # 在每次更新时，移除上一次的站点内车辆数量文本，如果存在的话
        if 'last_station_text' in station:
            station['last_station_text'].remove()

        # 添加新的站点内车辆数量文本
        station_text = ax.text(x, y - 5, car_count_text, ha='center', va='center', fontsize=10, color='black')
        station['last_station_text'] = station_text

        # 更新站点内车辆数量
        station['car_count'] = car_count
    for i, car in enumerate(cars[::-1]):
        # for i car in enumerate(cars[::-1]):
        if warm_up_phase:
            # 如果处于热身阶段
            if current_simulation_time >= warm_up_duration:
                warm_up_phase = False
                print("end up warm up")
        else:
            # 如果不处于热身阶段，将速度数据添加到 car.speed_history 列表中
            car.speed_history.append(car.velocity)
        # 计算与前面车辆的间距
        if car.velocity > 0:
            new_position = car.position + car.velocity * (interval / 1000) + 0.5 * car.acceleration * (
                    interval / 1000) ** 2
            if new_position < car.position:
                car.position = car.position  # 不要后退
            else:
                car.position = new_position % road_length
            car.angle = (car.position / road_length) * 360
            x, y = polar_to_cartesian(car.angle, road_radius)
            car.car_patch.set_xy((x, y))
            car.car_patch.angle = car.angle + 90
            patches.append(car.car_patch)
            car_labels[i].set_position((x, y))
        next_car = car.front_car
        if next_car.position < car.position:
            spacing = (next_car.position + road_length - car.position)
        else:
            spacing = (next_car.position - car.position)

        # 根据间距调整加速度
        if not car.event_triggered:
            car.T = random.uniform(0.5, 5)
            car.delta = random.uniform(4, 4)
            desired_velocities = s_0 + max(0, car.velocity * car.T + (
                    car.velocity * (car.velocity - car.front_car.velocity)) / (
                                                   2 * math.sqrt(acceleration * deceleration)))
            car.acceleration = np.clip(
                acceleration * (
                        1 - (car.velocity / car.max_velocity) ** car.delta - (desired_velocities / spacing) ** 2),
                -deceleration, acceleration)

        car.velocity = np.clip(car.velocity + car.acceleration * (interval / 1000), car.min_velocity, car.max_velocity)

    if (update_count - 1) * (interval / 1000) >= interval_duration:
        ani.event_source.stop()
        plt.close()

    return patches


flag = 1

# for num_car in range(start_num_cars, end_num_cars, 1):
# for interval in range(begin_interval, end_interval, -1):  # 从100到1000，每次增加100
while 1:
    if flag == 0:
        break
    initialize_simulation()  # 重新初始化模拟
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-1.2 * road_radius, 1.2 * road_radius)
    ax.set_ylim(-1.2 * road_radius, 1.2 * road_radius)
    road_radius = road_length / (2 * math.pi)  # 道路半径
    angles = np.linspace(0, 2 * np.pi, num_stations, endpoint=False)  # 均匀分布的角度
    station_x = road_radius * 0.98 * np.cos(angles)  # 计算X坐标
    station_y = road_radius * 0.98 * np.sin(angles)  # 计算Y坐标

    # 绘制站点
    for x, y in zip(station_x, station_y):
        station_circle = plt.Circle((x, y), 2, color='gray', zorder=2)
        plt.gca().add_patch(station_circle)

    # 添加站点标签
    for i, station in enumerate(station_data):
        x, y = polar_to_cartesian(station['position'] * 360 / road_length, road_radius * 0.98)
        plt.text(x, y, f'Station {station["index"] + 1}', ha='center', va='center', fontsize=10, color='black')

    for car in cars:
        ax.add_patch(car.car_patch)
    car_labels = [ax.text(0, 0, f'Car {i + 1}', ha='center', va='center', fontsize=8, color='black') for i in
                  range(len(cars) - 1, -1, -1)]

    play_button_ax = plt.axes([0.5, 0.85, 0.1, 0.075])
    play_button = Button(play_button_ax, 'Play', color='orange')
    play_button.on_clicked(play_animation)

    pause_button_ax = plt.axes([0.65, 0.85, 0.1, 0.075])
    pause_button = Button(pause_button_ax, 'Pause', color='black')
    pause_button.on_clicked(pause_animation)
    ani = animation.FuncAnimation(fig, update, frames=None, interval=interval, blit=False,
                                  cache_frame_data=False)

    # 绘制环形道路
    road_patch1 = patches.Circle((0, 0), road_radius * 0.97, fill=False, color='#DDDDDD', linewidth=18, zorder=0)
    # road_patch2 = patches.Circle((0, 0), road_radius * 1.05, fill=False, color='grey')
    ax.add_patch(road_patch1)
    # ax.add_patch(road_patch2)
    plt.axis('off')  # 隐藏坐标轴

    # ani.save('your_animation.gif', writer='pillow', fps=1)

    plt.show()
    # 不显示动画窗口
    flag = 0

# 创建一个空的DataFrame
df = pd.DataFrame()

# 将数据逐个提取并存储
for density, speeds in average_speeds_by_density.items():
    order = density // 10
    next_density = density % 10
    # 创建一个新的DataFrame，每个密度对应一列
    density_data = pd.DataFrame(speeds, columns=[f"Order = {f'Order{order}_to_{next_density}'}"])

    # 添加到主DataFrame
    df = pd.concat([df, density_data], axis=1)

# 将DataFrame写入CSV文件，选择逗号作为分隔符
df.to_csv('average_speeds_by_multi-density3.csv', sep=',', index=False)

# 创建频率分布直方图5
all_orders = list(average_speeds_by_density.keys())

df = pd.DataFrame()

# 遍历每辆车，将其到达时间信息添加到DataFrame中
for car in cars:
    car_df = pd.DataFrame(car.station_arrival_time, columns=[f'Car_{car.number + 1}_Station', f'Car_{car.number + 1}_ArrivalTime'])
    df = pd.concat([df, car_df], axis=1)

# 将DataFrame写入CSV文件
df.to_csv('car_station_arrival_times10.csv', sep=',', index=False)

for info in all_orders:
    # 从字典中提取适当密度下的速度数据
    order = info // 10
    next_density = info % 10
    speeds_data = average_speeds_by_density[info]

    # 生成频率分布直方图
    plt.hist(speeds_data, bins=20, density=True, alpha=0.7)
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Probability Density")
    plt.title(f"Order = {f'Order{order}_to_{next_density}'}")
    plt.show()
