import math
import sys
import time
import threading
import random
import numpy as np
import pandas as pd
import matplotlib

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


k = 1
road_length = 230
road_radius = road_length / (2 * math.pi)
num_cars = int(38 * 0.4)
car_length = 4

# start_num_cars = 22
# end_num_cars = 45
deceleration = 4
acceleration = 3 * k

begin_interval = 100
end_interval = 1
# interval = 1000
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

warm_up_duration = 30
interval_duration = 600  # 初始值为300秒（5分钟）
all_car_velocities_sets = []
average_velocities_per_cars = []
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


w99_parameters = {
    'cc0': 1.40,  # 静止距离 - 米
    'cc1': 1.20,  # 间距时间 - 秒
    'cc2': 8.00,  # 跟驰变化 - 米
    'cc3': -12.00,  # 进入“跟驰”阈值 - 秒
    'cc4': -0.15,  # 负“跟驰”阈值 - 米/秒
    'cc5': 2.10,  # 正“跟驰”阈值 - 米/秒
    'cc6': 6.00,  # 振荡速度依赖性 - 10^-4 弧度/秒
    'cc7': 0.25,  # 振荡加速度 - 米/秒^2
    'cc8': 2.00,  # 静止加速度 - 米/秒^2
    'cc9': 1.50,  # 80km/h 时的加速度 - 米/秒^2
}

TRACK_RADIUS = 100  # Define the track radius as needed


def car_following(leader, follower, leader_is_car0):
    cc0 = w99_parameters['cc0']
    cc1 = w99_parameters['cc1']
    cc2 = w99_parameters['cc2']
    cc3 = w99_parameters['cc3']
    cc4 = w99_parameters['cc4']
    cc5 = w99_parameters['cc5']
    cc6 = w99_parameters['cc6'] / 10000
    cc7 = w99_parameters['cc7']
    cc8 = w99_parameters['cc8']
    cc9 = w99_parameters['cc9']

    if next_car.position < car.position:
        dx = (leader.position + road_length - follower.position - leader.length)
    else:
        dx = (leader.position - follower.position - leader.length)
    dv = leader.velocity - follower.velocity

    if leader.v <= 0:
        sdxc = cc0
    else:
        v_slower = follower.velocity if (dv >= 0) or (leader.a < -1) else leader.velocity + dv * (simple_random(follower.seed) - 0.5)
        sdxc = cc0 + cc1 * v_slower
    sdxo = sdxc + cc2
    sdxv = sdxo + cc3 * (dv - cc4)

    sdv = cc6 * dx * dx
    sdvc = cc4 - sdv if leader.velocity > 0 else 0
    sdvo = sdv + cc5 if follower.velocity > cc5 else sdv

    follower_a = 0

    if dv < sdvo and dx <= sdxc:
        follower_a = 0
        if follower.v > 0:
            if dv < 0:
                if dx > cc0:
                    follower_a = min(leader.a + dv * dv / (cc0 - dx), follower_a)
                else:
                    follower_a = min(leader.a + 0.5 * (dv - sdvo), follower_a)
            if follower_a > -cc7:
                follower_a = -cc7
            else:
                follower_a = max(follower_a, -10 + 0.5 * math.sqrt(follower.velocity))
    elif dv < sdvc and dx < sdxv:
        follower_a = max(0.5 * dv * dv / (-dx + sdxc - 0.1), -10)
    elif dv < sdvo and dx < sdxo:
        if follower.a <= 0:
            follower_a = min(follower.a, -cc7)
        else:
            follower_a = max(follower.acc, cc7)
            follower_a = min(follower_a, follower.v_desired - follower.velocity)
    else:
        if dx > sdxc:
            if follower.status == 'w':
                follower_a = cc7
            else:
                a_max = cc8 + cc9 * min(follower.velocity, 80 * 1000 / 3600) + simple_random(follower.seed)
                if dx < sdxo:
                    follower_a = min(dv * dv / (sdxo - dx), a_max)
                else:
                    follower_a = a_max
            follower_a = min(follower_a, follower.v_desired - follower.velocity)
    return [follower_a, follower.status]


def simple_random(seed):
    x = math.sin(seed) * 10000
    return x - int(x)


# 示例用法
leader_car = {
    'x': 100,  # Replace with actual values
    'v': 20,
    'a': 2,
    'length': 5
}

follower_car = {
    'x': 110,  # Replace with actual values
    'v': 18,
    'v_desired': 25,
    'status': 'w',  # Replace with actual status
    'seed': random.random()
}

leader_is_car0 = True  # Replace with actual condition

# 调用car_following函数
result = car_following(leader_car, follower_car, leader_is_car0)
print(f"Follower Acceleration: {result[0]}")
print(f"Follower Status: {result[1]}")


# 添加一个函数来初始化模拟
def initialize_simulation():
    global cars, interval_duration, average_velocities, update_count, car_initial_positions, simulation_times, warm_up_phase, desired_spacing

    # 初始化车辆
    car_initial_positions = np.linspace(0, road_length, num_cars, endpoint=False)
    cars = [Car(position, f"C{i + 1}", number=i) for i, position in enumerate(car_initial_positions * 0.7)]

    for i, car in enumerate(cars):
        car.front_car = cars[(i + 1) % num_cars]

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
    # if interval == 1:
    #     ani.event_source.stop()
    #     plt.close()

    patches = []

    current_simulation_time = (update_count - 1) * (interval / 1000)
    update_count += 1
    if frame % (1 * int(1000 / interval)) == 0:
        if not info_printed:
            print(f'第{int(simulation_times)}次模拟')
            print(f"模拟时长: {(update_count - 1) * (interval / 1000):.2f} 秒")
            print("车辆信息：")
            # for car in cars:
            #     print(f"车辆{car.number + 1}:")
            #     print(f"  位置: {car.position:.2f} 米")
            #     print(f"  速度: {car.velocity:.2f} 米/秒")
            #     print(f"  加速度: {car.acceleration} 米/秒²")
            print(f"车辆{cars[0].number + 1}:")
            print(f"  位置: {cars[0].position:.2f} 米")
            print(f"  速度: {cars[0].velocity:.2f} 米/秒")
            print(f"  加速度: {cars[0].acceleration} 米/秒²")
            print()
            info_printed = True
    else:
        info_printed = False

    # for i, car in enumerate(cars):
    #     # 如果在10秒内车辆总共没有超过2次事件
    #     if sum(car.num_events_in_10s for car in cars) < 2:
    #         # 每次更新都有0.01的概率触发随机事件
    #         if np.random.random() < 0.001:
    #             # 如果在10秒内没有超过2次事件
    #             if car.num_events_in_10s < 1:  # 在10秒内最多触发2次事件
    #                 # 以0.5的概率选择是加速还是减速
    #                 car.event_triggered = True
    #                 if np.random.random() < 0.5:
    #                     # 加速
    #                     random_acceleration = np.random.uniform(0.5, 2.5) * k
    #                 else:
    #                     # 减速
    #                     random_acceleration = -np.random.uniform(0.5, 2.5) * k
    #
    #                 # 将加速度限制在 -5 到 5 之间
    #                 car.acceleration = np.clip(random_acceleration, -4.0, 3.0) * k
    #                 car.num_events_in_10s += 1
    #                 print(f"车辆{car.number} 触发了随机事件！加速度变化为: {car.acceleration}")
    #     else:
    #         # 事件发生后3秒重置加速度
    #         if (update_count - car.last_event_update) * interval > 1000:
    #             car.last_event_update = update_count
    #             car.num_events_in_10s = 0
    #             car.event_triggered = False

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
            car.position = (car.position + car.velocity * (interval / 1000) + 0.5 * car.acceleration * (
                    interval / 1000)) % road_length
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
            # desired_speeds = v_max * (1 - (velocities / v_max) ** 4 - (s_0 / spacings - car.T * velocities) ** 2)
            desired_velocities = s_0 + max(0, car.velocity * car.T + (
                    car.velocity * (car.velocity - car.front_car.velocity)) / (
                                                   2 * math.sqrt(acceleration * deceleration)))
            car.acceleration = np.clip(
                acceleration * (
                        1 - (car.velocity / car.max_velocity) ** car.delta - (desired_velocities / spacing) ** 2),
                -deceleration, acceleration)
            # car.acceleration = np.tanh((spacing - 10)) * 3  # 将间距映射到 [0, 1] 范围
            # car.acceleration = (spacing ** 2 / 10000 - 0.5) * 3  # 将间距映射到 [0, 1] 范围

        car.velocity = np.clip(car.velocity + car.acceleration * (interval / 1000), car.min_velocity, car.max_velocity)

        car.cumulative_distance += car.velocity * (interval / 1000) + 0.5 * car.acceleration * (interval / 1000)
        # 检查是否超车
        # if car.cumulative_distance > car.front_car.cumulative_distance + desired_spacing + car_length:
        #     print(f"车辆 {car.number + 1} 已超过车辆 {car.front_car.number + 1}！")
        #     car.velocity = 0
        #     car.acceleration = 0
        #     car.position = car.front_car.position - car_length - s_0
        #     car.cumulative_distance = car.front_car.cumulative_distance - car_length - s_0 + desired_spacing

        # scatter.set_offsets([polar_to_cartesian(car.angle, road_radius) for car in cars])
        # scatter.set_color([car.color for car in cars])

    if (update_count - 1) * (interval / 1000) >= interval_duration:
        ani.event_source.stop()
        plt.close()

    return patches,


# for num_car in range(start_num_cars, end_num_cars, 1):
for interval in range(begin_interval, end_interval, -1):  # 从100到1000，每次增加100
    # while 1:
    #     if interval == 1:
    #         break
    intervals.append(interval)
    # num_cars = num_car
    initialize_simulation()  # 重新初始化模拟
    # interval_duration = 60  # 重置interval_duration
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-1.2 * road_radius, 1.2 * road_radius)
    ax.set_ylim(-1.2 * road_radius, 1.2 * road_radius)
    # scatter = ax.scatter([], [], marker="s")
    for car in cars:
        ax.add_patch(car.car_patch)
    car_labels = [ax.text(0, 0, f'Car {i + 1}', ha='center', va='center', fontsize=8, color='black') for i in
                  range(len(cars))]

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
    # plt.close()

    average_velocities_per_car = [np.mean(car.speed_history) for car in cars]

    # 计算所有车辆的平均速度
    average_velocity_all_cars = np.mean(average_velocities_per_car)
    average_velocities_per_cars.append(average_velocities_per_car)

    # 将所有车辆的平均速度添加到列表中
    all_car_velocities_sets.append(average_velocity_all_cars)
    alpha = 0.05

    # if len(average_velocities_per_cars) >= 2:
    #     # 检查最新的一次模拟和前一次模拟是否有显著差异
    #     group1 = average_velocities_per_cars[-2]  # 倒数第二次模拟的数据
    #     group2 = average_velocities_per_cars[-1]  # 最新一次模拟的数据
    #
    #     # 执行方差分析
    #     f_statistic, p_value = stats.f_oneway(group1, group2)
    #     # 输出检测结果
    #     if p_value < alpha:
    #         print("最新一次模拟和前一次模拟的平均速度显著不同")
    #         interval //= 2
    #     else:
    #         print("最新一次模拟和前一次模拟的平均速度没有显著差异")
    #         break
    # else:
    #     interval //= 2
alpha = 0.05

# 从第1次模拟开始逐个增加次数
# for i in range(len(average_velocities_per_cars) - 1):
#     # 分割数据，包括前面的模拟次数和不包括前面的模拟次数
#     group1 = average_velocities_per_cars[i]
#     group2 = average_velocities_per_cars[i + 1]
#
#     # 执行方差分析
#     f_statistic, p_value = stats.f_oneway(group1, group2)
#
#     # 输出检测结果
#     if p_value < alpha:
#         print(f"从第{i + 2}次模拟开始平均速度显著不同")
#         break

x_data = list(range(begin_interval, end_interval, -1))
# x_data = list(range(start_num_cars, end_num_cars, 1))
# 将数据转换为NumPy数组

df = pd.DataFrame(average_velocities_per_cars)

# 将DataFrame保存为CSV文件
df.to_csv('average_velocities_per_cars.csv', index=False)
print(all_car_velocities_sets)
print(intervals)
plt.figure()
plt.semilogx(x_data, all_car_velocities_sets)  # 对数化 X 轴
# plt.plot(x_data, all_car_velocities_sets)
# plt.scatter(intervals, all_car_velocities_sets, marker='o')
plt.xlabel("Interval (ms)(log)")
# plt.xlabel("num_cars")
plt.ylabel("Average Velocity (m/s)")
plt.title("Avg_Spd_Curve")
plt.ylim(0, 8.33)  # 修改 Y 轴区间
plt.grid(False)
plt.show()
