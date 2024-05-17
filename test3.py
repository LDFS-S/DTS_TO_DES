import math
import sys
import time
import threading
import random
import numpy as np
import pandas as pd
import matplotlib
import csv
import heapq

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import io
import scipy.stats as stats


# matplotlib.use('Agg')
class Station:
    def __init__(self, position, index):
        self.position = position
        self.car_count = 0
        self.index = index
        self.car_count_text = None
        self.waiting_queue = []

    def update_car_count_text(self, ax):
        x, y = polar_to_cartesian(self.position * 360 / road_length, road_radius * 0.65)
        car_count_text = f'C_N: {self.car_count}'

        if self.car_count_text:
            self.car_count_text.remove()

        station_text = ax.text(x, y, car_count_text, ha='center', va='center', fontsize=8, color='black')
        self.car_count_text = station_text


class Event:
    def __init__(self, time, event_type, car):
        self.time = time
        self.event_type = event_type
        self.car = car

    def __lt__(self, other):
        # 自定义事件比较方法，比较事件的时间
        return self.time < other.time


class Car:
    def __init__(self, initial_position, color, number):
        self.angle = 0.00
        self.position = initial_position
        self.color = color
        self.number = number
        self.car_patch = patches.Rectangle((initial_position, 0), 4, 2, color=self.color)
        self.current_station = None
        self.order = 0


road_length = 230
road_radius = road_length / (2 * math.pi)
num_cars = int(38 * 0.8)
car_length = 4
num_stations = 8

interval = 100
simulation_times = 0
warm_up_phase = True
average_velocities = []
station_length = road_length / num_stations
update_count = 0
info_printed = False

average_speeds_by_density = {0: []}
cars = []
buf = io.BytesIO()
stations = []
k = 5

fig, ax = plt.subplots()
car_labels = [ax.text(0, 0, f'C{i + 1}', ha='center', va='center', fontsize=8, color='black') for i in
              range(0, 30, 1)]


def play_animation(event):
    ani.event_source.start()


def pause_animation(event):
    ani.event_source.stop()


def polar_to_cartesian(angle, radius):
    x = radius * np.cos(np.deg2rad(angle))
    y = radius * np.sin(np.deg2rad(angle))
    return x, y


def generate_speed(order_density):
    while True:
        if order_density == 0:
            mean_speed = 5.114205945266666
            std_dev = 3.946863947541551
        elif order_density == 1:
            mean_speed = 4.278285963619047
            std_dev = 2.2397785311431164
        elif order_density == 2:
            mean_speed = 2.8742288525075077
            std_dev = 0.5183311497456567
        elif order_density == 3:
            mean_speed = 2.230465426242904
            std_dev = 0.35299374430576996
        elif order_density == 4:
            mean_speed = 2.2492926779808347
            std_dev = 0.3582031432246253
        elif order_density == 5:
            mean_speed = 2.120087469979669
            std_dev = 0.46468562044381395
        elif order_density == 11:
            mean_speed = 4.460792312969136
            std_dev = 1.3339877471403225
        elif order_density == 12:
            mean_speed = 3.1005994259947354
            std_dev = 0.5009400849921266
        elif order_density == 13:
            mean_speed = 2.2465205078885444
            std_dev = 0.3743821025358046
        elif order_density == 14:
            mean_speed = 2.243462834076487
            std_dev = 0.3591710303809329
        elif order_density == 15:
            mean_speed = 2.0452810878035623
            std_dev = 0.43477713043309374
        elif order_density == 21:
            mean_speed = 5.32116684554369
            std_dev = 0.7661712009389063
        elif order_density == 22:
            mean_speed = 3.2008813694476563
            std_dev = 0.45391690509926813
        elif order_density == 23:
            mean_speed = 2.253772008675885
            std_dev = 0.3707400619422026
        elif order_density == 24:
            mean_speed = 2.2434437202827704
            std_dev = 0.35850015146537606
        elif order_density == 25:
            mean_speed = 2.3035882817840374
            std_dev = 0.3713966058076698
        elif order_density == 31:
            mean_speed = 2.9440437733825684
            std_dev = 0.3823156089964257
        elif order_density == 32:
            mean_speed = 2.9440437733825684
            std_dev = 0.3823156089964257
        elif order_density == 33:
            mean_speed = 2.259380613341976
            std_dev = 0.3505721913581803
        elif order_density == 34:
            mean_speed = 2.2234316497489446
            std_dev = 0.35370815490402496
        elif order_density == 35:
            mean_speed = 2.151267957901639
            std_dev = 0.29296515224008246
        elif order_density == 41:
            mean_speed = 2.419291050027027
            std_dev = 0.4525764334593125
        elif order_density == 42:
            mean_speed = 2.419291050027027
            std_dev = 0.4525764334593125
        elif order_density == 43:
            mean_speed = 2.15235964633026
            std_dev = 0.32696290195958955
        elif order_density == 44:
            mean_speed = 1.9110704156588239
            std_dev = 0.2944352185044185
        elif order_density == 45:
            mean_speed = 1.6065880797241068
            std_dev = 0.2986728892260327
        elif order_density == 51:
            mean_speed = 1.6805616568000001
            std_dev = 0.18983218185506565
        elif order_density == 52:
            mean_speed = 1.6805616568000001
            std_dev = 0.18983218185506565
        elif order_density == 53:
            mean_speed = 1.5839512520526313
            std_dev = 0.170782626145043
        elif order_density == 54:
            mean_speed = 1.5174037679145302
            std_dev = 0.32361150412208217
        elif order_density == 55:
            mean_speed = 1.4653246006212486
            std_dev = 0.3259503587810072
        else:
            print(order_density, "速度不在list映射中")
            sys.exit(0)
        # 生成速度
        speed = random.normalvariate(mean_speed, std_dev)
        if speed > 0:
            break

    # 确保速度不低于5
    return speed


def initialize_simulation_with_custom_assignment(num_cars, num_stations):
    global cars, stations, car_labels
    car_initial_positions = np.linspace(0, road_length, num_cars, endpoint=False)

    # 初始化站点
    station_positions = np.linspace(0, road_length, num_stations, endpoint=False)
    stations = [Station(position, index=i) for i, position in enumerate(station_positions)]

    # 初始化车辆

    car_idx = 0  # 车辆编号

    # 定义车辆与站点的对应关系
    car_station_assignment = {
        1: 1, 2: 1, 3: 1, 4: 1,
        5: 2, 6: 2, 7: 2, 8: 2,
        9: 3, 10: 3, 11: 3, 12: 3,
        13: 4, 14: 4, 15: 4,
        16: 5, 17: 5, 18: 5, 19: 5,
        20: 6, 21: 6, 22: 6, 23: 6,
        24: 7, 25: 7, 26: 7, 27: 7,
        28: 8, 29: 8, 30: 8
    }

    # 初始化车辆
    for car_number in range(1, num_cars + 1):
        car_position = car_initial_positions[car_number - 1]
        station_index = car_station_assignment.get(car_number, 1)  # 默认分配到站点1

        # 找到对应的站点对象
        assigned_station = stations[station_index - 1]

        # 创建车辆对象，并分配到对应站点
        car = Car(car_position, f"C{car_number}", number=car_number)

        car.current_station = assigned_station
        assigned_station.car_count += 1
        cars.append(car)
        car.order = car.current_station.car_count  # 设置order属性
        car.position = car.current_station.position
        car.angle = (car.position / road_length) * 360

    # 对车辆的位置和顺序进行排列
    for station in stations:
        cars_at_station = [car for car in cars if car.current_station == station]
        cars_at_station.sort(key=lambda x: x.number, reverse=True)  # 按order属性排序
        for i, sorted_car in enumerate(cars_at_station):
            sorted_car.order = i
            x, y = polar_to_cartesian(sorted_car.angle, road_radius + (sorted_car.order + 1) * k)
            sorted_car.car_patch.set_xy((x, y))
            sorted_car.car_patch.angle = sorted_car.angle + 90
            car_labels[sorted_car.number - 1].set_position((x, y))
            ax.add_patch(sorted_car.car_patch)  # 使用 sorted_car 而不是 car


initialize_simulation_with_custom_assignment(num_cars, num_stations)
# 初始化每辆车的到站时间列表
arrival_times = {car: [] for car in cars}
car_info = {car: [] for car in cars}

global_event_queue = []

# 添加初始到站事件
# 初始化事件
for car in cars:
    initial_arrival_time = None
    while initial_arrival_time is None:
        current_station = car.current_station
        next_station = stations[(car.current_station.index + 1) % num_stations]
        after_next_station = stations[(car.current_station.index + 2) % num_stations]
        order_density = next_station.car_count * 10 + after_next_station.car_count
        speed = generate_speed(order_density)
        travel_time = station_length / speed
        next_arrival_time = travel_time

        # 检查是否有在同一站点的车辆比这个时间更早到达
        earlier_arrivals = [event.time for event in global_event_queue if
                            event.car.current_station == car.current_station]
        if not earlier_arrivals or next_arrival_time < min(earlier_arrivals):
            initial_arrival_time = next_arrival_time

    arrival_times[car].append(initial_arrival_time)  # 初始到站时间为0.0
    global_event_queue.append(Event(initial_arrival_time, "Arrival", car))

heapq.heapify(global_event_queue)
# 主模拟循环
iteration_count = 0
current_time = 0


# 寻找下一个要处理的事件
def find_next_event():
    if global_event_queue:
        next_event = min(global_event_queue)
        global_event_queue.remove(next_event)
    else:
        next_event = None
    return next_event


# 更新函数
def update(frame):
    global update_count, info_printed, cars, interval_duration, warm_up_phase, global_event_queue, iteration_count, current_time, stations
    max_cars_per_station = 5

    next_event = find_next_event()
    if next_event is not None:
        current_time = next_event.time
        car = next_event.car

        if next_event.event_type == "Arrival":
            current_station = car.current_station
            next_station = stations[(car.current_station.index + 1) % num_stations]
            after_next_station = stations[(car.current_station.index + 2) % num_stations]

            if next_station.car_count < max_cars_per_station:
                print(
                    f"Car {car.number} arrived at Station {car.current_station.index + 1} at simulation time {current_time:.2f}")
                # 车辆离开站点时，当前站点车辆前往下个站点的事件时间不会重新计算，导致后面再进站的车辆计算的下个到达事件时间是根据当前密度来判断的，所以可能使用比前车更短的时间
                counter = 0
                while True:
                    order_density = next_station.car_count * 10 + after_next_station.car_count
                    speed = generate_speed(order_density)
                    travel_time = station_length / speed
                    next_arrival_time = current_time + travel_time

                    earlier_arrivals = [event.time for event in global_event_queue if
                                        event.car.current_station == current_station]
                    if not earlier_arrivals or next_arrival_time > max(earlier_arrivals):
                        break

                    counter += 1
                    if counter >= 1000000:
                        print(order_density)
                        print("速度分布不满足")
                        exit()

                simulation_time = arrival_times[car][-1] + travel_time

                car_information = (car.current_station.index + 1, simulation_time)
                car_info[car].append(car_information)
                arrival_times[car].append(simulation_time)

                car.current_station.car_count -= 1
                next_station.car_count += 1
                car.current_station.update_car_count_text(ax)
                next_station.update_car_count_text(ax)
                car.current_station = next_station

                car.order = next_station.car_count
                car.angle = (car.current_station.position / road_length) * 360
                x, y = polar_to_cartesian(car.angle, road_radius + car.order * k)
                car.car_patch.set_xy((x, y))
                car.car_patch.angle = car.angle + 90
                car_labels[car.number - 1].set_position((x, y))

                global_event_queue.append(Event(next_arrival_time, "Arrival", car))
                heapq.heapify(global_event_queue)
                sorted_cars = sorted([car for car in cars if car.current_station == current_station],
                                     key=lambda x: x.order)

                for i, sorted_car in enumerate(sorted_cars):
                    sorted_car.order = i
                    x, y = polar_to_cartesian(current_station.position * 360 / road_length, road_radius + (i + 1) * k)
                    sorted_car.car_patch.set_xy((x, y))
                    car_labels[sorted_car.number - 1].set_position((x, y))

                if current_station.waiting_queue:
                    car_to_arrive = current_station.waiting_queue.pop(0)
                    counter = 0
                    while True:
                        order_density = current_station.car_count * 10 + next_station.car_count
                        speed = generate_speed(order_density)
                        travel_time = station_length / speed
                        next_arrival_time = current_time + travel_time

                        earlier_arrivals = [event.time for event in global_event_queue if
                                            event.car.current_station == current_station]
                        if not earlier_arrivals or next_arrival_time > max(earlier_arrivals):
                            break

                        counter += 1
                        if counter >= 100000:
                            print(order_density)
                            print("速度分布不满足")
                            exit()

                    simulation_time = arrival_times[car_to_arrive][-1] + travel_time

                    car_to_arrive.current_station.car_count -= 1
                    current_station.car_count += 1
                    car_to_arrive.current_station.update_car_count_text(ax)
                    current_station.update_car_count_text(ax)
                    previous_station = car_to_arrive.current_station
                    car_to_arrive.current_station = current_station

                    car_to_arrive.order = car_to_arrive.current_station.car_count
                    car_to_arrive.angle = (car_to_arrive.current_station.position / road_length) * 360
                    x, y = polar_to_cartesian(car_to_arrive.angle, road_radius + car_to_arrive.order * k)
                    car_to_arrive.car_patch.set_xy((x, y))
                    car_to_arrive.car_patch.angle = car_to_arrive.angle + 90
                    car_labels[car_to_arrive.number - 1].set_position((x, y))

                    car_information = (car_to_arrive.current_station.index + 1, simulation_time)
                    car_info[car_to_arrive].append(car_information)
                    arrival_times[car_to_arrive].append(simulation_time)

                    global_event_queue.append(Event(next_arrival_time, "Arrival", car_to_arrive))
                    heapq.heapify(global_event_queue)
                    sorted_cars = sorted([car for car in cars if car.current_station == previous_station],
                                         key=lambda x: x.order)

                    for i, sorted_car in enumerate(sorted_cars):
                        sorted_car.order -= 1
                        x, y = polar_to_cartesian(previous_station.position * 360 / road_length,
                                                  road_radius + (i + 1) * k)
                        sorted_car.car_patch.set_xy((x, y))
                        car_labels[sorted_car.number - 1].set_position((x, y))

            else:
                # 下一个站点已满，将车辆放入下一个站点的等待队列
                next_station.waiting_queue.append(car)
                print(
                    f"Car {car.number} arrived at waiting queue {car.current_station.index + 1} at simulation time {current_time:.2f}")

    iteration_count += 1

    return []


ax.set_aspect('equal')
ax.set_xlim(-2 * road_radius, 2 * road_radius)
ax.set_ylim(-2 * road_radius, 2 * road_radius)
road_radius = road_length / (2 * math.pi)  # 道路半径
angles = np.linspace(0, 2 * np.pi, num_stations, endpoint=False)  # 均匀分布的角度
station_x = road_radius * 0.98 * np.cos(angles)  # 计算X坐标
station_y = road_radius * 0.98 * np.sin(angles)  # 计算Y坐标

# 绘制站点
for x, y in zip(station_x, station_y):
    station_circle = plt.Circle((x, y), 1, color='gray', zorder=2)
    plt.gca().add_patch(station_circle)

# 添加站点标签
for i, station in enumerate(stations):
    x, y = polar_to_cartesian(station.position * 360 / road_length, road_radius * 0.98)
    plt.text(x, y, f'S {station.index + 1}', ha='center', va='center', fontsize=10, color='black')

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
ax.add_patch(road_patch1)
plt.axis('off')  # 隐藏坐标轴

plt.show()

# CSV文件名
csv_filename = "car_info.csv"

# Write to CSV file
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header
    header = ["Car Index", "Station Index", "Arrival Time"]
    writer.writerow(header)

    # Write each car's arrival times
    for car, information in car_info.items():
        car_index = car.number
        for station_index, arrival_time in information:
            row = [car_index, station_index, arrival_time]
            writer.writerow(row)
