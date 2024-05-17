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
import scipy.stats as stats

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import io


# matplotlib.use('Agg')
class Station:
    def __init__(self, position, index):
        self.position = position
        self.car_count = 0
        self.index = index
        self.car_count_text = None
        self.waiting_queue = []

    def update_car_count_text(self, update_car_count_ax):
        update_car_count_x, update_car_count_y = polar_to_cartesian(self.position * 360 / road_length, road_radius * 0.65)
        car_count_text = f'C_N: {self.car_count}'

        if self.car_count_text:
            self.car_count_text.remove()

        station_text = update_car_count_ax.text(update_car_count_x, update_car_count_y, car_count_text, ha='center', va='center', fontsize=8, color='black')
        self.car_count_text = station_text


class Event:
    def __init__(self, event_time, event_type, car):
        self.event_time = event_time
        self.event_type = event_type
        self.car = car

    def __lt__(self, other):
        # 自定义事件比较方法，比较事件的时间
        return self.event_time < other.event_time


class Car:
    def __init__(self, initial_position, color, number):
        self.angle = 0.00
        self.position = initial_position
        self.color = color
        self.number = number
        self.car_patch = patches.Rectangle((initial_position, 0), 4, 2, color=self.color)
        self.current_station = None
        self.order = 0
        self.velocity = 0
        self.previous_velocity = 0
        self.last_update_time = 0


class SpeedParameters:
    def __init__(self):
        # 定义不同密度等级下的速度均值和标准差
        self.parameters = {
            0: (5.114205945266666, 3.946863947541551),
            1: (4.278285963619047, 2.2397785311431164),
            2: (2.8742288525075077, 0.5183311497456567),
            3: (2.230465426242904, 0.35299374430576996),
            4: (2.2492926779808347, 0.3582031432246253),
            5: (2.120087469979669, 0.46468562044381395),
            10: (5.0823, 2.2398),
            11: (4.460792312969136, 1.3339877471403225),
            12: (3.1005994259947354, 0.5009400849921266),
            13: (2.2465205078885444, 0.3743821025358046),
            14: (2.243462834076487, 0.3591710303809329),
            15: (2.0452810878035623, 0.43477713043309374),
            20: (4.2783, 0.5183),
            21: (4.02116684554369, 0.7661712009389063),
            22: (3.2008813694476563, 0.45391690509926813),
            23: (2.253772008675885, 0.3707400619422026),
            24: (2.2434437202827704, 0.35850015146537606),
            25: (2.3035882817840374, 0.3713966058076698),
            30: (3.7663, 0.3530),
            31: (2.9440437733825684, 0.3823156089964257),
            32: (2.9440437733825684, 0.3823156089964257),
            33: (2.259380613341976, 0.3505721913581803),
            34: (2.2234316497489446, 0.35370815490402496),
            35: (2.151267957901639, 0.29296515224008246),
            40: (3.3906, 0.3582),
            41: (2.419291050027027, 0.4525764334593125),
            42: (2.419291050027027, 0.4525764334593125),
            43: (2.15235964633026, 0.32696290195958955),
            44: (1.9110704156588239, 0.2944352185044185),
            45: (1.6065880797241068, 0.2986728892260327),
        }

    def get(self, parameters_od):
        # 返回给定密度等级的速度参数（均值和标准差）
        if parameters_od not in self.parameters:
            raise ValueError(f"Speed parameters for order density {parameters_od} not found.")
        return self.parameters.get(parameters_od)


speed_parameters = SpeedParameters()
road_length = 230
road_radius = road_length / (2 * math.pi)
num_cars = int(38 * 0.8)
car_length = 4
num_stations = 8

interval = 100
simulation_times = 0
warm_up_phase = True
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
    ptc_x = radius * np.cos(np.deg2rad(angle))
    ptc_y = radius * np.sin(np.deg2rad(angle))
    return ptc_x, ptc_y


def calculate_circular_distance(pos1, pos2, total_length):
    """Calculate the shortest distance between two points on a circular path.

    Args:
        pos1 (float): Position of the first point.
        pos2 (float): Position of the second point.
        total_length (float): Total length of the circular path.

    Returns:
        float: The shortest distance between pos1 and pos2 on the circular path.
    """
    direct_distance = abs(pos2 - pos1)
    return min(direct_distance, total_length - direct_distance)


def calculate_max_speed_based_on_order_density(max_order_density, car, current_time):
    mean_speed, std_dev = speed_parameters.get(max_order_density)

    previous_car = None

    # 如果没有前车，返回稍微低于理论最大速度的值
    if car.order == 0:
        return None

    flag = 0

    # 查找前车及其到达时间
    while flag == 0:
        for potential_previous_car in cars:
            if potential_previous_car.current_station == car.current_station and potential_previous_car.order == car.order - 1:
                previous_car = potential_previous_car
                flag = 1
                # print(car.number, previous_car.number)
                break
        else:
            print("未找到前车")
            sys.exit()

    # 计算前车的到达时间
    for event in global_event_queue:
        if event.car == previous_car and event.event_type == "Arrival":
            earliest_time_previous_car = event.event_time
            break
    else:
        return None
    # 计算当前车辆到达相同位置的最短时间
    time_to_reach_same_position = earliest_time_previous_car - current_time

    # 确保时间不是负数或零（避免除以零错误）
    if time_to_reach_same_position <= 0:
        return mean_speed + 3 * std_dev

    # 计算最大可能速度
    distance_to_next_car = calculate_circular_distance(previous_car.position, car.position, road_length)

    # 计算到达前车当前位置之前的最短时间
    time_to_reach_same_position = earliest_time_previous_car - current_time

    # 计算最大速度，确保在此时间内不追上前车
    max_speed_space_based = (distance_to_next_car + previous_car.velocity * time_to_reach_same_position) / time_to_reach_same_position
    # print("max", min(max_speed_space_based, mean_speed + 3 * std_dev), car.number)
    # 确保计算出的速度在合理范围内
    return min(max_speed_space_based, mean_speed + 3 * std_dev)


def calculate_min_speed_based_on_order_density(min_order_density, car, current_time):
    # print(min_order_density)
    mean_speed, std_dev = speed_parameters.get(min_order_density)

    min_next_station = stations[(car.current_station.index + 1) % num_stations]

    # 如果是最后一辆车，返回稍微高于理论最低速度的值
    if car.order == len([c for c in cars if c.current_station == car.current_station]) - 1:
        return mean_speed - 1 * std_dev

    # 查找后车及其最慢可能速度
    for potential_next_car in cars:
        if potential_next_car.current_station == car.current_station and potential_next_car.order == car.order + 1:
            next_car = potential_next_car
            next_order_density = next_car.order * 10 + stations[
                (next_car.current_station.index + 1) % num_stations].car_count
            next_mean_speed, next_std_dev = speed_parameters.get(next_order_density)
            slowest_speed_next_car = next_mean_speed - 1 * next_std_dev if next_mean_speed is not None else mean_speed - 1 * std_dev
            break
    else:
        # 如果未找到后车，返回稍微高于理论最低速度的值
        return mean_speed - 1 * std_dev

    # 计算后车的最早可能到达时间
    time_next_car_earliest = current_time + calculate_circular_distance(min_next_station.position, next_car.position, road_length) / slowest_speed_next_car

    # 计算当前车辆到达相同位置的最长时间
    time_to_reach_same_position = time_next_car_earliest - current_time

    # 确保时间不是负数或零（避免除以零错误）
    if time_to_reach_same_position <= 0:
        return mean_speed - 1 * std_dev

    # 计算最小可能速度
    distance_to_next_station = calculate_circular_distance(min_next_station.position, car.position, road_length)
    cal_min_speed = distance_to_next_station / time_to_reach_same_position

    # 确保计算出的速度在合理范围内
    # print("min", max(cal_min_speed, mean_speed - 3 * std_dev), car.number, "\n")
    return max(cal_min_speed, mean_speed - 3 * std_dev)


def generate_speed(final_order_density, final_min_speed=None, final_max_speed=None):
    mean_speed, std_dev = speed_parameters.get(final_order_density)

    if mean_speed is None or std_dev is None:
        print(f"Order density {final_order_density} is not mapped in the distribution.")
        sys.exit(0)

    if std_dev <= 0:
        print("Invalid standard deviation value. It must be positive.")
        sys.exit(0)

    if final_min_speed is not None and final_max_speed is not None:
        if final_min_speed > final_max_speed:
            print("Minimum speed is greater than maximum speed. Adjusting values.")
            final_min_speed = None

    # 设置截断的边界
    lower_bound = -float('inf') if final_min_speed is None else (final_min_speed - mean_speed) / std_dev
    upper_bound = float('inf') if final_max_speed is None else (final_max_speed - mean_speed) / std_dev

    # 创建截断正态分布
    trunc_normal_dist = stats.truncnorm(lower_bound, upper_bound, loc=mean_speed, scale=std_dev)

    # 生成一个随机数
    speed = trunc_normal_dist.rvs()
    print(f"Generated speed: {speed}")
    return speed


def initialize_simulation_with_custom_assignment(initialize_num_cars, initialize_num_stations):
    global cars, stations, car_labels
    car_initial_positions = np.linspace(0, road_length, initialize_num_cars, endpoint=False)

    # 初始化站点
    station_positions = np.linspace(0, road_length, initialize_num_stations, endpoint=False)
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
        initialize_station_index = car_station_assignment.get(car_number, 1)  # 默认分配到站点1

        # 找到对应的站点对象
        assigned_station = stations[initialize_station_index - 1]

        # 创建车辆对象，并分配到对应站点
        car = Car(car_position, f"C{car_number}", number=car_number)

        car.current_station = assigned_station
        assigned_station.car_count += 1
        cars.append(car)
        car.order = car.current_station.car_count  # 设置order属性
        car.angle = (car.current_station.position / road_length) * 360

    # 对车辆的位置和顺序进行排列
    for ini_station in stations:
        cars_at_station = [car for car in cars if car.current_station == ini_station]
        cars_at_station.sort(key=lambda e: e.number, reverse=True)  # 按order属性排序
        for i, sorted_car in enumerate(cars_at_station):
            sorted_car.order = i

            ini_x, ini_y = polar_to_cartesian(sorted_car.angle, road_radius + (sorted_car.order + 1) * k)
            sorted_car.car_patch.set_xy((ini_x, ini_y))
            sorted_car.car_patch.angle = sorted_car.angle + 90
            car_labels[sorted_car.number - 1].set_position((ini_x, ini_y))
            ax.add_patch(sorted_car.car_patch)


initialize_simulation_with_custom_assignment(num_cars, num_stations)
# 初始化每辆车的到站时间列表
arrival_times = {car: [] for car in cars}
car_info = {car: [] for car in cars}

global_event_queue = []

# 添加初始到站事件
# 初始化事件
sorted_cars = sorted(cars, key=lambda ie: ie.number, reverse=True)

# 对每辆车执行操作
for car in sorted_cars:
    initial_arrival_time = None

    while initial_arrival_time is None:
        current_station = car.current_station
        next_station = stations[(current_station.index + 1) % num_stations]
        after_next_station = stations[(current_station.index + 2) % num_stations]
        order_density = car.order * 10 + next_station.car_count

        # 初始化最小和最大速度
        min_speed = None
        max_speed = None

        # 如果不是当前站点的第一辆车，则计算最大速度
        if car.order > 0:  # 如果不是第一辆车
            max_speed = calculate_max_speed_based_on_order_density(order_density, car, 0)

        # 如果不是当前站点的最后一辆车，则计算最小速度
        if car.order < len([c for c in cars if c.current_station == current_station]):  # 检查是否不是最后一辆车
            min_speed = calculate_min_speed_based_on_order_density(order_density, car, 0)

        # 生成车辆的速度
        car.velocity = generate_speed(order_density, min_speed, max_speed)
        travel_time = calculate_circular_distance((car.current_station.position + station_length) % road_length,
                                                  car.position, road_length) / car.velocity
        next_arrival_time = travel_time

        # 检查是否有在同一站点的车辆比这个时间更早到达
        earlier_arrivals = [event.event_time for event in global_event_queue if event.car.current_station == current_station]
        if not earlier_arrivals or next_arrival_time > max(earlier_arrivals):
            initial_arrival_time = next_arrival_time

    arrival_times[car].append(0)  # 初始到站时间为0.0
    global_event_queue.append(Event(initial_arrival_time, "Arrival", car))

heapq.heapify(global_event_queue)
# 主模拟循环
iteration_count = 0


def recalculate_arrival_time_for_station(recalculate_arrival_station, current_time):
    global global_event_queue

    # 获取该站点的所有车辆并按照编号排序（从大到小）
    station_cars = sorted([car for car in cars if car.current_station == recalculate_arrival_station],
                          key=lambda re: re.order)
    re_next_station = stations[(recalculate_arrival_station.index + 1) % num_stations]
    for car in station_cars:
        if car not in re_next_station.waiting_queue:
            car.previous_velocity = car.velocity

            # 初始化 previous_car 和 next_car 变量
            previous_car = None
            next_car = None

            time_elapsed = current_time - car.last_update_time
            car.position = (car.position + car.velocity * time_elapsed) % road_length
            car.last_update_time = current_time  # 更新上次更新时间

            # 计算最大和最小速度限制
            recalculate_arrival_order_density_min_speed = None
            recalculate_arrival_order_density_max_speed = None
            recalculate_arrival_order_density = car.order * 10 + (stations[(car.current_station.index + 1) % num_stations]).car_count

            if car.order > 0:
                previous_car = station_cars[station_cars.index(car) - 1]
                recalculate_arrival_order_density_max_speed = calculate_max_speed_based_on_order_density(recalculate_arrival_order_density, car, current_time)

            if car.order < len(station_cars) - 1:
                recalculate_arrival_order_density_min_speed = calculate_min_speed_based_on_order_density(recalculate_arrival_order_density, car, current_time)

            car.velocity = generate_speed(recalculate_arrival_order_density, recalculate_arrival_order_density_min_speed,
                                          recalculate_arrival_order_density_max_speed)
            recalculate_arrival_travel_time = calculate_circular_distance(
                (car.current_station.position + station_length % road_length), car.position, road_length) / car.velocity
            new_arrival_time = current_time + recalculate_arrival_travel_time
            check_for_overtaking(car, new_arrival_time)

            # 更新车辆的事件
            for event in global_event_queue:
                if event.car == car:
                    global_event_queue.remove(event)
                    break

            # 添加新地到达事件
            new_event = Event(new_arrival_time, "Arrival", car)
            heapq.heappush(global_event_queue, new_event)
            # heapq.heapify(global_event_queue)


# 寻找下一个要处理的事件
def find_next_event():
    if global_event_queue:
        next_event = min(global_event_queue)
        global_event_queue.remove(next_event)
    else:
        next_event = None
    return next_event


def check_for_overtaking(car, check_next_arrival_time):
    # 获取当前车辆的前车
    previous_car = next((c for c in cars if c.current_station == car.current_station and c.order == car.order - 1),
                        None)

    if previous_car:
        # 获取前车的预计到达时间
        previous_car_arrival_time = next(
            (event.event_time for event in global_event_queue if event.car == previous_car and event.event_type == "Arrival"),
            None)

        if not previous_car_arrival_time:
            print("检测无效")
        if previous_car_arrival_time and check_next_arrival_time < previous_car_arrival_time:
            print(f"Potential overtaking detected: Car {car.number} (Order {car.order}) would arrive before Car {previous_car.number} (Order {previous_car.order}). Adjusting speed or delaying arrival.")
            # sys.exit()
            return False
    elif car.order != 0:
        print("未找到前车？？")
    return True


# 更新函数
def update(frame):
    global update_count, info_printed, cars, warm_up_phase, global_event_queue, iteration_count, stations
    max_cars_per_station = 5

    next_event = find_next_event()
    if next_event is not None:
        current_time = next_event.event_time
        car = next_event.car
        # if car.order != 0:
        # print("车辆的order不为0", car.number)
        if current_time >= 1000:
            ani.event_source.stop()
            plt.close()
        if next_event.event_type == "Arrival":
            up_current_station = car.current_station
            up_next_station = stations[(car.current_station.index + 1) % num_stations]
            up_after_next_station = stations[(car.current_station.index + 2) % num_stations]

            if up_next_station.car_count < max_cars_per_station:

                print(
                    f"Car {car.number} arrived at Station {up_next_station.index + 1} at simulation time {current_time:.2f}")
                # 车辆离开站点时，当前站点车辆前往下个站点的事件时间不会重新计算，导致后面再进站的车辆计算的下个到达事件时间是根据当前密度来判断的，所以可能使用比前车更短的时间

                car_information = (up_next_station.index + 1, current_time, car.velocity, car.position)
                car_info[car].append(car_information)
                arrival_times[car].append(current_time)

                up_current_station.car_count -= 1
                up_next_station.car_count += 1
                up_current_station.update_car_count_text(ax)
                up_next_station.update_car_count_text(ax)
                car.current_station = up_next_station

                car.order = up_next_station.car_count - 1

                car.position = car.current_station.position
                car.last_update_time = current_time

                car.angle = (car.position / road_length) * 360
                x, y = polar_to_cartesian(car.angle, road_radius + (car.order + 1) * k)
                car.car_patch.set_xy((x, y))
                car.car_patch.angle = car.angle + 90
                car_labels[car.number - 1].set_position((x, y))

                update_order_density = car.order * 10 + up_after_next_station.car_count

                station_cars = sorted([car for car in cars if car.current_station == station], key=lambda x: x.order)
                for station_car in station_cars:
                    time_elapsed = current_time - station_car.last_update_time
                    station_car.position = (station_car.position + station_car.velocity * time_elapsed) % road_length
                    station_car.last_update_time = current_time  # 更新上次更新时间

                car_max = calculate_max_speed_based_on_order_density(update_order_density, car, current_time)
                car.velocity = generate_speed(update_order_density, final_max_speed=car_max)

                update_travel_time = station_length / car.velocity
                up_next_arrival_time = current_time + update_travel_time
                check_for_overtaking(car, up_next_arrival_time)
                global_event_queue.append(Event(up_next_arrival_time, "Arrival", car))

                heapq.heapify(global_event_queue)
                update_sorted_cars = sorted([car for car in cars if car.current_station == up_current_station],key=lambda up: up.order)

                for i, sorted_car in enumerate(update_sorted_cars):
                    sorted_car.order = i

                    up_x, up_y = polar_to_cartesian(up_current_station.position * 360 / road_length, road_radius + (i + 1) * k)
                    sorted_car.car_patch.set_xy((up_x, up_y))
                    car_labels[sorted_car.number - 1].set_position((up_x, up_y))

                recalculate_arrival_time_for_station(up_current_station, current_time)
                waiting_station_index = up_current_station.index

                if up_current_station.waiting_queue:
                    while True:
                        waiting_station = stations[waiting_station_index]
                        car_to_arrive = waiting_station.waiting_queue.pop(0)
                        previous_station = car_to_arrive.current_station
                        waiting_next_station = stations[(waiting_station_index + 1) % num_stations]

                        print(f"Car {car_to_arrive.number} arrived at Station {waiting_station.index + 1} at simulation time {current_time:.2f}")
                        previous_station.car_count -= 1
                        waiting_station.car_count += 1
                        waiting_station.update_car_count_text(ax)
                        previous_station.update_car_count_text(ax)

                        car_to_arrive.current_station = waiting_station
                        car_to_arrive.position = car_to_arrive.current_station.position
                        car_to_arrive.last_update_time = current_time

                        car_to_arrive.order = waiting_station.car_count - 1

                        car_to_arrive.angle = (car_to_arrive.position / road_length) * 360
                        x, y = polar_to_cartesian(car_to_arrive.angle, road_radius + (car_to_arrive.order + 1) * k)
                        car_to_arrive.car_patch.set_xy((x, y))
                        car_to_arrive.car_patch.angle = car_to_arrive.angle + 90
                        car_labels[car_to_arrive.number - 1].set_position((x, y))

                        car_information = (up_current_station.index + 1, current_time, car_to_arrive.velocity, car_to_arrive.position)
                        car_info[car_to_arrive].append(car_information)
                        arrival_times[car_to_arrive].append(current_time)

                        previous_sorted_cars = sorted([car for car in cars if car.current_station == previous_station],
                                                      key=lambda cta: cta.order)

                        for i, sorted_car in enumerate(previous_sorted_cars):
                            sorted_car.order = i

                            cta_x, cta_y = polar_to_cartesian(previous_station.position * 360 / road_length, road_radius + (i + 1) * k)
                            sorted_car.car_patch.set_xy((cta_x, cta_y))
                            car_labels[sorted_car.number - 1].set_position((cta_x, cta_y))

                        recalculate_arrival_time_for_station(previous_station, current_time)

                        car_to_arrive_order_density = car_to_arrive.order * 10 + waiting_next_station.car_count
                        MAX_CAR_TO = calculate_max_speed_based_on_order_density(car_to_arrive_order_density, car_to_arrive, current_time)

                        car_to_arrive.velocity = generate_speed(car_to_arrive_order_density, final_max_speed=MAX_CAR_TO)
                        car_to_arrive_travel_time = station_length / car_to_arrive.velocity
                        car_to_arrive_next_arrival_time = current_time + car_to_arrive_travel_time

                        check_for_overtaking(car_to_arrive, car_to_arrive_next_arrival_time)

                        global_event_queue.append(Event(car_to_arrive_next_arrival_time, "Arrival", car_to_arrive))
                        heapq.heapify(global_event_queue)

                        waiting_station_index = (waiting_station_index - 1) % num_stations

                        # 如果回到起点或找到空的等待队列，停止处理
                        if waiting_station_index == up_current_station.index or not stations[waiting_station_index].waiting_queue:
                            break
            else:
                # 下一个站点已满，将车辆放入下一个站点的等待队列
                up_next_station.waiting_queue.append(car)
                print(
                    f"Car {car.number} arrived at waiting queue {up_next_station.index + 1} at simulation time {current_time:.2f}")
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
csv_filename = "car_info4.csv"

# Write to CSV file
with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)

    # Write the header
    header = ["Car Index", "Station Index", "Arrival Time", "Velocity", "Position"]
    writer.writerow(header)

    # Write each car's arrival times
    for car, information in car_info.items():
        car_index = car.number
        for station_index, arrival_time, Velocity, Position in information:
            row = [car_index, station_index, arrival_time, Velocity, Position]
            writer.writerow(row)
