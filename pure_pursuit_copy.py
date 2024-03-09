"""

Path tracking simulation with pure pursuit steering and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)
        Guillaume Jacquenot (@Gjacquenot)

"""
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from scipy import interpolate

PIXEL_SPEED = 57.846 # [pixel/s]
PIXEL_TO_CM = 0.4322 

LFC_OFFSET = .55
YAW_OFFSET = 90

# Parameters
k = 0.1  # look forward gain
Lfc = 42.75 * LFC_OFFSET  # [pixel] look-ahead distance
Kp = 1.0  # speed proportional gain
dt = 0.1  # [s] time tick
WB = 42.75  # [pixel] wheel base of vehicle

show_animation = True


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def update(self, a, delta):
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)


class States:

    def __init__(self):
        self.x = []
        self.y = []
        self.yaw = []
        self.v = []
        self.t = []

    def append(self, t, state):
        self.x.append(state.x)
        self.y.append(state.y)
        self.yaw.append(state.yaw)
        self.v.append(state.v)
        self.t.append(t)


def proportional_control(target, current):
    a = Kp * (target - current)

    return a


class TargetCourse:

    def __init__(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.old_nearest_point_index = None

    def search_target_index(self, state):

        # To speed up nearest point search, doing it at only first time.
        if self.old_nearest_point_index is None:
            # search nearest point index
            dx = [state.rear_x - icx for icx in self.cx]
            dy = [state.rear_y - icy for icy in self.cy]
            d = np.hypot(dx, dy)
            ind = np.argmin(d)
            self.old_nearest_point_index = ind
        else:
            ind = self.old_nearest_point_index
            distance_this_index = state.calc_distance(self.cx[ind],
                                                      self.cy[ind])
            while True:
                distance_next_index = state.calc_distance(self.cx[ind + 1],
                                                          self.cy[ind + 1])
                if distance_this_index < distance_next_index:
                    break
                ind = ind + 1 if (ind + 1) < len(self.cx) else ind
                distance_this_index = distance_next_index
            self.old_nearest_point_index = ind

        Lf = k * state.v + Lfc  # update look ahead distance

        # search look ahead target point index
        while Lf > state.calc_distance(self.cx[ind], self.cy[ind]):
            if (ind + 1) >= len(self.cx):
                break  # not exceed goal
            ind += 1

        return ind, Lf


def pure_pursuit_steer_control(state, trajectory, pind):
    ind, Lf = trajectory.search_target_index(state)

    if pind >= ind:
        ind = pind

    if ind < len(trajectory.cx):
        tx = trajectory.cx[ind]
        ty = trajectory.cy[ind]
    else:  # toward goal
        tx = trajectory.cx[-1]
        ty = trajectory.cy[-1]
        ind = len(trajectory.cx) - 1

    alpha = math.atan2(ty - state.rear_y, tx - state.rear_x) - state.yaw

    delta = math.atan2(2.0 * WB * math.sin(alpha) / Lf, 1.0)

    return delta, ind


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for ix, iy, iyaw in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

class Path:
    def __init__(self):
        self.pathType = ''
        self.interpType = ''

    def updatePathType(self, pathType, interpType):
        self.pathType = pathType
        self.interpType = interpType

    def generatePath(self):
        if self.pathType == 'yellowRoundAbout':
            cx = [143, 157, 167, 183, 197, 210, 220, 230, 244, 256, 269, 284, 
                  306, 327, 346, 360, 367, 373, 378, 372, 361, 355, 344, 336]
            cy = [350, 350, 350, 351, 351, 352, 354, 360, 368, 375, 387, 400, 
                  402, 400, 392, 380, 363, 345, 325, 306, 287, 270, 260, 260]
            
        elif self.pathType == 'purpleRoundAbout':
            cx = [464, 432, 402, 373, 339, 308, 284, 259, 239, 220]
            cy = [305, 303, 303, 287, 259, 252, 257, 280, 290, 300]

        elif self.pathType == 'orangeLeft':
            cx = [336, 334, 330, 331, 320, 307, 295, 280, 260]
            cy = [204, 180, 160, 137, 110, 90,  72,  50,  36]

        elif self.pathType == 'yellowLeft':
            cx = [25, 23, 24, 26, 27, 34, 45, 55, 75, 100]
            cy = [176, 200, 225, 250, 280, 305, 330, 342, 350, 350]

        elif self.pathType =='bridge':
            cx = [122, 143, 162, 180, 195, 217, 236, 250]
            cy = [583, 584, 584, 584, 584, 577, 570, 558]
        return cx, cy

def main():
    #  target course
    # cx = np.arange(100, 500, 50)
    # cy = [100, 100, 125, 150, 125, 110, 100, 100] 
    path = Path()
    path.updatePathType('yellowRoundAbout', 'slinear')
    cxx, cyy = path.generatePath()
    step = PIXEL_SPEED / 10
    yawDir = 1
    if cxx[0] > cxx[-1]:
        step = -PIXEL_SPEED / 10
        yawDir = -1
    f = interpolate.interp1d(cxx, cyy, kind=path.interpType)
    cx = np.arange(cxx[0], cxx[-1], step)
    cy = -1 * f(cx)

    print(cx)
    print(cy)
    
    # cy = [math.sin(ix / 5.0) * ix / 2.0 for ix in cx]
    # Đường tròn (x - a)^2 + (y - b)^2 = r^2
    # tam_x = -45
    # tam_y = 0
    # cy = [(np.sqrt(45**2 - (ix - tam_x)**2) + tam_y) for ix in cx]

    target_speed = PIXEL_SPEED  # [pixel/s]

    T = 200.0  # max simulation time

    # initial state
    state = State(x=cxx[0], y=-cyy[0], yaw=0, v=PIXEL_SPEED)
    pre_yaw = state.yaw
    ros_publish_yaw = 0

    lastIndex = len(cx) - 1
    timee = 0.0
    states = States()
    states.append(timee, state)
    target_course = TargetCourse(cx, cy)
    target_ind, _ = target_course.search_target_index(state)

    while T >= timee and lastIndex > target_ind:

        # Calc control input
        ai = proportional_control(target_speed, state.v)
        di, target_ind = pure_pursuit_steer_control(
            state, target_course, target_ind)

        state.update(ai, di)  # Control vehicle
        ros_publish_yaw += (state.yaw - pre_yaw) * 180 / math.pi
        pre_yaw = state.yaw
        if (ros_publish_yaw > 25.0):
            ros_publish_yaw = 25.0
        elif(ros_publish_yaw < -25.0):
            ros_publish_yaw = -25.0
        print(yawDir * ros_publish_yaw)

        time.sleep(0.1)
        timee += dt
        states.append(timee, state)

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plot_arrow(state.x, state.y, state.yaw)
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(states.x, states.y, "-b", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.axis("equal")
            plt.grid(True)
            plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
            plt.pause(0.001)

    # Test
    assert lastIndex >= target_ind, "Cannot goal"

    if show_animation:  # pragma: no cover
        plt.cla()
        plt.plot(cx, cy, ".r", label="course")
        plt.plot(states.x, states.y, "-b", label="trajectory")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.axis("equal")
        plt.grid(True)

        plt.subplots(1)
        plt.plot(states.t, [iv * 3.6 for iv in states.v], "-r")
        plt.xlabel("Time[s]")
        plt.ylabel("Speed[km/h]")
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    print("Pure pursuit path tracking simulation start")
    main()
