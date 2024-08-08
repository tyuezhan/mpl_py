import numpy as np

class Waypoint:
    def __init__(self, control='NONE', pos=None, vel=None, acc=None, jrk=None, yaw=0, t=0):
        self.pos = np.zeros(3) if pos is None else pos
        self.vel = np.zeros(3) if vel is None else vel
        self.acc = np.zeros(3) if acc is None else acc
        self.jrk = np.zeros(3) if jrk is None else jrk
        self.yaw = yaw
        self.t = t
        self.control = control

    def __eq__(self, other):
        return np.array_equal(self.pos, other.pos) and \
               np.array_equal(self.vel, other.vel) and \
               np.array_equal(self.acc, other.acc) and \
               np.array_equal(self.jrk, other.jrk) and \
               self.yaw == other.yaw and \
               self.t == other.t and \
               self.control == other.control

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((tuple(self.pos), tuple(self.vel), tuple(self.acc), tuple(self.jrk), self.yaw, self.t, self.control))

    def print(self, prefix=""):
        print(f"{prefix}pos: {self.pos}")
        print(f"{prefix}vel: {self.vel}")
        print(f"{prefix}acc: {self.acc}")
        print(f"{prefix}jrk: {self.jrk}")
        print(f"{prefix}yaw: {self.yaw}")
        print(f"{prefix}t: {self.t}")
        print(f"{prefix}control: {self.control}")