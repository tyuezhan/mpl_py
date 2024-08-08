import numpy as np
from mpl.waypoint import Waypoint

class Primitive:
    def __init__(self, *args) -> None:
        '''
        
        p: Initial state vector (x, y, z, yaw)
        u_v: Linear velocity
        u_w: Angular velocity
        '''

        if len(args) == 1:
            #t only
            p = None
            u_v = 0
            u_w = 0
            self.t = args[0]
        elif len(args) == 3:
            p = args[0]
            u_v = args[1][0]
            u_w = args[1][1]
            self.t = args[2]
        else:
            raise ValueError("Invalid number of arguments")
        if p is None:
            self.p_ = np.zeros(4)
        else:
            self.p_ = np.array(p, dtype=float)
        self.u_v_ = u_v
        self.u_w_ = u_w

    def J(self) -> float:
        '''
        Control effort
        '''
        return self.u_v_ * self.u_v_ * self.t + self.u_w_ * self.u_w_ * self.t

    def p(self, t: float) -> np.ndarray:
        '''
        Returns the state vector [x, y, z, yaw] at time t
        '''
        p_curr = np.zeros(4)
        if self.u_w_ == 0:
            p_curr[0] = self.p_[0] + self.u_v_ * t * np.cos(self.p_[3])
            p_curr[1] = self.p_[1] + self.u_v_ * t * np.sin(self.p_[3])
            p_curr[2] = self.p_[2]
            p_curr[3] = self.p_[3]
        else:
            p_curr[0] = self.p_[0] + self.u_v_ / self.u_w_ * (np.sin(self.p_[3] + self.u_w_ * t) - np.sin(self.p_[3]))
            p_curr[1] = self.p_[1] - self.u_v_ / self.u_w_ * (np.cos(self.p_[3] + self.u_w_ * t) - np.cos(self.p_[3]))
            p_curr[2] = self.p_[2]
            p_curr[3] = self.p_[3] + self.u_w_ * t
        return p_curr

    def v(self, t: float) -> np.ndarray:
        '''
        Returns the velocity vector [vx, vy, vz, wz] at time t
        '''
        theta = self.p_[3] + self.u_w_ * t
        v = np.zeros(4)
        v[0] = self.u_v_ * np.cos(theta)
        v[1] = self.u_v_ * np.sin(theta)
        v[2] = 0
        v[3] = self.u_w_
        return v

    def max_vel(self) -> float:
        return self.u_v_

    def u(self) -> np.ndarray:
        '''
        Returns the control vector [u_v, u_w]
        '''
        return np.array([self.u_v_, self.u_w_], dtype=float)

    def p_zero(self) -> np.ndarray:
        '''
        Returns the initial state vector [x, y, z, yaw]
        '''
        return self.p_

    def evaluate(self, t: float) -> Waypoint:
        '''
        Returns the waypoint at time t
        '''
        wp = Waypoint()
        p_curr = self.p(t)
        wp.pos = p_curr[:3]
        wp.yaw = p_curr[3]
        return wp



if __name__ == "__main__":
    # Example
    p = [0, 0, 0, 0]  # Initial state vector (x, y, z, yaw)
    u_v = 1.0         # Linear velocity
    u_w = 0.5         # Angular velocity

    car = Primitive(p, u_v, u_w)

    t = 1.0  # Time
    print("J(t):", car.J(t))
    print("p(t):", car.p(t))
    print("v(t):", car.v(t))
    print("max_vel:", car.max_vel())
    print("coeff:", car.u())
    print("p_zero:", car.p_zero())
