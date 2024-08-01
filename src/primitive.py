import numpy as np

class PrimitiveCar:
    def __init__(self, p: list | np.ndarray=None, u_v: float=0.0, u_w: float=0.0) -> None:
        '''
        p: Initial state vector (x, y, z, yaw)
        u_v: Linear velocity
        u_w: Angular velocity
        '''
        if p is None:
            self.p_ = np.zeros(4)
        else:
            self.p_ = np.array(p, dtype=float)
        self.u_v_ = u_v
        self.u_w_ = u_w

    def J(self, t: float) -> float:
        '''
        Control effort
        '''
        return self.u_v_ * self.u_v_ * t + self.u_w_ * self.u_w_ * t

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



if __name__ == "__main__":
    # Example
    p = [0, 0, 0, 0]  # Initial state vector (x, y, z, yaw)
    u_v = 1.0         # Linear velocity
    u_w = 0.5         # Angular velocity

    car = PrimitiveCar(p, u_v, u_w)

    t = 1.0  # Time
    print("J(t):", car.J(t))
    print("p(t):", car.p(t))
    print("v(t):", car.v(t))
    print("max_vel:", car.max_vel())
    print("coeff:", car.u())
    print("p_zero:", car.p_zero())
