import numpy as np
from mpl.primitive import Primitive
from scipy.spatial.transform import Rotation as R


class PrimitiveDict:
    def __init__(self, n_t, control):
        self.primitives = {}
        self.dts = np.linspace(0, 1.0, n_t).tolist()
        self.control = control
    
    def precompute(self):
        # Loop through every control input
        # Loop through every t in n_dt
        # compute states with (0, 0, 0, yaw=0) as initial state
        # Save all the states in a dictionary with key as control input and t
        for u in self.control:
            self.primitives[u] = {}
            for t in self.dts:
                pr = Primitive([0, 0, 0, 0], u, 1.0)
                pose = pr.p(t) # Compute the state at time t
                self.primitives[u][t] = pose

    def get_nt(self):
        return self.n_t

    def get_element(self, u):
        return self.primitives[u]


if __name__ == "__main__":
    # Let's create a primitive with state (0, 0, 0, yaw=0) and control input (u_v=1.0, u_w=0.5)
    p = [0, 0, 0, 0]  # Initial state vector (x, y, z, yaw)
    u_v = 1.0         # Linear velocity
    u_w = 0.5         # Angular velocity
    dt = 1.0          # Time step
    pr = Primitive(p, [u_v, u_w], 1.0)
    print("PR0: P(0): ", pr.p_zero())
    print("PR0: P(1): ", pr.p(1))
    print("PR0: V(1): ", pr.v(1))
    
    # Two cases, if put the robot at (1, 2, 0, yaw=1.57) and control input (u_v=1.0, u_w=0.5)
    # Compute the end state at time t=1.0
    pr1 = Primitive([1, 2, 0, 1.57], [u_v, u_w], 1.0)
    print("PR1: P(0): ", pr1.p_zero())
    print("PR1: P(1): ", pr1.p(1))

    # Just add the state from (0, 0, 0, yaw=0) at t=1 to the state (1, 2, 0, yaw=1.57)
    pr2 = Primitive([1, 2, 0, 1.57], [u_v, u_w], 1.0)
    print("PR2: P(0): ", pr2.p_zero())
    # Rotate pr.p(1) by yaw=1.57
    r = R.from_euler('z', 1.57, degrees=False)
    p1 = r.apply(pr.p(1)[:3])
    p1_new = [p1[0], p1[1], 0, pr.p(1)[3]]
    print("PR2: P(1) by adding P(0) with PR0", pr2.p_zero() + p1_new)
    

    # Starting 
