import numpy as np
from typing import List, Union
from waypoint import Waypoint

class Command:
    def __init__(self, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, jrk: np.ndarray,
                 yaw: float, yaw_dot: float, t: float):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.jrk = jrk
        self.yaw = yaw
        self.yaw_dot = yaw_dot
        self.t = t

class Primitive:
    def __init__(self, t: float, control: int):
        self.t = t
        self.control = control
    
    def t(self) -> float:
        return self.t
    
    def control(self) -> int:
        return self.control
    
    def pr(self, index: int):
        # Assuming pr returns a callable object for position, velocity, acceleration, and jerk
        return self
    
    def evaluate(self, tau: float) -> Waypoint:
        # Dummy implementation
        return Waypoint(self.control)

    def max_vel(self, index: int) -> float:
        # Dummy implementation
        return 1.0
    
    def extrema_vel(self, t: float) -> List[float]:
        # Dummy implementation
        return [0.0]
    
    def J(self, control: int) -> float:
        # Dummy implementation
        return 1.0
    
    def Jyaw(self) -> float:
        # Dummy implementation
        return 1.0

class Lambda:
    def __init__(self, vs: List[dict]):
        # Dummy initialization
        pass
    
    def getTau(self, time: float) -> float:
        # Dummy implementation
        return time
    
    def getT(self, tau: float) -> float:
        # Dummy implementation
        return tau
    
    def evaluate(self, tau: float) -> 'VirtualPoint':
        # Dummy implementation
        return VirtualPoint(1.0, 0.0, tau)
    
    def exist(self) -> bool:
        return True

class VirtualPoint:
    def __init__(self, p: float, v: float, t: float):
        self.p = p
        self.v = v
        self.t = t

class Trajectory:
    def __init__(self, primitives: List[Primitive]):
        self.segs = primitives
        self.taus = [0]
        for pr in primitives:
            self.taus.append(pr.t() + self.taus[-1])
        self.Ts = self.taus
        self.total_t_ = self.taus[-1]
        self.lambda_ = Lambda([])  # Initialize with dummy values
    
    def evaluate(self, time: float) -> Waypoint:
        tau = self.lambda_.getTau(time)
        if tau < 0: tau = 0
        if tau > self.total_t_: tau = self.total_t_

        for id, seg in enumerate(self.segs):
            if (tau >= self.taus[id] and tau < self.taus[id + 1]) or id == len(self.segs) - 1:
                tau -= self.taus[id]
                if seg.control() == 1:  # Assuming Control.CAR = 1
                    return seg.evaluate(tau)
                p = Waypoint(seg.control())
                for j in range(len(p.pos)):
                    pr = seg.pr(j)
                    p.pos[j] = pr.p(tau)
                    p.vel[j] = pr.v(tau)
                    p.acc[j] = pr.a(tau)
                    p.jrk[j] = pr.j(tau)
                p.yaw = np.arctan2(np.sin(seg.pr(0).p(tau)), np.cos(seg.pr(0).p(tau)))
                return p
        
        print(f"Cannot find tau according to time: {time}")
        return Waypoint(0)
    
    def evaluate_command(self, time: float, p: Command) -> bool:
        tau = self.lambda_.getTau(time)
        if tau < 0: tau = 0
        if tau > self.total_t_: tau = self.total_t_

        lambda_val = 1
        lambda_dot = 0

        if self.lambda_.exist():
            vt = self.lambda_.evaluate(tau)
            lambda_val = vt.p
            lambda_dot = vt.v

        for id, seg in enumerate(self.segs):
            if tau >= self.taus[id] and tau <= self.taus[id + 1]:
                tau -= self.taus[id]
                if seg.control() == 1:  # Assuming Control.CAR = 1
                    wp = seg.evaluate(tau)
                    p.pos = wp.pos
                    p.yaw = wp.yaw
                    p.vel = wp.vel
                    p.yaw_dot = wp.yaw_dot
                    p.t = time
                    return True
                
                for j in range(len(p.pos)):
                    pr = seg.pr(j)
                    p.pos[j] = pr.p(tau)
                    p.vel[j] = pr.v(tau) / lambda_val
                    p.acc[j] = (pr.a(tau) / lambda_val / lambda_val) - (p.vel[j] * lambda_dot / (lambda_val**3))
                    p.jrk[j] = (pr.j(tau) / lambda_val**2) - (3 / lambda_val**3 * p.acc[j]**2 * lambda_dot) + (3 / lambda_val**4 * p.vel[j] * lambda_dot**2)
                    p.yaw = np.arctan2(np.sin(seg.pr_yaw().p(tau)), np.cos(seg.pr_yaw().p(tau)))
                    p.yaw_dot = seg.pr_yaw().v(tau)
                    p.t = time
                return True
        
        print(f"Cannot find tau according to time: {time}")
        return False
    
    def scale(self, ri: float, rf: float) -> bool:
        vs = [VirtualPoint(1.0 / ri, 0, 0), VirtualPoint(1.0 / rf, 0, self.taus[-1])]
        self.lambda_ = Lambda(vs)
        self.Ts = [self.lambda_.getT(tau) for tau in self.taus]
        self.total_t_ = self.Ts[-1]
        return True
    
    def scale_down(self, mv: float, ri: float, rf: float) -> bool:
        vs = [VirtualPoint(ri, 0, 0), VirtualPoint(rf, 0, self.taus[-1])]

        for id, seg in enumerate(self.segs):
            for i in range(3):  # Assuming 3D
                if seg.max_vel(i) > mv:
                    ts = seg.extrema_vel(seg.t())
                    if id != 0:
                        ts.append(0)
                    ts.append(seg.t())
                    for tv in ts:
                        v = seg.pr(i).evaluate(tv)[1]  # assuming second element is velocity
                        lambda_v = abs(v) / mv
                        if lambda_v <= 1:
                            continue
                        vt = VirtualPoint(lambda_v, 0, tv + self.taus[id])
                        vs.append(vt)

        vs.append(vs[-1])

        vs.sort(key=lambda v: v.t)
        max_l = max(v.p for v in vs)

        if max_l <= 1:
            return False

        for i in range(1, len(vs) - 1):
            vs[i].p = max_l

        vs_s = [vs[0]] + [v for v in vs[1:] if v.t > vs_s[-1].t]

        self.lambda_ = Lambda(vs_s)
        self.Ts = [self.lambda_.getT(tau) for tau in self.taus]
        self.total_t_ = self.Ts[-1]
        return True
    
    def sample(self, N: int) -> List[Command]:
        ps = []
        dt = self.total_t_ / N
        for i in range(N + 1):
            cmd = Command(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3), 0.0, 0.0, 0.0)
            self.evaluate_command(i * dt, cmd)
            ps.append(cmd)
        return ps

    def J(self, control: int) -> float:
        return sum(seg.J(control) for seg in self.segs)

    def Jyaw(self) -> float:
        return sum(seg.Jyaw() for seg in self.segs)

    def get_segment_times(self) -> List[float]:
        return [self.Ts[i + 1] - self.Ts[i] for i in range(len(self.Ts) - 1)]

    def get_waypoints(self) -> List[Waypoint]:
        waypoints = []
        if not self.segs:
            return waypoints
        
        t = 0
        for seg in self.segs:
            wp = seg.evaluate(0)
            wp.t = t
            waypoints.append(wp)
            t += seg.t()
        wp = self.segs[-1].evaluate(self.segs[-1].t())
        wp.t = t
        waypoints.append(wp)
        return waypoints

    def get_primitives(self) -> List[Primitive]:
        return self.segs

    def lambda_(self) -> Lambda:
        return self.lambda_

    def get_total_time(self) -> float:
        return self.total_t_
