import numpy as np
from mpl.primitive import Primitive
from mpl.env_base import EnvBase

# calls the collison check from map_utils

class EnvMap(EnvBase):
    def __init__(self, map_util):
        super().__init__()
        self.map_util = map_util
        self.gradient_map = []
        self.potential_weight = 0.1
        self.gradient_weight = 0.0

    def is_goal(self, state):
        goaled = np.linalg.norm(state.pos - self.goal_node.pos) <= self.tol_pos

        if goaled and self.tol_vel >= 0:
            goaled = np.linalg.norm(state.vel - self.goal_node.vel) <= self.tol_vel
        if goaled and self.tol_acc >= 0:
            goaled = np.linalg.norm(state.acc - self.goal_node.acc) <= self.tol_acc
        if goaled and self.tol_yaw >= 0:
            goaled = abs(state.yaw - self.goal_node.yaw) <= self.tol_yaw
        
        return goaled

    def dist_to_goal(self, state):
        return np.linalg.norm(state.pos - self.goal_node.pos)

    # TODO: change this to use the gaussian collision check
    def is_free(self, pt):
        pn = self.map_util.float_to_int(pt)
        return self.map_util.is_free(pn)

    def traverse_primitive(self, primitive):
        max_v = max(primitive.max_vel)
        n = 2 * max(5, int(np.ceil(max_v * primitive.t / self.map_util.get_res())))
        c = 0.0

        dt = primitive.t / n
        for t in np.arange(0, primitive.t, dt):
            # TODO: precompute the primitive will make this faster
            pt = primitive.evaluate(t)
            # pn = self.map_util.float_to_int(pt.pos)
            # idx = self.map_util.get_index(pn)

            # if self.map_util.is_outside(pn) or (self.search_region and not self.search_region[idx]):
                # return float('inf')
            if self.map_util.is_outside(pt.pos):
                return float('inf')
            elif self.map_util.is_occupied(pt.pos):
                return float('inf')

            # if self.wyaw > 0 and pt.use_yaw:
            #     v = pt.vel[:2]
            #     if np.linalg.norm(v) > 1e-5:
            #         v_value = 1 - np.dot(v / np.linalg.norm(v), [np.cos(pt.yaw), np.sin(pt.yaw)])
            #         c += self.wyaw * v_value * dt

        # if primitive.control == 'CAR':
        #     p0 = primitive.evaluate(0)
        #     pt = primitive.evaluate(primitive.t)
        #     pos1 = np.array([p0.pos[0], p0.pos[1], 0])
        #     pos2 = np.array([pt.pos[0], pt.pos[1], 0])
        #     c += self.w_view * self.get_view_correlation(pos1, p0.yaw, pos2, pt.yaw)
        
        return c

    # TODO: vectorize this
    def get_succ(self, curr, succ, succ_cost, action_idx):
        succ.clear()
        succ_cost.clear()
        action_idx.clear()

        self.expanded_nodes.append(curr.pos)
        for i, u in enumerate(self.U):
            primitive = Primitive(curr, u, self.dt)
            # TODO: we can make this faster by pre-compute the primitive
            tn = primitive.evaluate(self.dt)

            if tn == curr:
                continue
            tn.t = curr.t + self.dt
            succ.append(tn)
            cost = 0 if curr.pos == tn.pos else self.traverse_primitive(primitive)
            if not np.isinf(cost):
                cost += self.calculate_intrinsic_cost(primitive)
                self.expanded_edges.append(primitive)
            succ_cost.append(cost)
            action_idx.append(i)

    def set_gradient_map(self, map_):
        self.gradient_map = map_

    def set_gradient_weight(self, weight):
        self.gradient_weight = weight

    def set_potential_map(self, map_):
        self.potential_map = map_

    def set_potential_weight(self, weight):
        self.potential_weight = weight

    def set_prior_trajectory(self, traj):
        self.prior_traj.clear()
        total_time = traj.get_total_time()
        n = int(np.ceil(self.v_max * total_time / self.map_util.get_res()))
        pts = traj.sample(n)

        costs = []
        for t in np.arange(0, total_time, self.dt):
            potential_cost = 0
            if self.potential_map:
                prev_idx = -1
                for pt in pts:
                    if pt.t >= t:
                        break
                    pn = self.map_util.float_to_int(pt.pos)
                    idx = self.map_util.get_index(pn)
                    if prev_idx == idx:
                        continue
                    prev_idx = idx
                    potential_cost += self.potential_weight * self.potential_map[idx] + \
                                      self.gradient_weight * np.linalg.norm(pt.vel)
            costs.append(self.w * t + potential_cost)
            print(f"t: {t:.2f}, cost: {costs[-1]:.2f}")

        total_cost = self.traverse_trajectory(traj) + self.w * total_time
        print(f"total cost: {total_cost:.2f}")

        for t in np.arange(0, total_time, self.dt):
            id_ = int(t / self.dt)
            self.prior_traj.append((traj.evaluate(t), total_cost - costs[id_]))

        self.goal_node = traj.evaluate(total_time)

    def traverse_trajectory(self, traj):
        total_time = traj.get_total_time()
        n = int(np.ceil(self.v_max * total_time / self.map_util.get_res()))
        c = 0.0
        pts = traj.sample(n)
        prev_idx = -1
        for pt in pts:
            pn = self.map_util.float_to_int(pt.pos)
            idx = self.map_util.get_index(pn)
            if prev_idx == idx:
                continue
            prev_idx = idx
            if self.map_util.is_outside(pn):
                return float('inf')
            if self.potential_map:
                if 0 < self.potential_map[idx] < 100:
                    c += self.potential_weight * self.potential_map[idx] + \
                         self.gradient_weight * np.linalg.norm(pt.vel)
                elif self.potential_map[idx] >= 100:
                    return float('inf')
            elif self.map_util.is_occupied(pn):
                return float('inf')
        return c

    def info(self):
        print("++++++++++++++++++++ EnvMap ++++++++++++++++++")
        print(f"+                  w: {self.w:.2f}               +")
        print(f"+               wyaw: {self.wyaw:.2f}               +")
        print(f"+                 dt: {self.dt:.2f}               +")
        print(f"+              t_max: {self.t_max:.2f}               +")
        print(f"+              v_max: {self.v_max:.2f}               +")
        print(f"+              a_max: {self.a_max:.2f}               +")
        print(f"+              j_max: {self.j_max:.2f}               +")
        print(f"+            yaw_max: {self.yaw_max:.2f}               +")
        print(f"+              U num: {len(self.U)}                +")
        print(f"+            tol_pos: {self.tol_pos:.2f}               +")
        print(f"+            tol_vel: {self.tol_vel:.2f}               +")
        print(f"+            tol_acc: {self.tol_acc:.2f}               +")
        print(f"+            tol_yaw: {self.tol_yaw:.2f}               +")
        print(f"+         plan_t_max: {self.plan_t_max:.2f}          +")
        print(f"+heur_ignore_dynamics: {self.heur_ignore_dynamics}                 +")
        if self.potential_map:
            print(f"+    potential_weight: {self.potential_weight:.2f}                 +")
        if self.gradient_map:
            print(f"+     gradient_weight: {self.gradient_weight:.2f}                 +")
        if self.prior_traj:
            print("+     use_prior_traj: true                 +")
        print("++++++++++++++++++++ EnvMap ++++++++++++++++++")

    def get_view_correlation(self, pos1, yaw1, pos2, yaw2):
        # Placeholder implementation
        return 0.0