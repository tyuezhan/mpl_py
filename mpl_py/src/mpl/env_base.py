import numpy as np
import time
import rospy

class EnvBase:
    def __init__(self):
        self.dt = 1.0
        self.tol_pos = 0.5
        self.tol_vel = -1.0
        self.tol_acc = -1.0
        self.tol_yaw = -1.0
        self.w = 10.0
        self.h_fov = np.pi / 2
        self.w_view = 0.0
        self.v_max = -1.0
        self.a_max = -1.0
        self.j_max = -1.0
        self.yaw_max = -1.0
        self.t_max = float('inf')
        self.plan_t_max = float('inf')
        self.plan_start_time = rospy.Time.now()
        self.U = []
        self.goal_node = None
        self.prior_traj = []
        self.search_region = []
        self.expanded_nodes = []
        self.expanded_edges = []
        self.all_init_yaws = []
        self.all_control_pts = {}
        self.all_view_costs = {}


    def set_dt(self, dt):
        self.dt = dt

    def set_tol_pos(self, pos):
        self.tol_pos = pos

    def set_tol_vel(self, vel):
        self.tol_vel = vel

    def set_tol_acc(self, acc):
        self.tol_acc = acc

    def set_tol_yaw(self, yaw):
        self.tol_yaw = yaw

    def set_w(self, w):
        self.w = w

    def set_wyaw(self, wyaw):
        self.wyaw = wyaw

    def set_max_ray_len(self, len):
        self.max_ray_len = len

    def set_h_fov(self, fov):
        self.h_fov = fov

    def set_w_view(self, w):
        self.w_view = w

    def set_t_max(self, t):
        self.t_max = t

    def set_plan_t_max(self, t):
        self.plan_t_max = t

    def set_plan_start_time(self):
        self.plan_start_time = rospy.Time.now()

    def set_all_init_yaws(self, all_init_yaws):
        self.all_init_yaws = all_init_yaws

    def set_all_control_pts(self, all_control_pts):
        self.all_control_pts = all_control_pts

    def set_all_view_costs(self, all_view_costs):
        self.all_view_costs = all_view_costs

    def plan_timeout(self):
        t_now = rospy.Time.now()
        duration_s = (t_now - self.plan_start_time).to_sec()
        if duration_s >= self.plan_t_max:
            print("elapsed_time:", duration_s)
        return duration_s >= self.plan_t_max

    def set_goal(self, state):
        if not self.prior_traj:
            self.goal_node = state
        return not self.prior_traj

    def set_search_region(self, search_region):
        self.search_region = search_region

    def set_heur_ignore_dynamics(self, ignore):
        self.heur_ignore_dynamics = ignore

    def info(self):
        print("\033[33m")
        print("++++++++++++++++++++ env_base ++++++++++++++++++")
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
        print(f"+heur_ignore_dynamics: {int(self.heur_ignore_dynamics)}                 +")
        print("++++++++++++++++++++ env_base ++++++++++++++++++")
        print("\033[0m")

    def is_free(self, pt):
        print("Used Null is_free() for pt")
        return True

    def calculate_intrinsic_cost(self, pr):
        return pr.J(pr.control()) + self.w * self.dt

    def get_dt(self):
        return self.dt

    def get_succ(self, curr, succ, succ_cost, action_idx):
        print("Used Null get_succ()")

    def get_succ_with_yaw(self, curr, succ, succ_cost, action_idx, yaw_idx):
        print("Used Null get_succ()")

    def set_type(self, use_topp):
        self.use_topp = use_topp

    def get_type(self):
        return self.use_topp

    def get_search_region(self):
        return self.search_region
