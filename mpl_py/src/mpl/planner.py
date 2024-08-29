import numpy as np
from typing import List, Optional
from mpl.state_space import StateSpace
from mpl.env_base import EnvBase
from mpl.env_map_gs import EnvMap
from mpl.primitive import Primitive
from mpl.trajectory import Trajectory
from mpl.graph_search import GraphSearch
from mpl.waypoint import Waypoint
from scipy.spatial.transform import Rotation as R
import torch
import rospy
from visualization_msgs.msg import Marker, MarkerArray
class Planner:
    def __init__(self, eval_traj_func, verbose: bool = False):
        self.ENV: Optional[EnvBase] = None
        self.ss_ptr: Optional[StateSpace] = None
        self.traj: Trajectory = Trajectory()
        self.traj_cost: float = float('inf')
        self.epsilon: float = 1.0
        self.max_num: int = -1
        self.planner_verbose: bool = verbose
        self.map_util = None
        self.eval_traj_func = eval_traj_func
        self.debug = False
        self.prs_pub = rospy.Publisher("/all_candidate_trajs", MarkerArray, queue_size=1)

    def setMapUtil(self, map_util, primitive_dict):
        self.ENV = EnvMap(map_util, primitive_dict)
        print("[MapPlanner] use MPL")
        self.map_util = map_util

    def initialized(self) -> bool:
        return self.ss_ptr is not None
    
    def getTraj(self) -> Trajectory:
        return self.traj
    
    def setWorld2Map(self, world2map):
        '''
        world2map should be 4x4 matrix
        '''
        self.w2m = world2map

    # def getValidPrimitives(self) -> List[Primitive]:
    #     prs = []
    #     if self.ss_ptr:
    #         for it in self.ss_ptr.hm_.values():
    #             if it and it.pred_coord:
    #                 for i, key in enumerate(it.pred_coord):
    #                     if np.isinf(it.pred_action_cost[i]):
    #                         continue
    #                     pr = Primitive(self.ENV.get_dt())
    #                     self.ENV.forward_action(self.ss_ptr.hm_[key].coord, it.pred_action_id[i], pr)
    #                     prs.append(pr)
        
    #     if self.planner_verbose:
    #         print(f"number of states in hm: {len(self.ss_ptr.hm_)}, number of valid prs: {len(prs)}")
        
    #     return prs
    
    def getAllPrimitives(self) -> List[Primitive]:
        prs = []
        if self.ss_ptr:
            for it in self.ss_ptr.hm_.values():
                if it and it.pred_coord:
                    for i, key in enumerate(it.pred_coord):
                        # pr = Primitive(self.ENV.get_dt())
                        pr = self.ENV.forward_action(key, it.pred_action_id[i])
                        prs.append(pr)
        
        if self.planner_verbose:
            print(f"getAllPrimitives number of states in hm: {len(self.ss_ptr.hm_)}, number of prs: {len(prs)}")
        
        return prs
    
    def getOpenSet(self) -> List[np.ndarray]:
        ps = []
        if self.ss_ptr:
            for it in self.ss_ptr.pq_.values():
                ps.append(it.coord.pos)
        return ps
    
    def getCloseSet(self) -> List[np.ndarray]:
        ps = []
        if self.ss_ptr:
            for it in self.ss_ptr.hm_.values():
                if it and it.iterationclosed:
                    ps.append(it.coord.pos)
        return ps
    
    def getNullSet(self) -> List[np.ndarray]:
        ps = []
        if self.ss_ptr:
            for it in self.ss_ptr.hm_.values():
                if it and not it.iterationopened:
                    ps.append(it.coord.pos)
        return ps
    
    def getStates(self, state) -> List[np.ndarray]:
        ps = []
        vels = []
        if self.ss_ptr:
            for it in self.ss_ptr.hm_.values():
                if it:
                    coord = it.coord
                    add = True
                    if hasattr(state, 'use_vel') and (np.linalg.norm(state.vel - coord.vel) > 1e-3):
                        add = False
                    if hasattr(state, 'use_acc') and (np.linalg.norm(state.acc - coord.acc) > 1e-3):
                        add = False
                    if hasattr(state, 'use_jrk') and (np.linalg.norm(state.jrk - coord.jrk) > 1e-3):
                        add = False
                    if add:
                        ps.append(coord.pos)
                        if coord.vel not in vels:
                            vels.append(coord.vel)
        
        for vel in vels:
            print(f"vel: {vel}")
        print("=========================")
        return ps
    
    def getTraj(self):
        return self.traj

    def getExpandedNodes(self) -> List[np.ndarray]:
        return self.ENV.get_expanded_nodes() if self.ENV else []
    
    def getExpandedEdges(self) -> List[Primitive]:
        return self.ENV.get_expanded_edges() if self.ENV else []
    
    def getExpandedNum(self) -> int:
        return self.ss_ptr.expand_iteration_ if self.ss_ptr else 0
    
    def getSubStateSpace(self, time_step: int):
        if self.ss_ptr:
            self.ss_ptr.getSubStateSpace(time_step)
    
    def getTrajCost(self) -> float:
        return self.traj_cost
    
    def checkValidation(self):
        if self.ss_ptr:
            self.ss_ptr.checkValidation(self.ss_ptr.hm_)
    
    def reset(self):
        self.ss_ptr = None
        self.traj = Trajectory(self.traj.dimension)
    
    def setVmax(self, v: float):
        if self.ENV:
            self.ENV.set_v_max(v)
        if self.planner_verbose:
            print(f"[PlannerBase] set v_max: {v}")
    
    def setAmax(self, a: float):
        if self.ENV:
            self.ENV.set_a_max(a)
        if self.planner_verbose:
            print(f"[PlannerBase] set a_max: {a}")
    
    def setJmax(self, j: float):
        if self.ENV:
            self.ENV.set_j_max(j)
        if self.planner_verbose:
            print(f"[PlannerBase] set j_max: {j}")
    
    def setYawmax(self, yaw: float):
        if self.ENV:
            self.ENV.set_yaw_max(yaw)
        if self.planner_verbose:
            print(f"[PlannerBase] set yaw_max: {yaw}")
    
    def setTmax(self, t: float):
        if self.ENV:
            self.ENV.set_t_max(t)
        if self.planner_verbose:
            print(f"[PlannerBase] set max time: {t}")
    
    def setPlanTmax(self, t: float):
        if self.ENV:
            self.ENV.set_plan_t_max(t)
        if self.planner_verbose:
            print(f"[PlannerBase] set plan max time: {t}")
    
    def setDt(self, dt: float):
        if self.ENV:
            self.ENV.set_dt(dt)
        if self.planner_verbose:
            print(f"[PlannerBase] set dt: {dt}")
    
    def setW(self, w: float):
        if self.ENV:
            self.ENV.set_w(w)
        if self.planner_verbose:
            print(f"[PlannerBase] set w: {w}")
    
    def setWyaw(self, w: float):
        if self.ENV:
            self.ENV.set_wyaw(w)
        if self.planner_verbose:
            print(f"[PlannerBase] set wyaw: {w}")
    
    def setEpsilon(self, eps: float):
        self.epsilon = eps
        if self.planner_verbose:
            print(f"[PlannerBase] set epsilon: {eps}")
    
    def setHeurIgnoreDynamics(self, ignore: bool):
        if self.ENV:
            self.ENV.set_heur_ignore_dynamics(ignore)
        if self.planner_verbose:
            print(f"[PlannerBase] set heur_ignore_dynamics: {ignore}")
    
    def setMaxNum(self, num: int):
        self.max_num = num
        if self.planner_verbose:
            print(f"[PlannerBase] set max num: {num}")
    
    def setU(self, U: List[np.ndarray]):
        if self.ENV:
            self.ENV.set_u(U)
    
    def setPriorTrajectory(self, traj: Trajectory):
        if self.ENV:
            self.ENV.set_prior_trajectory(traj)
        if self.planner_verbose:
            print("[PlannerBase] set prior trajectory")
    
    def setTol(self, tol_pos: float, tol_vel: float = -1, tol_acc: float = -1):
        if self.ENV:
            self.ENV.set_tol_pos(tol_pos)
            self.ENV.set_tol_vel(tol_vel)
            self.ENV.set_tol_acc(tol_acc)
        if self.planner_verbose:
            print(f"[PlannerBase] set tol_pos: {tol_pos}")
            print(f"[PlannerBase] set tol_vel: {tol_vel}")
            print(f"[PlannerBase] set tol_acc: {tol_acc}")
    
    def setTolYaw(self, tol_yaw: float):
        if self.ENV:
            self.ENV.set_tol_yaw(tol_yaw)
        if self.planner_verbose:
            print(f"[PlannerBase] set tol_yaw: {tol_yaw}")
    

    def traj_to_wps_in_map(self, cost_lst, prs_lst):
        wps_map = []
        traj = []
        cost = []
        marker_array = MarkerArray()
        for i in range(len(cost_lst)):
            if cost_lst[i] != float('inf'):
                # transform the waypoints to the map frame
                prs = prs_lst[i]
                wps = np.zeros((len(prs), 7))
                for j in range(len(prs)):
                    pose = prs[j].p(1)
                    r_world = R.from_euler('z', pose[3], degrees=False).as_matrix()
                    t_world_homo = np.array([pose[0], pose[1], pose[2], 1])
                    t_map_homo = np.dot(self.w2m, t_world_homo)
                    r_map = np.dot(self.w2m[:3, :3], r_world)
                    quat_map = R.from_matrix(r_map).as_quat()
                    wps[j][:3] = t_map_homo[:3]
                    wps[j][3:] = quat_map

                    if self.debug:
                        # compose the marker array
                        marker = Marker()
                        marker.header.frame_id = "gs_map"
                        marker.header.stamp = rospy.Time.now()
                        marker.ns = "waypoints"
                        marker.id = i*len(prs) + j
                        marker.type = Marker.ARROW
                        marker.action = Marker.ADD
                        marker.pose.position.x = t_map_homo[0]
                        marker.pose.position.y = t_map_homo[1]
                        marker.pose.position.z = t_map_homo[2]
                        marker.pose.orientation.x = quat_map[0]
                        marker.pose.orientation.y = quat_map[1]
                        marker.pose.orientation.z = quat_map[2]
                        marker.pose.orientation.w = quat_map[3]
                        marker.scale.x = 0.05
                        marker.scale.y = 0.05
                        marker.scale.z = 0.05
                        marker.color.a = 1.0
                        marker.color.r = 0.2 * i
                        marker.color.g = 0.2 * i
                        marker.color.b = 0.2 * i
                        marker_array.markers.append(marker)
                        
                wps = torch.tensor(wps)
                wps_map.append(wps)
                traj.append(prs)
                cost.append(cost_lst[i])
        if self.debug: 
            self.prs_pub.publish(marker_array)
        return wps_map, cost, traj
        
    
    def plan(self, start: Waypoint, goal: Waypoint, params, intrinsics) -> bool:
        if self.planner_verbose:
            start.print("Start:")
            goal.print("Goal:")
            self.ENV.info()
        
        if not self.ENV.is_free(start.pos):
            print("[PlannerBase] start is not free!")
            return False
        
        planner_ptr = GraphSearch(self.planner_verbose)
        
        # If use A*, reset the state space 
        self.ss_ptr = StateSpace(self.epsilon)

        self.ENV.set_goal(goal)
        if self.ENV:
            self.ENV.get_expanded_nodes().clear()
            self.ENV.get_expanded_edges().clear()
        
        if self.ss_ptr:
            self.ss_ptr.dt_ = self.ENV.get_dt()
            # self.traj_cost, prs = planner_ptr.Astar(start, self.ENV, self.ss_ptr, self.traj, self.max_num)
            traj_cost_lst, prs_lst = planner_ptr.Astar_best_k(start, self.ENV, self.ss_ptr, self.traj, self.max_num)
            # First, transform all waypoints to the map frame
            s_time = rospy.Time.now()
            print("trajectory cost list: ", traj_cost_lst)
            if len(traj_cost_lst) == 1 and traj_cost_lst[0] == 0:
                return False
            wps, costs, trajs = self.traj_to_wps_in_map(traj_cost_lst, prs_lst)
            if len(wps) == 0:
                return False
            # Call gaussian gradient cost here
            utils = self.eval_traj_func(wps, params, intrinsics)
            utils = [util.detach().cpu().numpy() for util in utils]
            best_idx = np.argmin(utils)
            best_traj = trajs[best_idx]           
            rospy.loginfo("Eval traj util time: {}".format((rospy.Time.now() - s_time).to_sec()))
            self.traj.init_trajectory(best_traj)
            self.traj_cost = costs[best_idx]
        if np.isinf(self.traj_cost):
            if self.planner_verbose:
                print("[PlannerBase] Cannot find a trajectory!")
            return False
        
        return True
