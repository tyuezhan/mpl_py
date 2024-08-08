from mpl.state_space import State
# import heapq
import heapdict
import numpy as np
from mpl.primitive import Primitive

class GraphSearch:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def Astar(self, start_coord, ENV, ss, traj, max_expand=-1):
        ENV.set_plan_start_time()

        if ENV.is_goal(start_coord):
            return 0, []

        if start_coord not in ss.hm_:
            curr_node = State(start_coord)
            curr_node.g = 0
            if ss.eps_ == 0:
                curr_node.h = 0
            else:
                curr_node.h = ENV.get_heur(start_coord)
            fval = curr_node.g + ss.eps_ * curr_node.h
            ss.pq_[curr_node] = fval
            # curr_node.heapkey = (fval, curr_node)
            # heapq.heappush(ss.pq_, curr_node.heapkey)
            curr_node.iterationopened = True
            ss.hm_[start_coord] = curr_node
        else:
            curr_node = ss.hm_[start_coord]

        expand_iteration = 0
        best_dist = float('inf')
        best_node = curr_node

        while ss.pq_:
            expand_iteration += 1
            # curr_node = heapq.heappop(ss.pq_)[1]
            curr_node = ss.pq_.popitem()[0]
            curr_node.iterationclosed = True

            dist_to_goal = ENV.dist_to_goal(curr_node.coord)
            if dist_to_goal < best_dist:
                best_dist = dist_to_goal
                best_node = curr_node

            succ_coord, succ_cost, succ_act_id = [], [], []
            ENV.get_succ(curr_node.coord, succ_coord, succ_cost, succ_act_id)

            for s, succ in enumerate(succ_coord):
                if np.isinf(succ_cost[s]):
                    continue

                if succ not in ss.hm_:
                    succ_node = State(succ)
                    succ_node.h = ss.eps_ * ENV.get_heur(succ)
                    ss.hm_[succ] = succ_node
                else:
                    succ_node = ss.hm_[succ]

                succ_node.pred_coord.append(curr_node.coord)
                succ_node.pred_action_cost.append(succ_cost[s])
                succ_node.pred_action_id.append(succ_act_id[s])

                tentative_gval = curr_node.g + succ_cost[s]

                if tentative_gval < succ_node.g:
                    succ_node.g = tentative_gval
                    fval = succ_node.g + ss.eps_ * succ_node.h

                    if succ_node.iterationopened and not succ_node.iterationclosed:
                        # succ_node.heapkey = (fval, succ_node)
                        # heapq.heappush(ss.pq_, succ_node.heapkey)
                        ss.pq_[succ_node] = fval # update priority
                    else:
                        # succ_node.heapkey = (fval, succ_node)
                        # heapq.heappush(ss.pq_, succ_node.heapkey)
                        ss.pq_[succ_node] = fval
                        succ_node.iterationopened = True

            if ENV.is_goal(curr_node.coord):
                break

            if ENV.plan_timeout():
                print("Reach Max Search Time!")
                find_traj, traj_prs = self.recover_traj(best_node, ss, ENV, start_coord)
                return best_node.g, traj_prs

            if max_expand > 0 and expand_iteration >= max_expand:
                print("MaxExpandStep reached!")
                return float('inf'), []

        if self.verbose:
            fval = ss.calculateKey(curr_node)
            print(f"goalNode fval: {fval}, g: {curr_node.g}")
            print(f"Expand {expand_iteration} nodes!")

        ss.expand_iteration_ = expand_iteration
        find_traj, traj_prs = self.recover_traj(curr_node, ss, ENV, start_coord)
        if find_traj:
            return curr_node.g, traj_prs
        else:
            return float('inf'), []
        

    def recover_traj(self, curr_node, ss, ENV, start_key):
        print("--------------------------------------------")
        ss.best_child_.clear()
        find_traj = False

        prs = []
        while curr_node.pred_coord:
            if self.verbose:
                print(f"t: {curr_node.coord.t} pos: {curr_node.coord.pos.transpose()} vel: {curr_node.coord.vel.transpose()}")
                print(f"g: {curr_node.g}, rhs: {curr_node.rhs}, h: {curr_node.h}")

            ss.best_child_.append(curr_node)
            min_id = -1
            min_rhs = float('inf')
            min_g = float('inf')

            for i, key in enumerate(curr_node.pred_coord):
                tentative_rhs = ss.hm_[key].g + curr_node.pred_action_cost[i]
                if min_rhs > tentative_rhs:
                    min_rhs = tentative_rhs
                    min_g = ss.hm_[key].g
                    min_id = i
                elif not np.isinf(curr_node.pred_action_cost[i]) and min_rhs == tentative_rhs:
                    if min_g < ss.hm_[key].g:
                        min_g = ss.hm_[key].g
                        min_id = i

            if min_id >= 0:
                key = curr_node.pred_coord[min_id]
                action_idx = curr_node.pred_action_id[min_id]
                # if ENV.get_type():
                #     yaw_idx = curr_node.pred_yaw_id[min_id]
                curr_node = ss.hm_[key]
                # pr = Primitive()
                # if not ENV.get_type():
                pr = ENV.forward_action(curr_node.coord, action_idx)
                # else:
                #     ENV.forward_action(curr_node.coord, action_idx, yaw_idx, pr)
                prs.append(pr)
            else:
                if self.verbose:
                    print("Trace back failure, the number of predecessors is {}:".format(len(curr_node.pred_coord)))
                    for i, key in enumerate(curr_node.pred_coord):
                        print(f"i: {i}, t: {key.t}, g: {ss.hm_[key].g}, rhs: {ss.hm_[key].rhs}, action cost: {curr_node.pred_action_cost[i]}")

                break

            if curr_node.coord == start_key:
                ss.best_child_.append(curr_node)
                find_traj = True
                if self.verbose:
                    print(f"t: {curr_node.coord.t} pos: {curr_node.coord.pos.transpose()}")
                    print(f"g: {curr_node.g}, rhs: {curr_node.rhs}, h: {curr_node.h}")
                break

        prs.reverse()
        ss.best_child_.reverse()
        # traj.prs = prs if find_traj else []
        # return find_traj
        if find_traj:
            return find_traj, prs
        else:
            return find_traj, []