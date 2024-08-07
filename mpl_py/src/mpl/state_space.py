import numpy as np
import heapq
from mpl.primitive import Primitive
from collections import defaultdict

class State:
    def __init__(self, coord):
        self.coord = coord
        self.succ_coord = []
        self.succ_action_id = []
        self.succ_action_cost = []
        self.pred_coord = []
        self.pred_action_id = []
        self.pred_action_cost = []
        self.pred_yaw_id = []
        self.heapkey = None
        self.g = float('inf')
        self.rhs = float('inf')
        self.h = float('inf')
        self.iterationopened = False
        self.iterationclosed = False

class StateSpace:
    def __init__(self, eps=1.0):
        self.pq_ = []
        self.hm_ = defaultdict(lambda: None)
        self.eps_ = eps
        self.dt_ = 0.0
        self.best_child_ = []
        self.expand_iteration_ = 0
        self.start_t_ = 0.0
        self.start_g_ = 0.0
        self.start_rhs_ = 0.0

    def getInitTime(self):
        if not self.best_child_:
            return 0
        else:
            return self.best_child_[0].coord.t

    def updateNode(self, currNode_ptr):
        if currNode_ptr.rhs != self.start_rhs_:
            currNode_ptr.rhs = float('inf')
            for i in range(len(currNode_ptr.pred_coord)):
                pred_key = currNode_ptr.pred_coord[i]
                if currNode_ptr.rhs > self.hm_[pred_key].g + currNode_ptr.pred_action_cost[i]:
                    currNode_ptr.rhs = self.hm_[pred_key].g + currNode_ptr.pred_action_cost[i]

        if currNode_ptr.iterationopened and not currNode_ptr.iterationclosed:
            self.pq_.remove((self.calculateKey(currNode_ptr), currNode_ptr))
            heapq.heapify(self.pq_)
            currNode_ptr.iterationclosed = True

        if currNode_ptr.g != currNode_ptr.rhs:
            fval = self.calculateKey(currNode_ptr)
            heapq.heappush(self.pq_, (fval, currNode_ptr))
            currNode_ptr.iterationopened = True
            currNode_ptr.iterationclosed = False

    def calculateKey(self, node):
        return min(node.g, node.rhs) + self.eps_ * node.h

    def checkValidation(self, hm):
        for coord, state in hm.items():
            if state is None:
                print("error!!! detect null element!")
                coord.print("Not exist!")
            else:
                null_succ = False
                for succ_coord in state.succ_coord:
                    if succ_coord not in hm:
                        print("error!!! detect null succ !")
                        null_succ = True
                if null_succ:
                    coord.print("From this pred:")
                    print(f"rhs: {state.rhs}, g: {state.g}, open: {state.iterationopened}, closed: {state.iterationclosed}")

        return

        print("Check rhs and g value of closeset")
        close_cnt = 0
        for state in hm.values():
            if state.iterationopened and state.iterationclosed:
                print(f"g: {state.g}, rhs: {state.rhs}")
                close_cnt += 1

        print("Check rhs and g value of openset")
        open_cnt = 0
        for state in hm.values():
            if state.iterationopened and not state.iterationclosed:
                print(f"g: {state.g}, rhs: {state.rhs}")
                open_cnt += 1

        print("Check rhs and g value of nullset")
        null_cnt = 0
        for state in hm.values():
            if not state.iterationopened:
                print(f"g: {state.g}, rhs: {state.rhs}")
                null_cnt += 1

        print(f"hm: [{len(hm)}], open: [{open_cnt}], closed: [{close_cnt}], null: [{null_cnt}]")
