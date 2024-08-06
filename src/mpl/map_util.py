import numpy as np
import torch
# this is very specific to use points but not voxels as the map, so we can interface with Gaussians


# TODO: we need to keep Gaussians here somehow
# define the collisions check here
class MapUtil:

    def __init__(self, x_min, x_max, y_min, y_max, res):
        # Let's set a map boundary
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.gaussians = {}
    
    def is_free(self, pt):
        # check if the point is free
        pass


    def is_occupied(self, pt):
        # check if the point is occupied
        pass

    def is_outside(self, pt):
        # check if the point is outside
        pass
