import numpy as np
import torch
# this is very specific to use points but not voxels as the map, so we can interface with Gaussians


# TODO: we need to keep Gaussians here somehow
# define the collisions check here
class MapUtil:

    def __init__(self, x_min, x_max, y_min, y_max, agent_radius):
        # Let's set a map boundary
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.agent_radius = agent_radius
        self.gaussians = {}

    def set_gaussians(self, gaussians):
        self.gaussians = gaussians

    # TODO: vectorize this so we check collision for all points at once
    def is_free(self, pt):
        # check if the point is free
        if pt.shape == (3, ):
            # make it a 1x3 array
            pt = pt.reshape(1, 3)
        ret = self.collision_testing(pt)
        return torch.sum(ret[:, :, 1]) == 0

    
    # TODO: vectorize this so we check collision for all points at once
    def is_occupied(self, pt):
        # check if the point is occupied
        if pt.shape == (3, ):
            # make it a 1x3 array
            pt = pt.reshape(1, 3)
        ret = self.collision_testing(pt)
        # sum up the results[:, :, 1] to see if there is any True
        return torch.sum(ret[:, :, 1]) > 0


    def is_outside(self, pt):
        # check if the point is outside
        if pt[0] < self.x_min or pt[0] > self.x_max or pt[1] < self.y_min or pt[1] > self.y_max:
            return True


    def collision_testing(self, points):
        '''
        This function computes each points distance to all the gaussians in gaussians['means3D']
        and save the distance to each gaussian and whether it is smaller than the radius
        points: [num_points, 3]
        gaussians['means3D']: [num_gaussians, 3] centers of points
        gaussians['radius']: [num_gaussians] the radius of the gaussians
        return: [num_points, num_gaussians, 2] where the first column is the distance to each gaussian
        and the second column is whether the distance is smaller than the 3*radius+agent radius
        '''
        print("shape of points: ", points.shape)
        print("shape of gaussians: ", self.gaussians['means3D'].shape)
        print("shape of radius: ", self.gaussians['radius'].shape)
        # if poitns not tensor, convert to tensor
        if not torch.is_tensor(points):
            points = torch.tensor(points)

        num_points = points.shape[0]
        num_gaussians = self.gaussians['means3D'].shape[0]
        
        # Initialize the output array
        results = torch.zeros(num_points, num_gaussians, 2)
        
        # Vectorized implementation
        points = points.unsqueeze(1).repeat(1, num_gaussians, 1)
        gaussians = self.gaussians['means3D'].unsqueeze(0).repeat(num_points, 1, 1)
        radii = self.gaussians['radius'].unsqueeze(0).repeat(num_points, 1)
        
        # Compute distances
        dists = torch.norm(points - gaussians, dim=-1)
        
        # Check if distances are smaller than radii
        within_radius = dists < (radii*3 + self.agent_radius)
        
        # Fill the results array
        results[:, :, 0] = dists
        results[:, :, 1] = within_radius.float()  # convert boolean to float for storage
        
        print("return results shape: ", results.shape)
        print(results)
        return results