import numpy as np
import torch
# this is very specific to use points but not voxels as the map, so we can interface with Gaussians
import rospy

# TODO: we need to keep Gaussians here somehow
# define the collisions check here
class MapUtil:

    def __init__(self, x_min, x_max, y_min, y_max, agent_radius, collision_tol):
        # Let's set a map boundary
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.agent_radius = agent_radius
        self.collision_tol = collision_tol
        self.gaussians = {}
        print("[MapUtil] map boundary, x_min: ", self.x_min)
        print("[MapUtil] map boundary, x_max: ", self.x_max)
        print("[MapUtil] map boundary, y_min: ", self.y_min)
        print("[MapUtil] map boundary, y_max: ", self.y_max)
        print("[MapUtil] agent radius: ", self.agent_radius)
        print("[MapUtil] collision tol: ", self.collision_tol)


    def set_gaussians(self, gaussians):
        self.gaussians = gaussians


    def is_free(self, pts):
        if not torch.is_tensor(pts):
            pts = torch.tensor(pts).to(self.gaussians['means3D'].device)
        # check if a single point is free
        if pts.shape == (3, ):
            # make it a 1x3 array
            pts = pts.reshape(1, 3)
        # ret = self.collision_testing(pts)
        ret = self.new_collision_testing(pts)
        # return torch.all(ret[:, :, 1] == 0)
        return torch.sum(ret[:, :, 1]) <= self.collision_tol


    def is_occupied(self, pts):
        if not torch.is_tensor(pts):
            pts = torch.tensor(pts).to(self.gaussians['means3D'].device)
        # check if a single point is occupied
        if pts.shape == (3, ):
            # make it a 1x3 array
            pts = pts.reshape(1, 3)
        # ret = self.collision_testing(pts)
        ret = self.new_collision_testing(pts)
        # return torch.any(ret[:, :, 1] > 0)
        return torch.sum(ret[:, :, 1]) > self.collision_tol


    def is_outside(self, pts):
        if not torch.is_tensor(pts):
            pts = torch.tensor(pts).to(self.gaussians['means3D'].device)
        # pts is an n x 3 tensor where each row is a 3D point.
        # Extract the x and y coordinates.
        x = pts[:, 0]
        y = pts[:, 1]
        # comparison for all points at once
        outside_x = (x < self.x_min) | (x > self.x_max)
        outside_y = (y < self.y_min) | (y > self.y_max)
        # combine
        is_outside = outside_x | outside_y
        # Return a boolean tensor indicating which points are outside
        return is_outside.any()

    def new_is_occupied(self, pts):
        '''
        Return 2D array of shape (M, 2)
        First row is whether the traj has collision
        Second row is the collision cost
        '''
        # pts is Nx3 or MxNx3
        if not torch.is_tensor(pts):
            pts = torch.tensor(pts).to(self.gaussians['means3D'].device)
        
        # Handle single point case (1x3)
        if pts.shape[-1] == 3 and pts.dim() == 1:
            pts = pts.reshape(1, 3)
        
        # Apply collision testing
        # ret = [M, N, num_gaussians, 2] or [N, num_gaussians, 2] if input is [N, 3]
        ret = self.new_collision_testing(pts)
        
        # If the input is MxNx3, return a tensor of shape (M,)
        if pts.dim() == 3:
            # First, sum over Kx2 to get MxN. Each element is number of collisions for each point in each trajectory
            all_collisions = torch.sum(ret[:, :, :, 1], dim=2)
            # Then, compare to collision_tol for each element
            collision_filtered = all_collisions > self.collision_tol
            # print("collision_filtered: ", collision_filtered)
            # reduce to M dimension by sum over N
            # print(torch.any(collision_filtered, dim=1))

            # Collision cost
            dist = ret[:, :, :, 0]
            dist[dist > 3] = 0
            dist = -dist + 3
            dist_cost = torch.sum(dist, dim=(1,2))
            return dist_cost, torch.any(collision_filtered, dim=1) # Checks across N dimension
        else:
            all_collisions = torch.sum(ret[:, :, 1], dim=1)
            collision_filtered = all_collisions > self.collision_tol
            # print("else:" , torch.any(collision_filtered, dim=1))
            # collision cost
            dist = ret[:, :, 0]
            dist[dist > 3] = 0
            dist = -dist + 3
            dist_cost = torch.sum(dist[:, :, 0], dim=1)
            return dist_cost, torch.any(collision_filtered, dim=1)
            # return torch.any(collision_filtered, dim=1)  # Checks across N dimension
            # return torch.sum(ret[:, 1]) > self.collision_tol

    # def collision_testing(self, points, use_point_radius=True):
    #     '''
    #     This function computes each points distance to all the gaussians in gaussians['means3D']
    #     and save the distance to each gaussian and whether it is smaller than the radius
    #     points: [num_points, 3]
    #     gaussians['means3D']: [num_gaussians, 3] centers of points
    #     gaussians['radius']: [num_gaussians,1] the radius of the gaussians
    #     return: [num_points, num_gaussians, 2] where the first column is the distance to each gaussian
    #     and the second column is whether the distance is smaller than the 3*radius+agent radius
    #     '''
    #     # Mask out min_height
    #     # take first point height
    #     # min_z = points[0, 2] + 0.5
    #     # z_mask = self.gaussians['means3D'][:, 2] > min_z
    #     gaussians = self.gaussians['means3D']
    #     if use_point_radius:
    #         radii = self.gaussians['radius']

    #         radii = radii.squeeze()
    #     # print("shape of points: ", points.shape)
    #     # print("shape of gaussians: ", self.gaussians['means3D'].shape)
    #     # print("shape of radius: ", self.gaussians['radius'].shape)
    #     # if poitns not tensor, convert to tensor
    #     if not torch.is_tensor(points):
    #         points = torch.tensor(points).to(self.gaussians['means3D'].device)

    #     num_points = points.shape[0]
    #     num_gaussians = gaussians.shape[0]
        
    #     # Initialize the output array
    #     results = torch.zeros(num_points, num_gaussians, 2)
        
    #     # Vectorized implementation
    #     points = points.unsqueeze(1).repeat(1, num_gaussians, 1)
    #     gaussians = gaussians.unsqueeze(0).repeat(num_points, 1, 1)
    #     if use_point_radius:
    #         radii = radii.unsqueeze(0).repeat(num_points, 1)
        
    #     # Compute distances
    #     dists = torch.norm(points - gaussians, dim=-1)
        
    #     # Check if distances are smaller than radii
    #     if use_point_radius:
    #         within_radius = dists < (radii*3 + self.agent_radius)
    #     else:
    #         within_radius = dists < (self.agent_radius)
        
    #     # Fill the results array
    #     results[:, :, 0] = dists
    #     results[:, :, 1] = within_radius.float()  # convert boolean to float for storage
        
    #     # print("return results shape: ", results.shape)
    #     # print(results)
    #     return results

    def collision_testing_debug(self, points, use_point_radius=True):
        '''
        This function computes each points distance to all the gaussians in gaussians['means3D']
        and save the distance to each gaussian and whether it is smaller than the radius
        points: [num_points, 3]
        gaussians['means3D']: [num_gaussians, 3] centers of points
        gaussians['radius']: [num_gaussians,1] the radius of the gaussians
        return: [num_points, num_gaussians, 2] where the first column is the distance to each gaussian
        and the second column is whether the distance is smaller than the 3*radius+agent radius
        '''
        # Mask out min_height
        gaussians = self.gaussians['means3D']
        # gaussians = self.gaussians['means3D'][z_mask]
        if use_point_radius:
            radii = self.gaussians['radius']
            # radii = self.gaussians['radius'][z_mask]

            # Ensure radius is squeezed to correct shape
            radii = radii.squeeze()
        
        # Ensure points is a tensor and move it to the correct device
        if not torch.is_tensor(points):
            points = torch.tensor(points).to(self.gaussians['means3D'].device)
        
        # Handle both [M, N, 3] and [N, 3] cases
        if points.dim() == 2:
            points = points.unsqueeze(0)  # Convert [N, 3] to [1, N, 3]

        M, N, _ = points.shape
        num_gaussians = gaussians.shape[0]
        
        # Initialize the output array
        results = torch.zeros(M, N, num_gaussians, 2, device=points.device)
        
        # Vectorized implementation
        points = points.unsqueeze(2).repeat(1, 1, num_gaussians, 1)  # Shape: [M, N, num_gaussians, 3]
        gaussians = gaussians.unsqueeze(0).unsqueeze(0).repeat(M, N, 1, 1)  # Shape: [M, N, num_gaussians, 3]
        if use_point_radius:
            radii = radii.unsqueeze(0).unsqueeze(0).repeat(M, N, 1)  # Shape: [M, N, num_gaussians]
        
        # Compute distances
        dists = torch.norm(points - gaussians, dim=-1)  # Shape: [M, N, num_gaussians]
        
        # Check if distances are smaller than radii\
        if use_point_radius:
            within_radius = dists < (radii * 3 + self.agent_radius)  # Shape: [M, N, num_gaussians]
        else:
            within_radius = dists < self.agent_radius
        
        # Fill the results array
        results[:, :, :, 0] = dists
        results[:, :, :, 1] = within_radius.float()  # Convert boolean to float for storage
        
        # If the input was [N, 3], return [N, num_gaussians, 2] instead of [1, N, num_gaussians, 2]
        if M == 1:
            results = results.squeeze(0)

        return results, gaussians

    def new_collision_testing(self, points, use_point_radius=True):
        '''
        This function computes each point's distance to all the gaussians in gaussians['means3D']
        and checks whether the distance is smaller than the radius.
        
        points: [M, N, 3] or [N, 3]
        gaussians['means3D']: [num_gaussians, 3] centers of points
        gaussians['radius']: [num_gaussians,1] the radius of the gaussians
        
        return: [M, N, num_gaussians, 2] or [N, num_gaussians, 2] if input is [N, 3]
        '''
        # Mask out min_height
        # take first point height
        # if points.dim() == 2:
        #     min_z = points[0, 2] + 0.5
        # else:
        #     min_z = points[0, 0, 2] + 0.5
        # z_mask = self.gaussians['means3D'][:, 2] > min_z
        gaussians = self.gaussians['means3D']
        # gaussians = self.gaussians['means3D'][z_mask]
        if use_point_radius:
            radii = self.gaussians['radius']
            # radii = self.gaussians['radius'][z_mask]

            # Ensure radius is squeezed to correct shape
            radii = radii.squeeze()
        
        # Ensure points is a tensor and move it to the correct device
        if not torch.is_tensor(points):
            points = torch.tensor(points).to(self.gaussians['means3D'].device)
        
        # Handle both [M, N, 3] and [N, 3] cases
        if points.dim() == 2:
            points = points.unsqueeze(0)  # Convert [N, 3] to [1, N, 3]

        M, N, _ = points.shape
        num_gaussians = gaussians.shape[0]
        
        # Initialize the output array
        results = torch.zeros(M, N, num_gaussians, 2, device=points.device)
        
        # Vectorized implementation
        points = points.unsqueeze(2).repeat(1, 1, num_gaussians, 1)  # Shape: [M, N, num_gaussians, 3]
        gaussians = gaussians.unsqueeze(0).unsqueeze(0).repeat(M, N, 1, 1)  # Shape: [M, N, num_gaussians, 3]
        if use_point_radius:
            radii = radii.unsqueeze(0).unsqueeze(0).repeat(M, N, 1)  # Shape: [M, N, num_gaussians]
        
        # Compute distances
        dists = torch.norm(points - gaussians, dim=-1)  # Shape: [M, N, num_gaussians]
        
        # Check if distances are smaller than radii\
        if use_point_radius:
            within_radius = dists < (radii * 3 + self.agent_radius)  # Shape: [M, N, num_gaussians]
        else:
            within_radius = dists < self.agent_radius
        
        # Fill the results array
        results[:, :, :, 0] = dists
        results[:, :, :, 1] = within_radius.float()  # Convert boolean to float for storage
        
        # If the input was [N, 3], return [N, num_gaussians, 2] instead of [1, N, num_gaussians, 2]
        if M == 1:
            results = results.squeeze(0)

        return results
