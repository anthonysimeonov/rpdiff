import os, os.path as osp
import numpy as np
import trimesh
import random
import copy
import time
import pybullet as p

from rpdiff.utils import util


class SegmentationAugmentation:
    def __init__(self, img_size, circle_radius_hl=[0.6, 0.05], rectangle_side_hl=[0.1, 0.025]):
        self.img_size = img_size
        s = img_size
        xx, yy = np.linspace(0, 1, s[1]), np.linspace(1, 0, s[0]) 
        gx, gy = np.meshgrid(xx, yy)
        self.pixel_pos = np.vstack([gx.ravel(), gy.ravel()]).T

        self.circle_radius_hl = circle_radius_hl
        self.rectangle_side_hl = rectangle_side_hl
        self.rand_val = lambda high, low: np.random.random() * (high - low) + low

    @staticmethod
    def pix2cont(pixel_coord, img_size):
        x, y = pixel_coord[1] / img_size[1], 1.0 - pixel_coord[0] / img_size[0]
        cont_coord = np.asarray([x, y])
        return cont_coord

    @staticmethod
    def cont2pix(cont_coord, img_size):
        x, y = cont_coord[0], cont_coord[1]
        u, v = (1.0 - y) * img_size[0], x * img_size[1]
        u, v = int(u), int(v)
        pixel_coord = np.asarray([v, u])
        return pixel_coord 

    def sample_halfspace(self, obj_mask):
        """
        Masks part of an image that is on one side of a randomly sampled halfplane

        Args:
            obj_mask (np.ndarray): Size (H, W) of boolean values
        
        Returns:
            np.ndarray: Segmentation mask size (H, W) for just the sampled halfspace
        """
        # sample a random point in the obj mask
        s = self.img_size
        obj_pix = np.where(obj_mask)

        # sample random pixel, map to normalized coords, and get random point in random direction
        center_pix = (np.random.choice(obj_pix[0]), np.random.choice(obj_pix[1]))
        center_pt = self.pix2cont(center_pix, s)
        rand_angle = np.random.random() * 2*np.pi - np.pi
        dx, dy = np.cos(rand_angle)+0.001, np.sin(rand_angle)

        # compute normal vector for line connecting two points
        slope = dy / dx
        n = np.array([-1.0*slope, 1])
        n = n / np.linalg.norm(n)
        offset = np.matmul(n, center_pt)

        line_points = np.matmul(n, self.pixel_pos.T) - offset

        if np.random.random() > 0.5:
            line_inds = np.where(line_points > 0)[0]
        else:
            line_inds = np.where(line_points < 0)[0]
        line_mask = np.zeros(s, dtype=np.uint8).reshape(-1)
        line_mask[line_inds] = 1
        line_mask = line_mask.reshape(s)
        # from rpdiff.utils.fork_pdb import ForkablePdb; ForkablePdb().set_trace()
        return line_mask

    def sample_rectangle(self, obj_mask, outside=True):
        """
        Masks part of an image that is outside of a randomly sampled rectangle

        Args:
            obj_mask (np.ndarray): Size (H, W) of boolean values
            outside (bool): If True, we will keep points OUTSIDE of box 

        Returns:
            np.ndarray: Segmentation mask size (H, W) for everything except the rectangle
        """
        # sample a random point in the obj mask
        s = self.img_size
        obj_pix = np.where(obj_mask)

        # sample random pixel, map to normalized coords, and get random point in random direction
        center_pix = (np.random.choice(obj_pix[0]), np.random.choice(obj_pix[1]))
        center_pt = self.pix2cont(center_pix, s)

        rectangle_mask = np.ones(s, dtype=np.uint8)
        h = self.rand_val(max(self.rectangle_side_hl), min(self.rectangle_side_hl))
        w = self.rand_val(max(self.rectangle_side_hl), min(self.rectangle_side_hl))
        normals = np.array([
            [1, 0], 
            [0, 1],
            [-1, 0], 
            [0, -1],
        ])
        centers = np.array([
            [center_pt[0] + w/2.0, center_pt[1]], 
            [center_pt[0], center_pt[1] + h/2.0],
            [center_pt[0] - w/2.0, center_pt[1]],
            [center_pt[0], center_pt[1] - h/2.0]
        ])
        for i in range(4):
            n = normals[i]
            # n = np.array([-1.0*slope, 1])
            n = n / np.linalg.norm(n)
            # offset = np.matmul(n, center_pt)
            offset = np.matmul(n, centers[i])

            line_points = np.matmul(n, self.pixel_pos.T) - offset

            # line_inds = np.where(line_points > 0)[0]
            line_inds = np.where(line_points < 0)[0]

            line_mask = np.zeros(s, dtype=np.uint8).reshape(-1)
            line_mask[line_inds] = 1
            line_mask = line_mask.reshape(s)
            
            rectangle_mask = rectangle_mask & line_mask
            # print('here in rect mask')
            # from IPython import embed; embed()
        if outside:
            rectangle_mask = np.logical_not(rectangle_mask)
        return rectangle_mask

    def sample_circle(self, obj_mask, inside=True):
        """
        Masks part of an image that is on one side of a randomly sampled circle

        Args:
            obj_mask (np.ndarray): Size (H, W) of boolean values
        
        Returns:
            np.ndarray: Segmentation mask size (H, W) for just the sampled circle 
        """
        # sample a random point in the obj mask
        s = self.img_size
        obj_pix = np.where(obj_mask)

        center_pix = (np.random.choice(obj_pix[0]), np.random.choice(obj_pix[1]))
        center_pt = self.pix2cont(center_pix, s)

        r = self.rand_val(max(self.circle_radius_hl), min(self.circle_radius_hl))
        # r = self.rand_val(0.6, 0.05)
        circle_points = r**2 - (self.pixel_pos[:, 0] - center_pt[0])**2 - (self.pixel_pos[:, 1] - center_pt[1])**2

        if inside:
            circle_inds = np.where(circle_points < 0)[0]
        else:
            circle_inds = np.where(circle_points > 0)[0]

        circle_mask = np.zeros(s, dtype=np.uint8).reshape(-1)
        circle_mask[circle_inds] = 1
        circle_mask = circle_mask.reshape(s)
        # from rpdiff.utils.fork_pdb import ForkablePdb; ForkablePdb().set_trace()
        return circle_mask
