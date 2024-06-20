### adopted and changed from https://github.com/RenFeiTemp/SRN/blob/master/realtime/vis_tool.py


import numpy as np
import torch
import os
from enum import Enum
# import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
       
       


def get_sketch_setting(dataset):
    if dataset == 'icvl':
        return [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                (0, 7), (7, 8), (8, 9), (0, 10), (10, 11), (11, 12),
                (0, 13), (13, 14), (14, 15)]
    elif dataset == 'nyu':
        return [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (9, 10), (1, 13),
                (3, 13), (5, 13), (7, 13), (10, 13), (11, 13), (12, 13)]
    elif dataset == 'nyu_full':
        return [(20,3),(3,2),(2,1),(1,0),(20,7),(7,6),(6,5),(5,4),(20,11),(11,10),(10,9),(9,8),(20,15),(15,14),(14,13),(13,12),(20,19),(19,18),(18,17),(17,16),
               (20,21),(20,22)]
    elif dataset == 'msra':
        return [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
                (0, 17), (17, 18), (18, 19), (19, 20)]
    elif dataset == 'hands17':
        return [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 6), (6, 7), (7, 8),
                (2, 9), (9, 10), (10, 11), (3, 12), (12, 13), (13, 14), (4, 15), (15, 16),
                (16, 17), (5, 18), (18, 19), (19, 20)]


class Color(Enum):
    RED = (0, 0, 255)
    GREEN = (75, 255, 66)
    BLUE = (255, 0, 0)
    YELLOW = (17, 240, 244) #(204, 153, 17) 
    PURPLE = (255, 255, 0)
    CYAN = (255, 0, 255)
    BROWN = (204, 153, 17)


def get_sketch_color(dataset):
    if dataset == 'icvl':
        return [Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'nyu':
        return (Color.GREEN, Color.RED, Color.PURPLE, Color.YELLOW, Color.BLUE, Color.BLUE, Color.GREEN,
                Color.RED, Color.PURPLE, Color.YELLOW, Color.BLUE, Color.CYAN, Color.CYAN)
    elif dataset == 'nyu_full':
        return (Color.GREEN,Color.GREEN,Color.GREEN,Color.GREEN, Color.RED, Color.RED, Color.RED, Color.RED,  Color.PURPLE, Color.PURPLE,Color.PURPLE,Color.PURPLE,
                Color.YELLOW,Color.YELLOW,Color.YELLOW,Color.YELLOW,
                Color.BLUE, Color.BLUE,  Color.BLUE, Color.BLUE,
                Color.CYAN, Color.CYAN)
    elif dataset == 'msra':
        return [Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'hands17':
        return [Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED,
              Color.GREEN, Color.GREEN, Color.GREEN,
              Color.BLUE, Color.BLUE, Color.BLUE,
              Color.YELLOW, Color.YELLOW, Color.YELLOW,
              Color.PURPLE, Color.PURPLE, Color.PURPLE,
              Color.RED, Color.RED, Color.RED]



def get_joint_color(dataset):
    if dataset == 'icvl':
        return [Color.CYAN, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'nyu':
        return (Color.GREEN, Color.GREEN, Color.RED, Color.RED, Color.PURPLE, Color.PURPLE, Color.YELLOW, Color.YELLOW,
                Color.BLUE, Color.BLUE, Color.BLUE,
                Color.CYAN, Color.CYAN, Color.CYAN)
    elif dataset == 'nyu_full':
        return (Color.GREEN, Color.GREEN,Color.GREEN, Color.GREEN, Color.RED, Color.RED, Color.RED, Color.RED,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.BLUE, Color.BLUE, Color.BLUE,Color.BLUE,
                Color.CYAN, Color.CYAN, Color.CYAN)
    elif dataset == 'msra':
        return [Color.CYAN, Color.RED, Color.RED, Color.RED, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW,
                Color.PURPLE, Color.PURPLE, Color.PURPLE, Color.PURPLE]
    elif dataset == 'hands17':
        return [Color.CYAN, Color.GREEN, Color.BLUE, Color.YELLOW, Color.PURPLE, Color.RED, Color.GREEN, Color.GREEN, Color.GREEN,
                Color.BLUE, Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.PURPLE, Color.PURPLE, Color.PURPLE,
                Color.RED, Color.RED, Color.RED]


def draw_pose(img, pose, img_cnt=None, gt_pose=None, dataset="nyu"):
    # img is a tensor of shape (H,W) its values are in [-1,1] background is 1
    # pose is tensor or np.array of shape (k,d)
    disply_gt = False
    colors = get_sketch_color(dataset)
    colors_joint = get_joint_color(dataset)
    img = np.array(128 * np.array((img + 1) / 2), dtype=np.uint8)
    pose = np.array(128 * (pose + 1) / 2, dtype=np.uint8)
    img = Image.fromarray(img).convert("RGB")
    
    draw = ImageDraw.Draw(img)


    if disply_gt:
        gt_pose = np.array(128 * (gt_pose + 1) / 2, dtype=np.uint8)
        idx = 0
        for gtp in gt_pose:
            draw.ellipse((int(gtp[0]) - 2, int(gtp[1]) - 2, int(gtp[0]) + 2, int(gtp[1]) + 2),
                     fill=(255,255,255))
            idx = idx + 1

        idx = 0
        for x, y in get_sketch_setting(dataset):
            draw.line((int(gt_pose[x, 0]), int(gt_pose[x, 1]),
                    int(gt_pose[y, 0]), int(gt_pose[y, 1])),
                    fill=(255,255,255), width=1)
            idx = idx + 1


    idx = 0    
    for pt in pose:
        draw.ellipse((int(pt[0]) - 2, int(pt[1]) - 2, int(pt[0]) + 2, int(pt[1]) + 2),
                     fill=colors_joint[idx].value)
        idx = idx + 1
        
    idx = 0
    for x, y in get_sketch_setting(dataset):
        draw.line((int(pose[x, 0]), int(pose[x, 1]),
                   int(pose[y, 0]), int(pose[y, 1])),
                  fill=colors[idx].value, width=1)
        idx = idx + 1

    # fig, ax = plt.subplots()
    # ax.imshow(img)
    # plt.axis('off') 
    # plt.show()
        
    img.save('/home/rrangan2/HandDiff/result_imgs/2d_3d_rand_feat/'+str(img_cnt)+'.png')
    print("Image saved")