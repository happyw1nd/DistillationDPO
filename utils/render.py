import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np
import pyrender
import trimesh
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def offscreen_render(pcd, output_name):

    # load point cloud
    points = np.asarray(pcd.points)
    z_values = points[:, 2]

    lower_bound = np.percentile(z_values, 1)
    upper_bound = np.percentile(z_values, 99) 
    z_values = np.clip(z_values, lower_bound, upper_bound)

    norm = mcolors.Normalize(vmin=z_values.min(), vmax=z_values.max())
    cmap = plt.get_cmap("gist_rainbow_r")
    colors = cmap(norm(z_values))
    colors = colors * 0.5

    # object
    m = pyrender.Mesh.from_points(points, colors=colors)

    # light
    dl = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0,innerConeAngle=0.05, outerConeAngle=0.5)
    light_theta = np.radians(-40)
    light_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],  # x axis ←
        [0.0, np.cos(light_theta), -np.sin(light_theta), 0.0],  # y axis ↓
        [0.0, np.sin(light_theta), np.cos(light_theta), 0.0],  # z axis ·
        [0.0, 0.0, 0.0, 1.0]
    ])

    # camera
    pc = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_theta = np.radians(20)
    camera_pose_rot = np.array([
        [1.0, 0.0, 0.0, 0.0],  # x axis ←
        [0.0, np.cos(camera_theta), -np.sin(camera_theta), 0.0],  # y axis ↓
        [0.0, np.sin(camera_theta), np.cos(camera_theta), 0.0],  # z axis ·
        [0.0, 0.0, 0.0, 1.0]
    ])
    camera_pose_trans = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 50.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    camera_pose = np.matmul(camera_pose_rot, camera_pose_trans)

    # scene
    scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[1.0, 1.0, 1.0])
    scene.add(m)
    scene.add(dl, pose=light_pose)
    scene.add(pc, pose=camera_pose)

    # renderer
    r = pyrender.OffscreenRenderer(viewport_width=1000, viewport_height=700, point_size=3.0)
    flags = pyrender.RenderFlags.SHADOWS_ALL
    color, depth = r.render(scene, flags=flags)
    r.delete()

    # save pics
    Image.fromarray(color).save(output_name)
