# reference https://pytorch3d.org/tutorials/render_textured_meshes

import os
import random
import sys
import torch
import numpy as np
import matplotlib.image
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    SoftPhongShader,
    TexturesVertex
)

# A helper function to load an OFF file as a PyTorch3D mesh
def load(filename ,device):
    verts, faces, aux = load_obj(filename)
    faces_idx = faces.verts_idx.to(torch.int64)
    verts_rgb = torch.ones_like(verts)[None]
    textures = TexturesVertex(verts_features=verts_rgb.to(device))
    mesh = Meshes(
        verts=[verts.to(device)],
        faces=[faces_idx.to(device)],
        textures=textures
    )
    return mesh

# A helper function to render a mesh into an RGB image and a depth image
def render_mesh(mesh, device, image_size=256, elev=0, azim=60):
    # Set the camera position
    R, T = look_at_view_transform(dist=100, elev=elev, azim=azim)
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)

    # Set the light source position
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Set the rasterizer settings
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Set the shader settings
    shader = HardPhongShader(device=device, cameras=cameras, lights=lights)

    # Create a renderer object
    renderer = MeshRenderer(rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings),
        shader=shader
    )

    # Render the mesh into an RGB image and a depth image
    rgb_image = renderer(mesh)
    rgb_image = rgb_image[..., :3].squeeze().cpu().numpy()
    # TODO: Have some issue in depth rendering
    depth_image = renderer(mesh, cameras=cameras, lights=lights).squeeze().cpu().numpy()

    return rgb_image, depth_image

def get_file_paths(directory_path):
    file_paths = []
    for root, directories, files in os.walk(directory_path):
        for filename in files:
            if filename[-4:] == ".obj":
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
    return file_paths


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        print("ERROR: Pass the folder root and output folder", file=sys.stderr)
        exit(1)
    # Set the device
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Load a sample obj file from the dataset
    folder_root = sys.argv[1]
    categories_folder = [f.path for f in os.scandir(folder_root) if f.is_dir()]
    selected_file = []
    for c in categories_folder: 
        paths = get_file_paths(c)
        selected_file += random.choices(paths, k=1)

    azim = [0, 60, 90, 180]
    output_folder = sys.argv[2]

    for i, filename in enumerate(selected_file):
        mesh = load(filename, device)
        # Render the mesh into an RGB image and a depth image with some camera angles
        for a in azim:
            rgb_image, depth_image = render_mesh(mesh, device, elev=0, azim=a)
            out_filename = (f"render_{i}.png", f"depth_{i}.png")
            matplotlib.image.imsave(os.path.join(output_folder, out_filename[0]), rgb_image)
            matplotlib.image.imsave(os.path.join(output_folder, out_filename[1]), depth_image)