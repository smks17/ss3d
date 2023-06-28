import os
import sys

import trimesh

def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, n_dontknow = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces

def get_file_paths(directory_path):
    file_paths = []
    for root, directories, files in os.walk(directory_path):
        for filename in files:
            if filename[-4:] == ".off":
                file_path = os.path.join(root, filename)
                file_paths.append(file_path)
    return file_paths


def off2obj(root_folder):
    print("start...")
    for filepath in get_file_paths(root_folder):
        if filepath[-4:] == ".off":
            with open(filepath, "r") as f:
                mesh = trimesh.Trimesh(*read_off(f))
            out_filename =  filepath[:-4] + ".obj"
            data = mesh.export(file_type='obj')
            with open(out_filename, "w") as f:
                f.write(data)
    print("finish!")

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("ERROR: Pass the root folder", file=sys.stderr)
        exit(1)
    off2obj(sys.argv[1])