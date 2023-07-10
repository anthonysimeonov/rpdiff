import trimesh
import os, os.path as osp

meshesv = [osp.join(os.getcwd(), 'visual', fn) for fn in os.listdir('visual')]
meshesc = [osp.join(os.getcwd(), 'collision', fn) for fn in os.listdir('collision')]

meshes = meshesv + meshesc

for i, fname in enumerate(meshes):
    mesh = trimesh.load(fname)
    new_fname = fname.replace('.stl', '.obj')
    mesh.export(new_fname)
