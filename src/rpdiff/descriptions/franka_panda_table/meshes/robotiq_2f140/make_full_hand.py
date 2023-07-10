from rrp_robot.utils.quick_viz_util import *
import trimesh

data = np.load('hand_links.npz')

# link_names_to_fnames = {
# 'robotiq_arg2f_base_link': 'collision/robotiq_arg2f_base_link.obj',
# 'left_outer_knuckle': 'collision/robotiq_arg2f_140_outer_knuckle.obj',
# 'left_outer_finger': 'collision/robotiq_arg2f_140_outer_finger.obj',
# 'left_inner_finger': 'collision/robotiq_arg2f_140_inner_finger.obj',
# 'right_outer_knuckle': 'collision/robotiq_arg2f_140_outer_knuckle.obj',
# 'right_outer_finger': 'collision/robotiq_arg2f_140_outer_finger.obj',
# 'right_inner_finger': 'collision/robotiq_arg2f_140_inner_finger.obj'
# }
link_names_to_fnames = {
'robotiq_arg2f_base_link': 'visual/robotiq_arg2f_base_link.obj',
'left_outer_knuckle': 'visual/robotiq_arg2f_140_outer_knuckle.obj',
'left_inner_knuckle': 'visual/robotiq_arg2f_140_inner_knuckle.obj',
'left_outer_finger': 'visual/robotiq_arg2f_140_outer_finger.obj',
'left_inner_finger': 'visual/robotiq_arg2f_140_inner_finger.obj',
'right_outer_knuckle': 'visual/robotiq_arg2f_140_outer_knuckle.obj',
'right_inner_knuckle': 'visual/robotiq_arg2f_140_inner_knuckle.obj',
'right_outer_finger': 'visual/robotiq_arg2f_140_outer_finger.obj',
'right_inner_finger': 'visual/robotiq_arg2f_140_inner_finger.obj'
}

link_names_to_poses = {data['names'][i]: data['poses'][i] for i in range(len(data['names']))}

# fnames_to_poses = {link_names_to_fnames[key]: link_names_to_poses[key] for key in list(link_names_to_fnames.keys())}

meshes = []
# for fname, pose in fnames_to_poses.items():
for link_name, pose in link_names_to_poses.items():
    if link_name in link_names_to_fnames.keys():
        fname = link_names_to_fnames[link_name]
        if osp.exists(fname):
            mesh = trimesh.load(fname)
            mesh.apply_transform(pose)
            meshes.append(mesh)

# manually add the pads
right_pad_mesh = trimesh.creation.box([0.027, 0.065, 0.0075])
left_pad_mesh = trimesh.creation.box([0.027, 0.065, 0.0075])
right_pad_mesh.apply_transform(link_names_to_poses['right_inner_finger_pad'])
left_pad_mesh.apply_transform(link_names_to_poses['left_inner_finger_pad'])

full_mesh = trimesh.util.concatenate(meshes + [right_pad_mesh, left_pad_mesh])
mc_vis['scene'].delete()
util.meshcat_trimesh_show(mc_vis, 'scene/full_hand', full_mesh)
full_mesh.export('full_hand_2f140.obj')

from IPython import embed; embed()

