import os, os.path as osp
import copy
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import meshcat

# from airobot.utils import common

from rpdiff_robot.utils import util, path_util, trimesh_util


def rhl(bounds):
    high, low = max(bounds), min(bounds)
    sample = np.random.random() * (high - low) + low
    return sample

dev_op = 0.3

def make_can_cabinet(bl, bw, bt, n_shelves, n_cans_per_shelf_max,
                   cr, ch,
                   wh, wt, top_bool, per_can_prob, eps,
                   scale_stack_prob=0.3, stack_hsr=(0.5, 0.95), stack_rsr=(0.75, 0.95),
                   remove_stack_prob=0.3, new_stack_can_prob=0.5, dense=False, nonuniform=True,
                   show=False, mc_vis=None, mc_show=False):
    """
    Args:
        bl (float): Base length
        bw (float): Base width
        bt (float): Base thickness
        n_shelves (int): Number of shelves (between 1 and 3)
        bkl (float): Book length mean
        bkw (float): Book width mean
        bkt (float): Book thickness mean
        bkl_range (float): +/- range for can length
        bkw_range (float): +/- range for can width
        bkt_range (float): +/- range for can thickness
        wt (float): Wall thickness
        top_bool (bool): If True, add a top

    Returns:
        trimesh.Trimesh: concatenated mesh of the whole canshelf
        dict: Dictionary with all the part meshes
    """
    scene = trimesh.Scene()
    # l, h, t corresponds to [x, y, z]

    # can, min and max dimensions
    min_can_r, min_can_h = cr[0], ch[0]
    max_can_r, max_can_h = cr[1], ch[1]

    min_can_dims = [min_can_r, min_can_h]
    max_can_dims = [max_can_r, max_can_h]

    have_can_mesh = False
    can_mesh = None

    # base 
    # bl, bw, bt = 0.08, 0.22, 0.005
    
    # base length based on can radius
    bl = max_can_r * 2 * rhl((1.1, 1.7))

    base_dims = [bl, bw, bt]
    base = trimesh.creation.box(base_dims)

    shelves = {}
    if n_shelves > 1:
        shelves = {k: base.copy() for k in range(n_shelves - 1)}

    top = base.copy()

    # shelf height based on can height
    # sh = np.array([0, 0, max_can_h * n_cans_per_shelf_max]) * 1.15
    sh = np.array([0, 0, np.mean([max_can_h, min_can_h]) * n_cans_per_shelf_max]) * 1.25

    wh = sh[-1]  # wall height based on shelf height
    for i, v in enumerate(shelves.values()):
        trans = sh * (i + 1)
        v.apply_translation(trans)

    top.apply_translation(n_shelves * sh)

    scene.add_geometry([base, top] + list(shelves.values()))

    # global wall dims
    # wt = 0.005  # wall thickness
    # wh = 0.04  # wall height

    # wh = n_shelves * sh[2]  # wall height

    # front/back walls
    # fbl, fbw, fbt = wh, bw, wt
    fbl, fbw, fbt = wh*n_shelves, bw, bt
    fb_dims = [fbl, fbw, fbt]
    back_wall = trimesh.creation.box(fb_dims)
    # scene.add_geometry([base, front_wall])
    
    # side walls
    sl, sw, st = bl, wh*n_shelves, wt
    side_dims = [sl, sw, st]
    left_wall = trimesh.creation.box(side_dims)
    right_wall = trimesh.creation.box(side_dims)
    side_walls = dict(left=left_wall, right=right_wall)
    # scene.add_geometry([base, right_wall])

    # transformations for all the walls
    # front/back need to translate in x and rotate about y (pitch)
    # back_trans = np.array([-bl/2, 0, wh/2])
    back_trans = np.array([-bl/2, 0, wh/2*n_shelves])
    back_rot = R.from_euler('xyz', [0, -(np.pi/2), 0]).as_matrix()
    back_trans_mat = np.eye(4); back_trans_mat[:-1, :-1] = back_rot; back_trans_mat[:-1, -1] = back_trans

    back_wall.apply_transform(back_trans_mat)
        
    # left/right need to translate in y and rotate about x (roll)
    # left_trans = np.array([0, bw/2, wh/2])
    left_trans = np.array([0, bw/2, wh/2*n_shelves])
    left_rot = R.from_euler('xyz', [-(np.pi/2), 0, 0]).as_matrix()
    left_trans_mat = np.eye(4); left_trans_mat[:-1, :-1] = left_rot; left_trans_mat[:-1, -1] = left_trans

    # right_trans = np.array([0, -bw/2, wh/2])
    right_trans = np.array([0, -bw/2, wh/2*n_shelves])
    right_rot = R.from_euler('xyz', [(np.pi/2), 0, 0]).as_matrix()
    right_trans_mat = np.eye(4); right_trans_mat[:-1, :-1] = right_rot; right_trans_mat[:-1, -1] = right_trans

    side_trans = dict(left=left_trans_mat, right=right_trans_mat)
    for k, v in side_walls.items():
        v.apply_transform(side_trans[k])

    # scene.add_geometry([base, top, back_wall] + list(shelves.values()) + list(side_walls.values()))
    # scene.show()

    # # merge
    mesh_dict = {'base': base, 'back_wall': back_wall}
    for k, v in shelves.items():
        mesh_dict[k] = v
    for k, v in side_walls.items():
        mesh_dict[k] = v
    if top_bool:
        mesh_dict['top'] = top

    full_bookshelf_mesh = trimesh.util.concatenate(list(mesh_dict.values()))

    # if show:
    #     # scene.add_geometry(
    #     #     [base, back_wall] + 
    #     #     list(shelves.values()) + 
    #     #     list(side_walls.values())) 
    #     if mc_vis is not None:
    #         util.meshcat_trimesh_show(mc_vis, 'scene/full_bookshelf', full_bookshelf_mesh)
    #     # scene.show()
    #     input('waiting for user to view in meshcat')

    util.meshcat_trimesh_show(mc_vis, f'scene/make_can_cab/full_bookshelf', full_bookshelf_mesh, opacity=dev_op)
    # if mc_show:
    #     util.meshcat_trimesh_show(mc_vis, 'scene/full_bookshelf', full_bookshelf_mesh)

    canonical_base_can_mesh_list = []
    base_can_mesh_list = []
    
    # keep track of per-shelf, and per-slot/stack, and per-can info
    shelf_info_list = []
    for sh_idx in range(n_shelves):
        start_right = np.random.random() > 0.5

        canonical_base_can_mesh_shelf_list = []
        base_can_mesh_shelf_list = []
        
        slot_info_list = []
        base_can_info_list = []
        can_idx = 0
        total_dist = 0.0
        # nom_y_last = bw/2
        nom_y_last = bw/2 - bt - bt*2
        # now create slots for all the rest the cans, based on randomly sampled can radius values
        while True:
            
            # if total_dist > 0.95 * bw:
            #     print(f'Maximum number reached')
            #     break

            can_idx += 1
            
            if dense:
                if can_idx == 1 or nonuniform:
                    canr = rhl((min_can_r, max_can_r*0.5))
                    canh = rhl((min_can_h, max_can_h))
            else:
                canr = rhl((min_can_r, max_can_r*0.5))
                canh = rhl((min_can_h, max_can_h))

            # canr = rhl((min_can_r, max_can_r*0.5))
            # # canr = rhl((min_can_r, max_can_r))
            # canh = rhl((min_can_h, max_can_h))

            # canr_p = canr*1.6
            # canr_p = canr*1.2
            if dense:
                canr_p = canr
            else:
                canr_p = canr*1.2

            # nom_x = -bl/2 + canr + bt  # -shelf length/2 + shelf thickness + can radius
            if dense:
                if can_idx == 1:
                    nom_x = 0.0 + rhl((-bl/5.0, bl/5.0))  # origin plus delta
            else:
                nom_x = 0.0 + rhl((-bl/5.0, bl/5.0))  # origin plus delta
            # nom_x = 0.0 + rhl((-bl/5.0, bl/5.0))  # origin plus delta
            
            if dense:
#                 nom_y_abs = bw/2 - total_dist - bt - canr_p - bt*2 # shelf width/2 - shelf thickness - can radius
                nom_y_abs = bw/2 - total_dist - canr_p # shelf width/2 - shelf thickness - can radius
                # if can_idx == 1:
                if True:
                    nom_y_abs -= 3*bt # shelf width/2 - shelf thickness - can radius
                # nom_y_abs = bw/2 - total_dist - bt - canr_p - bt*2 # shelf width/2 - shelf thickness - can radius
            else:
                nom_y_abs = bw/2 - total_dist - bt - canr_p - rhl((0.0, bt*3)) # shelf width/2 - shelf thickness - can radius
            # nom_y_abs = bw/2 - total_dist - bt - canr_p - rhl((0.0, bt*3)) # shelf width/2 - shelf thickness - can radius

            if start_right:
                nom_y = nom_y_abs
            else:
                nom_y = -1.0*nom_y_abs
            
            if nom_y_abs < (-1.0*bw/2+canr):
                # print(f'Maximum dist reached')
                break

            # nom_y = nom_y if start_right else -1.0*nom_y
            nom_z = canh/2 + bt + sh[-1]*sh_idx

            can_trans = np.array([nom_x, nom_y, nom_z])
            can_rot = R.from_euler('xyz', [0, 0, 0]).as_matrix()
            can_tf = np.eye(4); can_tf[:-1, :-1] = can_rot; can_tf[:-1, -1] = can_trans

            nominal_can_unposed = trimesh.creation.cylinder(radius=canr, height=canh)
            nominal_can = trimesh.creation.cylinder(radius=canr, height=canh).apply_transform(can_tf)

            canonical_base_can_mesh_shelf_list.append(nominal_can_unposed)
            base_can_mesh_shelf_list.append(nominal_can)

            util.meshcat_trimesh_show(mc_vis, f'scene/make_can_cab/cans/shelf_{sh_idx}_can_{can_idx}', nominal_can, opacity=dev_op)
            # if mc_show:
            #     util.meshcat_trimesh_show(mc_vis, f'scene/make_can_cab/cans/shelf_{sh_idx}_can_{can_idx}', nominal_can)
            #     util.meshcat_trimesh_show(mc_vis, f'scene/make_can_cab/cans/can_{can_idx}', nominal_can)

            # save can/slot info
            slot_cent_pose = np.eye(4)
            slot_cent_pose[:-1, -1] = can_trans; slot_cent_pose[2, -1] = bt + sh[-1]*sh_idx  # bottom of slot
            if mc_show:
                util.meshcat_frame_show(mc_vis, f'scene/make_can_cab/slots/shelf_{sh_idx}_slot_{can_idx}', slot_cent_pose    )

            can_pose = can_tf
            can_dims = dict(r=canr, h=canh)

            can_top_pose = copy.deepcopy(can_tf); can_top_pose[2, -1] += canh/2
            if mc_show:
                util.meshcat_frame_show(mc_vis, f'scene/make_can_cab/can_tops/shelf_{sh_idx}_cantop_{can_idx}', can_top_pose )

            can_info = dict(can_pose=can_pose, can_dims=can_dims, can_top_pose=can_top_pose)
            slot_info = dict(slot_cent_pose=slot_cent_pose)
            base_can_info_list.append(can_info)
            slot_info_list.append(slot_info)
            
            if dense:
                # total_dist += (nom_y_last - nom_y_abs)
                # total_dist += (np.abs(nom_y_last) - np.abs(nom_y_abs)) + canr
                # total_dist += (np.abs(nom_y_last) - np.abs(nom_y_abs))
                total_dist += (np.abs(nom_y_last - nom_y_abs)) 
                total_dist += canr
                print(f'Last: {nom_y_last:.4f}, Current: {nom_y_abs:.4f}, Radius: {canr:.4f}')
            else:
                total_dist += (nom_y_last - nom_y_abs) + canr

            # total_dist += (nom_y_last - nom_y_abs) + canr

            # total_dist += (nom_y_last - nom_y) + canr_p

            nom_y_last = nom_y_abs - canr
            # nom_y_last = nom_y_abs
            # nom_y_last = nom_y
        
        shelf_info = dict(slot_info=slot_info_list, base_can_info=base_can_info_list)
        shelf_info_list.append(shelf_info)

        canonical_base_can_mesh_list.append(canonical_base_can_mesh_shelf_list)
        base_can_mesh_list.append(base_can_mesh_shelf_list)

    shelf_dims = dict(base_dims=[bl, bw, bt], shelf_height=sh[-1])
    shelf_info_list.append(shelf_dims)

    # FULL set of can STACK info
    full_stack_info = []
    full_can_mesh_list = []
    full_can_mesh_list_all = []
    full_can_mesh_list_canon_all = []
    full_avail_can_top_poses = []
    for sh_idx in range(n_shelves):
        # go through all slots, and check if the base cans, with their top poses + their dims + the shelf dimensions, can fit more on top

        shelf_stacks_info = []
        full_can_mesh_shelf_list = []

        sh_info = shelf_info_list[sh_idx]
        n_slots = len(sh_info['slot_info'])
        for st_idx in range(n_slots):
            
            # stack info
            slot_info = sh_info['slot_info'][st_idx]
            base_can_info = sh_info['base_can_info'][st_idx]

            base_can_pose = base_can_info['can_pose']
            base_can_dims = base_can_info['can_dims']
            base_can_top_pose = base_can_info['can_top_pose']

            can_height = base_can_dims['h']

            base_can_mesh_canon = canonical_base_can_mesh_list[sh_idx][st_idx]
            base_can_mesh = base_can_mesh_list[sh_idx][st_idx]

            if np.random.random() > (1 - remove_stack_prob):
                avail_stack_dims = base_can_dims
                avail_base_stack_can_top_pose = copy.deepcopy(slot_info['slot_cent_pose'])
                util.meshcat_frame_show(mc_vis, f'scene/make_can_cab/avail_top_poses/top_pose_{len(full_avail_can_top_poses)}', avail_base_stack_can_top_pose)
                avail_pose_info = dict(pose=avail_base_stack_can_top_pose, dims=avail_stack_dims)
                full_avail_can_top_poses.append(avail_pose_info)
                continue

            # sometimes scale the meshes down
            if np.random.random() > 1 - scale_stack_prob:
                stack_height_scale = rhl(stack_hsr)
                stack_radius_scale = rhl(stack_rsr)
                can_stack_radius = base_can_dims['r']*stack_radius_scale
                can_stack_height = base_can_dims['h']*stack_height_scale
                # base_can_stack_mesh_canon = base_can_mesh_canon.copy()
                base_can_stack_mesh_canon = trimesh.creation.cylinder(radius=can_stack_radius, height=can_stack_height)
            else:
                can_stack_height = can_height
                can_stack_radius = base_can_dims['r']
                base_can_stack_mesh_canon = base_can_mesh_canon.copy()
            
            can_stack_top_pose_list = []
            can_stack_top_pose_list.append(base_can_top_pose)
            full_can_mesh_shelf_stack_list = []
            full_can_mesh_shelf_stack_list.append(base_can_mesh)
            full_can_mesh_list_canon_all.append(base_can_mesh_canon)
            full_can_mesh_list_all.append(base_can_mesh)
            
            # check if we can fit a new can
            cur_stack_h = 0
            while True:
                # vertical_space = sh[-1]*(sh_idx+1) - bt - base_can_top_pose[2, -1]*1.1 - cur_stack_h*can_height
                vertical_space = sh[-1]*(sh_idx+1) - bt - base_can_top_pose[2, -1]*1.1 - cur_stack_h*can_stack_height

                # new_can = np.random.random() > 0.5
                new_can = np.random.random() > (1 - new_stack_can_prob)

                # if can_height < vertical_space:
                if can_stack_height < vertical_space and (cur_stack_h < n_cans_per_shelf_max-1) and new_can:
                    new_can_pose = copy.deepcopy(base_can_pose)
                    new_can_pose[2, -1] = base_can_top_pose[2, -1] + can_stack_height*cur_stack_h + can_stack_height/2
                    # new_can_pose[2, -1] += can_height*cur_stack_h
                    # new_can_pose[2, -1] += can_stack_height*cur_stack_h

                    # new_can_mesh = base_can_stack_mesh_canon.copy().apply_transform(new_can_pose)
                    new_can_mesh = base_can_stack_mesh_canon.copy()
                    full_can_mesh_list_canon_all.append(new_can_mesh.copy())
                    # new_can_mesh.apply_transform(new_can_pose)
                    new_can_mesh = new_can_mesh.apply_transform(new_can_pose)

                    util.meshcat_trimesh_show(mc_vis, f'scene/make_can_cab/can_stacks/shelf_{sh_idx}/can_stack_{st_idx}/can_{cur_stack_h}', new_can_mesh, opacity=dev_op)
                    if mc_show:
                        # util.meshcat_trimesh_show(mc_vis, f'scene/make_can_cab/can_stacks/shelf_{sh_idx}/can_stack_{st_idx}/can_{cur_stack_h}', new_can_mesh)
                        util.meshcat_frame_show(mc_vis, f'scene/make_can_cab/can_stacks/shelf_{sh_idx}/can_stack_{st_idx}/can_{cur_stack_h}_pose', new_can_pose)

                    cur_stack_h += 1

                    full_can_mesh_list_all.append(new_can_mesh)
                    full_can_mesh_shelf_stack_list.append(new_can_mesh)
                    new_can_top_pose = copy.deepcopy(new_can_pose); new_can_top_pose[2, -1] += can_stack_height/2
                    can_stack_top_pose_list.append(new_can_top_pose)
                else:
                    # print(f'No more vertical space')
                    break
            
            avail_stack_dims = dict(r=can_stack_radius, h=can_stack_height)
            vertical_space = sh[-1]*(sh_idx+1) - bt - base_can_top_pose[2, -1]*1.1 - cur_stack_h*can_stack_height
            if (cur_stack_h+1) < n_cans_per_shelf_max and can_stack_height < vertical_space:
                avail_base_stack_can_top_pose = copy.deepcopy(can_stack_top_pose_list[-1])
                
                util.meshcat_frame_show(mc_vis, f'scene/make_can_cab/avail_top_poses/top_pose_{len(full_avail_can_top_poses)}', avail_base_stack_can_top_pose)

                avail_pose_info = dict(pose=avail_base_stack_can_top_pose, dims=avail_stack_dims)
                full_avail_can_top_poses.append(avail_pose_info)

            full_can_mesh_shelf_list.append(full_can_mesh_shelf_stack_list)

        full_can_mesh_list.append(full_can_mesh_shelf_list)

    # full_bookshelf_mesh = trimesh.util.concatenate(list(mesh_dict.values()))
    full_can_and_cabinet_mesh = trimesh.util.concatenate([full_bookshelf_mesh] + full_can_mesh_list_all)
    util.meshcat_trimesh_show(mc_vis, f'scene/make_can_cab/full_can_cabinet', full_can_and_cabinet_mesh, opacity=1.0)

    # print('here with full')
    # from IPython import embed; embed()

    # base (single can) dev
    if False:
        # now fill in the cans
        canr = rhl((min_can_r, max_can_r))
        canh = rhl((min_can_h, max_can_h))
        canr_p = canr*1.6

        # nom_x = -bl/2 + canr + bt  # -shelf length/2 + shelf thickness + can radius
        nom_x = 0.0 + rhl((-bl/5.0, bl/5.0))  # origin plus delta
        nom_y = bw/2 - bt - canr_p # shelf width/2 - shelf thickness - can radius
        nom_y = nom_y if start_right else -1.0*nom_y
        nom_z = canh/2
        can_trans = np.array([nom_x, nom_y, nom_z])
        can_rot = R.from_euler('xyz', [0, 0, 0]).as_matrix()
        can_tf = np.eye(4); can_tf[:-1, :-1] = can_rot; can_tf[:-1, -1] = can_trans
        # nominal_can_unposed = trimesh.creation.box(can_dims)
        # nominal_can = trimesh.creation.box(can_dims).apply_transform(can_tf)
        nominal_can_unposed = trimesh.creation.cylinder(radius=canr, height=canh)
        nominal_can = trimesh.creation.cylinder(radius=canr, height=canh).apply_transform(can_tf)

        print('here with nominal can')
        util.meshcat_trimesh_show(mc_vis, 'scene/can1', nominal_can)
    
    if len(full_can_mesh_list_canon_all) < 1:
        raise ValueError('Didn"t get any cans')
    if len(full_avail_can_top_poses) < 1:
        raise ValueError('Didn"t get slots')

    out_can_mesh = full_can_mesh_list_canon_all[np.random.randint(0, len(full_can_mesh_list_canon_all))].copy()

    return full_can_and_cabinet_mesh, full_avail_can_top_poses, out_can_mesh


if __name__ == "__main__":
    # bl = rhl(cfg.BASE_LENGTH_LOW_HIGH) 
    # bw = rhl(cfg.BASE_WIDTH_LOW_HIGH)
    # bt = rhl(cfg.BASE_THICKNESS_LOW_HIGH)
    # wt = rhl(cfg.WALL_THICKNESS_LOW_HIGH)
    # wh = rhl(cfg.WALL_HEIGHT_LOW_HIGH)
    # theta = rhl(cfg.WALL_THETA_LOW_HIGH)

    mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    mc_vis['scene'].delete()

    bl, bw, bt = 0.19, 0.5, 0.005
    wt = 0.005
    n_shelves = 2
    top_bool = True
    bkl, bkw, bkt = 0.18, 0.25, 0.04
    per_can_prob = 0.7

    full_canshelf_mesh, _ = make_canshelf(
        bl=bl, bw=bw, bt=bt, n_shelves=n_shelves,
        bkl=bkl, bkw=bkw, bkt=bkt,
        wt=wt, top_bool=top_bool, per_can_prob=per_can_prob,
        show=True, mc_vis=mc_vis)

    full_canshelf_mesh.show()
    from IPython import embed; embed()
