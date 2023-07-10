import os, os.path as osp
import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
import trimesh
import meshcat

# from airobot.utils import common

from ndf_robot.utils import util, path_util, trimesh_util

def make_bookshelf(bl, bw, bt, n_shelves, 
                   bkl, bkw, bkt, 
                   bkl_range, bkw_range, bkt_range,
                   wt, top_bool, per_book_prob, eps,
                   show=False, mc_vis=None):
    """
    Args:
        bl (float): Base length
        bw (float): Base width
        bt (float): Base thickness
        n_shelves (int): Number of shelves (between 1 and 3)
        bkl (float): Book length mean
        bkw (float): Book width mean
        bkt (float): Book thickness mean
        bkl_range (float): +/- range for book length
        bkw_range (float): +/- range for book width
        bkt_range (float): +/- range for book thickness
        wt (float): Wall thickness
        top_bool (bool): If True, add a top

    Returns:
        trimesh.Trimesh: concatenated mesh of the whole bookshelf
        dict: Dictionary with all the part meshes
    """
    scene = trimesh.Scene()
    # l, h, t corresponds to [x, y, z]

    # base 
    # bl, bw, bt = 0.08, 0.22, 0.005
    base_dims = [bl, bw, bt]
    base = trimesh.creation.box(base_dims)

    shelves = {}
    if n_shelves > 1:
        shelves = {k: base.copy() for k in range(n_shelves - 1)}

    top = base.copy()

    # shelf height based on book width
    sh = np.array([0, 0, bkw]) * 1.15
    for i, v in enumerate(shelves.values()):
        trans = sh * (i + 1)
        v.apply_translation(trans)

    top.apply_translation(n_shelves * sh)

    scene.add_geometry([base, top] + list(shelves.values()))

    # global wall dims
    # wt = 0.005  # wall thickness
    # wh = 0.04  # wall height
    wh = n_shelves * sh[2]  # wall height

    # front/back walls
    # fbl, fbw, fbt = wh, bw, wt
    fbl, fbw, fbt = wh, bw, bt
    fb_dims = [fbl, fbw, fbt]
    back_wall = trimesh.creation.box(fb_dims)
    # scene.add_geometry([base, front_wall])

    # side walls
    sl, sw, st = bl, wh, wt
    side_dims = [sl, sw, st]
    left_wall = trimesh.creation.box(side_dims)
    right_wall = trimesh.creation.box(side_dims)
    side_walls = dict(left=left_wall, right=right_wall)
    # scene.add_geometry([base, right_wall])

    # transformations for all the walls
    # front/back need to translate in x and rotate about y (pitch)
    back_trans = np.array([-bl/2, 0, wh/2])
    back_rot = R.from_euler('xyz', [0, -(np.pi/2), 0]).as_matrix()
    back_trans_mat = np.eye(4); back_trans_mat[:-1, :-1] = back_rot; back_trans_mat[:-1, -1] = back_trans

    back_wall.apply_transform(back_trans_mat)
        
    # left/right need to translate in y and rotate about x (roll)
    left_trans = np.array([0, bw/2, wh/2])
    left_rot = R.from_euler('xyz', [-(np.pi/2), 0, 0]).as_matrix()
    left_trans_mat = np.eye(4); left_trans_mat[:-1, :-1] = left_rot; left_trans_mat[:-1, -1] = left_trans

    right_trans = np.array([0, -bw/2, wh/2])
    right_rot = R.from_euler('xyz', [(np.pi/2), 0, 0]).as_matrix()
    right_trans_mat = np.eye(4); right_trans_mat[:-1, :-1] = right_rot; right_trans_mat[:-1, -1] = right_trans

    side_trans = dict(left=left_trans_mat, right=right_trans_mat)
    for k, v in side_walls.items():
        v.apply_transform(side_trans[k])

    # scene.add_geometry([base, top, back_wall] + list(shelves.values()) + list(side_walls.values()))
    # scene.show()

    # # merge
    # mesh_dict = {'base': base, 'back_wall': back_wall}
    # for k, v in shelves.items():
    #     mesh_dict[k] = v
    # for k, v in side_walls.items():
    #     mesh_dict[k] = v
    # if top_bool:
    #     mesh_dict['top'] = top
    # full_bookshelf_mesh = trimesh.util.concatenate(list(mesh_dict.values()))

    # if show:
    #     # scene.add_geometry(
    #     #     [base, back_wall] + 
    #     #     list(shelves.values()) + 
    #     #     list(side_walls.values())) 
    #     if mc_vis is not None:
    #         util.meshcat_trimesh_show(mc_vis, 'scene/full_bookshelf', full_bookshelf_mesh)
    #     # scene.show()
    #     input('waiting for user to view in meshcat')

    # now fill in the books

    # book, nominal shape
    book_dims = [bkl, bkw, bkt]
    start_right = np.random.random() > 0.5
    
    nom_x = -bl/2 + bkl/2 + bt
    nom_y = bw/2 - bt - bkt/2
    nom_y = nom_y if start_right else -1.0*nom_y
    nom_z = bkw/2
    book_trans = np.array([nom_x, nom_y, nom_z])
    book_rot = R.from_euler('xyz', [np.pi/2, 0, 0]).as_matrix()
    book_tf = np.eye(4); book_tf[:-1, :-1] = book_rot; book_tf[:-1, -1] = book_trans
    nominal_book_unposed = trimesh.creation.box(book_dims)
    nominal_book = trimesh.creation.box(book_dims).apply_transform(book_tf)

    # scene = trimesh.Scene()
    # scene.add_geometry([base, top, back_wall, nominal_book] + list(shelves.values()) + list(side_walls.values()))
    # scene.show()

    # now create slots for all the rest of the books
    books_per_shelf = int(bw / bkt)
    total_books = int(books_per_shelf*n_shelves)

    book_trans_shelf_list = []
    book_trans_all_list = []
    for i in range(n_shelves):
        shelf_trans_list = []
        for j in range(books_per_shelf):
            x_trans = 0.0
            y_trans = (j*bkt + j*eps)  # offset from the nominal book in the corner
            y_trans = -1.0 * y_trans if start_right else y_trans
            z_trans = i*sh[2]

            # check if we're already past the max length with y_trans
            if start_right:
                if y_trans <= -1.0 * (bw - bkt): # (bw - 3*eps):
                    break
            else:
                if y_trans >= (bw - bkt): # (bw - 3*eps):
                    break

            trans = np.array([x_trans, y_trans, z_trans])  
            shelf_trans_list.append(trans)
            book_trans_all_list.append(trans)
        book_trans_shelf_list.append(shelf_trans_list)

    # available pose criteria
    # 1. if we switch from NO to YES, save the NO
    # 2. if we switch from YES to NO, save the NO
    # 3. if we start on a new shelf and we are a NO, save the NO
    # 4. if we end on a shelf and we are a NO, save the NO
    true_books_per_shelf = len(book_trans_shelf_list[0])
    available_poses = []
    book_meshes = []

    n_trans = 0
    last_yes = True
    last_no = False
    last_trans = None
    for i, trans in enumerate(book_trans_all_list):
        n_trans += 1
        # new shelf -- start fresh
        if n_trans == true_books_per_shelf:
            n_trans = 0
            last_yes = True
            last_no = False

        if np.random.random() > (1 - per_book_prob):
            book = nominal_book.copy().apply_translation(trans)
            book_meshes.append(book)
            # we picked -- was the last one a no? if yes, let's use the LAST one
            if last_no:
                avail_pose_mat = np.eye(4)
                avail_pose_mat[:-1, :-1] = book_rot
                avail_pose_mat[:-1, -1] = book_trans + last_trans
                available_pose = util.pose_stamped2list(util.pose_from_matrix(avail_pose_mat))
                available_poses.append(available_pose)
            last_yes = True
            last_no = False
        else:
            # we did not pick -- was the last one a yes? if yes, let's use THIS one
            if last_yes:
                avail_pose_mat = np.eye(4)
                avail_pose_mat[:-1, :-1] = book_rot
                avail_pose_mat[:-1, -1] = book_trans + trans
                available_pose = util.pose_stamped2list(util.pose_from_matrix(avail_pose_mat))
                available_poses.append(available_pose)
            last_yes = False
            last_no = True

        last_trans = trans

    # scene = trimesh.Scene()
    # scene.add_geometry([base, top, back_wall] + list(shelves.values()) + list(side_walls.values()) + book_meshes)
    # scene.show()

    # merge
    mesh_dict = {'base': base, 'back_wall': back_wall}
    for k, v in shelves.items():
        mesh_dict[k] = v
    for k, v in side_walls.items():
        mesh_dict[k] = v
    if top_bool:
        mesh_dict['top'] = top
    for i in range(len(book_meshes)):
        mesh_dict[f'book_{i}'] = book_meshes[i]
    full_bookshelf_mesh = trimesh.util.concatenate(list(mesh_dict.values()))

    if show:
        # scene.add_geometry(
        #     [base, back_wall] + 
        #     list(shelves.values()) + 
        #     list(side_walls.values())) 
        if mc_vis is not None:
            util.meshcat_trimesh_show(mc_vis, 'scene/full_bookshelf', full_bookshelf_mesh)
        # scene.show()
        # input('waiting for user to view in meshcat')

    return full_bookshelf_mesh, nominal_book_unposed, mesh_dict, available_poses


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
    per_book_prob = 0.7

    full_bookshelf_mesh, _ = make_bookshelf(
        bl=bl, bw=bw, bt=bt, n_shelves=n_shelves,
        bkl=bkl, bkw=bkw, bkt=bkt,
        bkl_range=0.0, bkw_range=0.0, bkt_range=0.0,
        wt=wt, top_bool=top_bool, per_book_prob=per_book_prob,
        show=True, mc_vis=mc_vis)

    full_bookshelf_mesh.show()
    from IPython import embed; embed()
