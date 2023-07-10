import os, os.path as osp

def obj2urdf(obj_filename, obj_name, save_dir=None, cwd=False, scaling=None):
    if scaling is None:
        scale = [1.0, 1.0, 1.0]
    else:
        assert isinstance(scaling, list), 'scaling must be a 3-element list'
        assert len(scaling) == 3, 'scaling must be a 3-element list'
        scale = scaling
    urdf_str = '''<?xml version="1.0" ?>
<robot name="%s">
    <link name="base_link">
        <inertial>
            <origin rpy="0 0 0" xyz="0 0 0"/>
            <mass value="0.1"/>
            <inertia ixx="1e-08" ixy="0" ixz="0" iyy="1e-08" iyz="0" izz="1e-08"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0" />
            <geometry>
                <mesh filename="%s" scale="%f %f %f"/>
            </geometry>         
        </visual>

        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="%s" scale="%f %f %f"/>
            </geometry>
        </collision>
    </link>
</robot>
    ''' % (obj_name, obj_filename, scale[0], scale[1], scale[2], obj_filename, scale[0], scale[1], scale[2])

    obj_fname = obj_filename.split('/')[-1] + '.urdf'
    if save_dir is None:
        if not cwd:
            save_dir = '/'.join(obj_filename.split('/')[:-1])
        else:
            save_dir = os.getcwd()
    
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    urdf_fname = osp.join(save_dir, obj_fname)
    # print(urdf_fname)
    open(urdf_fname, 'w').write(urdf_str)
    return urdf_str, urdf_fname
