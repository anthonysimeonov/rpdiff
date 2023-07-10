import numpy as np


def parse_intrinsics(intrinsics):
    if len(intrinsics.shape) < 3:
        intrinsics = intrinsics[None, :, :]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    return fx, fy, cx, cy


def expand_as(x, y):
    if len(x.shape) == len(y.shape):
        return x

    for i in range(len(y.shape) - len(x.shape)):
        x = x.unsqueeze(-1)

    return x


def lift(x, y, z, intrinsics, homogeneous=False):
    '''
    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    if isinstance(x, int) or isinstance(y, int) or isinstance(z, int) or isinstance(z, float):
        x = np.asarray([x])
        y = np.asarray([y])
        z = np.asarray([z])

    if isinstance(x, list) or isinstance(y, list) or isinstance(z, list):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_lift = (x - expand_as(cx, x)) / expand_as(fx, x) * z
    y_lift = (y - expand_as(cy, y)) / expand_as(fy, y) * z

    if len(x_lift.shape) == 1:
        x_lift = x_lift[:, None]
        y_lift = y_lift[:, None]
    
    if len(z.shape) == 1:
        z = z[:, None]

    return np.concatenate((x_lift, y_lift, z), axis=-1)


def project(x, y, z, intrinsics):
    '''
    :param self:
    :param x: Shape (batch_size, num_points)
    :param y:
    :param z:
    :param intrinsics:
    :return:
    '''
    if isinstance(x, int) or isinstance(y, int) or isinstance(z, int) or isinstance(z, float):
        x = np.asarray([x])
        y = np.asarray([y])
        z = np.asarray([z])

    if isinstance(x, list) or isinstance(y, list) or isinstance(z, list):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
    fx, fy, cx, cy = parse_intrinsics(intrinsics)

    x_proj = expand_as(fx, x) * x / z + expand_as(cx, x)
    y_proj = expand_as(fy, y) * y / z + expand_as(cy, y)

    # return np.concatenate((x_proj, y_proj, z), axis=-1)
    if x_proj.ndim == 1:
        x_proj = x_proj.reshape(-1, 1)
    if y_proj.ndim == 1:
        y_proj = y_proj.reshape(-1, 1)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    # return np.concatenate((x_proj, y_proj, z), axis=1)
    return np.concatenate((x_proj, y_proj, z), axis=-1)