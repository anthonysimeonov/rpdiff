from yacs.config import CfgNode as CN

_C = CN()

_C.BASE_LENGTH_LOW_HIGH = [0.13, 0.35]
_C.BASE_WIDTH_LOW_HIGH = [0.1, 0.3]
_C.BASE_THICKNESS_LOW_HIGH = [0.001, 0.005]
_C.WALL_HEIGHT_LOW_HIGH = [0.04, 0.14]
_C.WALL_THICKNESS_LOW_HIGH = [0.0025, 0.005]
_C.WALL_THETA_LOW_HIGH = [0, 15]

def get_syn_container_default_cfg():
    return _C.clone()
