import os, os.path as osp


def get_rpdiff_src() -> str:
    return os.environ['RPDIFF_SOURCE_DIR']


def get_rpdiff_catkin_src() -> str:
    return os.environ['RPDIFF_CWS_SOURCE_DIR']


def get_rpdiff_config() -> str:
    return osp.join(get_rpdiff_src(), 'config')


def get_rpdiff_share() -> str:
    return osp.join(get_rpdiff_src(), 'share')


def get_rpdiff_data() -> str:
    return osp.join(get_rpdiff_src(), 'data')


def get_rpdiff_recon_data() -> str:
    return osp.join(get_rpdiff_src(), 'data_gen/data')


def get_rpdiff_eval_data() -> str:
    return osp.join(get_rpdiff_src(), 'eval_data')


def get_rpdiff_descriptions() -> str:
    return osp.join(get_rpdiff_src(), 'descriptions')


def get_rpdiff_obj_descriptions() -> str:
    return osp.join(get_rpdiff_descriptions(), 'objects')


def get_rpdiff_demo_obj_descriptions() -> str:
    return osp.join(get_rpdiff_descriptions(), 'demo_objects')


def get_rpdiff_assets() -> str:
    return osp.join(get_rpdiff_src(), 'assets')


def get_rpdiff_model_weights() -> str:
    return osp.join(get_rpdiff_src(), 'model_weights')


def get_train_config_dir() -> str:
    return osp.join(get_rpdiff_config(), 'train_cfgs')


def get_eval_config_dir() -> str:
    return osp.join(get_rpdiff_config(), 'full_eval_cfgs')


def get_demo_config_dir() -> str:
    return osp.join(get_rpdiff_config(), 'full_demo_cfgs')

