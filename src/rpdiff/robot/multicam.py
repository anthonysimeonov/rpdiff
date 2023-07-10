from yacs.config import CfgNode as CN
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet


class MultiCams:
    """
    Class for easily obtaining simulated camera image observations in pybullet
    """
    def __init__(self, cam_cfg, pb_client, n_cams=2):
        """
        Constructor, sets up base class and additional camera setup
        configuration parameters.

        Args:
            cam_cfg (rpdiff.utils.config_util.AttrDict): dict with dot
                set/get ability for keys
            pb_client (airobot.pb_util.BulletClient): Interface to pybullet
            n_cams (int): Number of cameras to put in the world
        """
        super(MultiCams, self).__init__()
        self.n_cams = n_cams
        self.cams = []
        self.cfg = cam_cfg
        self.pb_client = pb_client
        self.focus_pt = self.cfg.focus_pt
        for _ in range(n_cams):
            self.cams.append(RGBDCameraPybullet(cfgs=self._camera_cfgs(),
                                                pb_client=pb_client))

        self.cam_setup_cfg = {}
        # if isinstance(self.cfg.FOCUS_PT, list):
        if self.cfg.single_focus_pt:
            self.cam_setup_cfg['focus_pt'] = [self.cfg.focus_pt] * self.n_cams
        else:
            self.cam_setup_cfg['focus_pt'] = self.cfg.focus_pt_set[:self.n_cams]

        # yaw
        self.cam_setup_cfg['yaw'] = self.cfg.yaw_angles[:self.n_cams]

        # distance from focus pt
        # if isinstance(self.cfg.DISTANCE, list):
        if self.cfg.single_distance:
            self.cam_setup_cfg['dist'] = [self.cfg.distance] * self.n_cams
        else:
            self.cam_setup_cfg['dist'] = self.cfg.distance_set[:self.n_cams]

        # pitch angles
        # if isinstance(self.cfg.PITCH, list):
        if self.cfg.single_pitch:
            self.cam_setup_cfg['pitch'] = [self.cfg.pitch] * self.n_cams
        else:
            self.cam_setup_cfg['pitch'] = self.cfg.pitch_set[:self.n_cams]
        
        # set roll to 0
        self.cam_setup_cfg['roll'] = [0] * self.n_cams

        self._setup_cameras()

    def _camera_cfgs(self):
        """
        Returns a set of camera config parameters

        Returns:
            YACS CfgNode: Cam config params
        """
        _C = CN()
        _C.ZNEAR = 0.01
        _C.ZFAR = 10
        _C.WIDTH = 640
        _C.HEIGHT = 480
        _C.FOV = 60
        _ROOT_C = CN()
        _ROOT_C.CAM = CN()
        _ROOT_C.CAM.SIM = _C
        return _ROOT_C.clone()

    def _setup_cameras(self):
        """
        Function to set up multiple pybullet cameras in the simulated environment
        """
        for i, cam in enumerate(self.cams):
            cam.setup_camera(
                focus_pt=self.cam_setup_cfg['focus_pt'][i],
                dist=self.cam_setup_cfg['dist'][i],
                yaw=self.cam_setup_cfg['yaw'][i],
                pitch=self.cam_setup_cfg['pitch'][i],
                roll=self.cam_setup_cfg['roll'][i]
            )
