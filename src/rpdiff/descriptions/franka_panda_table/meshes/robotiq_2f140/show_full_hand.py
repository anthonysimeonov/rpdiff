from rrp_robot.utils.quick_viz_util import *
from rrp_robot.utils.visualize import Robotiq2F140Hand

hand = Robotiq2F140Hand()

hand.meshcat_show(mc_vis, 'scene/debug/full_hand')

from IPython import embed; embed()
