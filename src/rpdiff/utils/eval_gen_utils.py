import numpy as np
import pybullet as p


def constraint_obj_world(obj_id: int, pos: np.ndarray, ori: np.ndarray) -> int:
    o_cid = p.createConstraint(
        obj_id,
        -1,
        -1,
        -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        pos, childFrameOrientation=ori,
    )
    return o_cid


def constraint_grasp_open(cid: int=None) -> None:
    if cid is not None:
        p.removeConstraint(cid)


def safeCollisionFilterPair(
        bodyUniqueIdA: int, bodyUniqueIdB: int, 
        linkIndexA: int, linkIndexB: int, 
        enableCollision: bool, *args, **kwargs) -> None:
    if bodyUniqueIdA is not None and bodyUniqueIdB is not None and linkIndexA is not None and linkIndexB is not None:
        p.setCollisionFilterPair(bodyUniqueIdA=bodyUniqueIdA, bodyUniqueIdB=bodyUniqueIdB, linkIndexA=linkIndexA, linkIndexB=linkIndexB, enableCollision=enableCollision)


def safeRemoveConstraint(cid: int) -> None:
    if cid is not None:
        p.removeConstraint(cid)

