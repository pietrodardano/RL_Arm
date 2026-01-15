import numpy as np
import mujoco as mj


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
    mat = np.zeros((3, 3))
    mj.mju_quat2Mat(mat, quat)
    return mat


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    quat = np.zeros(4)
    mj.mju_mat2Quat(quat, mat)
    return quat


def quat_mul(quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions."""
    result = np.zeros(4)
    mj.mju_mulQuat(result, quat1, quat2)
    return result


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    """Get quaternion conjugate."""
    conj = quat.copy()
    conj[:3] *= -1
    return conj


def mat_mul(mat1: np.ndarray, mat2: np.ndarray) -> np.ndarray:
    """Multiply two 3x3 rotation matrices."""
    result = np.zeros((3, 3))
    mj.mju_mulMatMat(result, mat1, mat2)
    return result


def get_object_pose(model: mj.MjModel, data: mj.MjData, obj_name: str) -> tuple:
    """Get object position and rotation matrix."""
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, obj_name)
    pos = data.xpos[body_id].copy()
    rot = data.xmat[body_id].reshape(3, 3).copy()
    return pos, rot


def get_all_objects(model: mj.MjModel, data: mj.MjData, 
                    obj_prefix: str = "obj") -> dict:
    """Get all objects matching prefix."""
    objects = {}
    for i in range(model.nbody):
        name = mj.id2name(model, mj.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith(obj_prefix):
            pos, rot = get_object_pose(model, data, name)
            objects[name] = {"pos": pos, "rot": rot}
    return objects


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return np.linalg.norm(p1 - p2)
