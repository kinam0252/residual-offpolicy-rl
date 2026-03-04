import numpy as np
import copy
import sys

from isaaclab_tasks.utils.third_party.urdf_parser_py.urdf import URDF


class Transform:
    def __init__(self, data=None):
        if data is None:
            self.R = np.eye(3)
            self.T = np.zeros(3)
        else:
            flat_R = data[0:9]
            R = np.reshape(flat_R, (3, 3))
            self.R = R
            self.T = data[9:12]

    def to_array(self):
        res = np.zeros(12)
        flat_R = np.reshape(self.R, 9)
        res[0:9] = flat_R
        res[9:12] = self.T
        return res

    def inv(self):
        res = Transform()
        res.R = copy.deepcopy(self.R.T)
        res.T = -res.R.dot(self.T)
        return res

    def dot(self, trj):
        res = Transform()
        res.R = self.R.dot(trj.R)
        res.T = self.R.dot(trj.T) + self.T
        return res


def add4(mat):
    res = np.eye(4)
    res[0:3, 0:3] = mat
    return res


def reverse(offset_in):
    offset = copy.deepcopy(offset_in)
    offset.xyz = -np.array(offset.xyz)
    offset.rpy = -np.array(offset.rpy)
    return offset


def off2trans(offset):
    res = Transform()
    res.T = np.array(offset.xyz)
    cr = np.cos(offset.rpy[0])
    cp = np.cos(offset.rpy[1])
    cy = np.cos(offset.rpy[2])
    sr = np.sin(offset.rpy[0])
    sp = np.sin(offset.rpy[1])
    sy = np.sin(offset.rpy[2])
    res.T = np.array(offset.xyz)
    res.R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return res


def rev2trans(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
    res = Transform()
    res.R = np.array(
        [
            [
                c + axis[0] * axis[0] * (1 - c),
                axis[0] * axis[1] * (1 - c) - axis[2] * s,
                axis[0] * axis[2] * (1 - c) + axis[1] * s,
            ],
            [
                axis[1] * axis[0] * (1 - c) + axis[2] * s,
                c + axis[1] * axis[1] * (1 - c),
                axis[1] * axis[2] * (1 - c) - axis[0] * s,
            ],
            [
                axis[2] * axis[0] * (1 - c) - axis[1] * s,
                axis[2] * axis[1] * (1 - c) + axis[0] * s,
                c + axis[2] * axis[2] * (1 - c),
            ],
        ]
    )
    return res


def chainname2trans(robot, chain_name, joints, fixed_excluded, get_com=False):
    # search
    chain = [None for i in range(len(chain_name))]
    for l in robot.links:
        if l.name in chain_name:
            print(
                "!!!!!!!!!!!!!!!!please use get_chain with links=False option! aborting!!!!!!!!!!!!!!!!"
            )
            sys.exit(0)
    for j in robot.joints:
        if j.name in chain_name:
            idx = chain_name.index(j.name)
            chain[idx] = j

    M = Transform()
    # hack to match results for pybullet
    if get_com:
        com_root = robot.link_map[robot.get_root()].inertial
        M = M.dot(off2trans(reverse(com_root.origin)))

    j = 0
    for i in range(len(chain)):
        # joint
        if chain[i].type == "revolute" or chain[i].type == "continuous":
            if chain[i].mimic is None:
                M = M.dot(
                    off2trans(chain[i].origin).dot(rev2trans(chain[i].axis, joints[j]))
                )
                j += 1
            else:
                # assume
                ang = joints[j - 1] * chain[i].mimic.multiplier + chain[i].mimic.offset
                M = M.dot(off2trans(chain[i].origin).dot(rev2trans(chain[i].axis, ang)))
        elif chain[i].type == "fixed":
            M = M.dot(off2trans(chain[i].origin))
            if not fixed_excluded:
                j += 1

    # hack to match results for pybullet
    if get_com:
        com_end = robot.link_map[chain[-1].child].inertial
        # print(com_end)
        M = M.dot(off2trans(com_end.origin))
    return M


if __name__ == "__main__":
    urdf_file = "/sim/tm_robot_interface/NextageAOpen.urdf"
    robot = URDF.from_xml_file(urdf_file)

    chain_name = robot.get_chain("WAIST", "rh_forearm")
    print(chain_name)
    joints = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2]
    r2h = chainname2trans(robot, chain_name, joints, fixed_excluded=True)

    print(r2h.R)
    print(r2h.T)
