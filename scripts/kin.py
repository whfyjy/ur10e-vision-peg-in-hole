"""Forward kinematics and damped-least-squares IK for the genpih UR10e + peg.

Numbers are transcribed from assets/ur10e_peg.urdf (the same URDF the USD was
converted from), and were checked against genpih's own measured home pose:
at HOME the peg tip comes out at (0.1037, 0.7054, 0.5568) with the peg axis
exactly (0, 0, -1). Used by the scripted probe and the GUI demo, which need to
drive the peg to a Cartesian target without training anything.

Joint order is the URDF's: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2,
wrist_3. The robot's own column order may differ -- scripts must map through
robot.find_joints(...) rather than assuming.
"""

import numpy as np

PI = np.pi

# (xyz, rpy) of each frame, in chain order. First entry is the fixed
# base_link -> base_link_inertia rotation of pi about z (REP-103 vs the UR
# controller frame); the next six are revolute joints.
FIXED_PRE = [((0.0, 0.0, 0.0), (0.0, 0.0, PI))]
REVOLUTE = [
    ((0.0, 0.0, 0.1807), (0.0, 0.0, 0.0)),          # shoulder_pan
    ((0.0, 0.0, 0.0), (PI / 2, 0.0, 0.0)),          # shoulder_lift
    ((-0.6127, 0.0, 0.0), (0.0, 0.0, 0.0)),         # elbow
    ((-0.57155, 0.0, 0.17415), (0.0, 0.0, 0.0)),    # wrist_1
    ((0.0, -0.11985, 0.0), (PI / 2, 0.0, 0.0)),     # wrist_2
    ((0.0, 0.11655, 0.0), (PI / 2, PI, PI)),        # wrist_3
]
POST = [
    ((0.0, 0.0, 0.0), (0.0, -PI / 2, -PI / 2)),     # flange
    ((0.0, 0.0, 0.0), (PI / 2, 0.0, PI / 2)),       # tool0
    ((0.0, 0.0, 0.120), (0.0, 0.0, 0.0)),           # peg tip (tool0 + 0.120 z)
]

HOME_JOINTS = np.array([1.178097, -1.570796, 1.570796, -1.570796, -1.570796, 0.0])


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _tf(xyz, rpy, q=0.0):
    m = np.eye(4)
    m[:3, :3] = _rz(rpy[2]) @ _ry(rpy[1]) @ _rx(rpy[0]) @ _rz(q)
    m[:3, 3] = xyz
    return m


def fk(q):
    """4x4 world pose of the PEG TIP frame for joint vector q (URDF order)."""
    m = np.eye(4)
    for xyz, rpy in FIXED_PRE:
        m = m @ _tf(xyz, rpy)
    for (xyz, rpy), angle in zip(REVOLUTE, q):
        m = m @ _tf(xyz, rpy, angle)
    for xyz, rpy in POST:
        m = m @ _tf(xyz, rpy)
    return m


def tip_pos(q):
    return fk(q)[:3, 3]


def tilt_deg(q):
    """Angle between the peg axis and straight down, in degrees.

    The peg points straight down when its frame +z equals (0, 0, -1), so the
    tilt is acos(z . (0,0,-1)) -- NOT acos(-z . (0,0,-1)), which would report a
    perfectly vertical peg as 180 deg.
    """
    axis = fk(q)[:3, 2]
    return float(np.degrees(np.arccos(np.clip(axis @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0))))


def _residual(q, target):
    m = fk(q)
    return np.concatenate([(m[:3, 3] - target) * 10.0, m[:3, 2] - np.array([0.0, 0.0, -1.0])])


def ik_tip(target, q0=None):
    """Joint angles putting the peg tip at `target` with the peg pointing straight down.

    Returns (q, pos_err_m, tilt_deg). Uses scipy's least_squares on a 6-vector
    residual (position weighted 10x, axis alignment 1x). The arm has a redundant
    yaw about the vertical, so the solution found depends on q0; start from the
    previous pose to get a smooth, joint-space-continuous trajectory.
    """
    from scipy.optimize import least_squares

    if q0 is None:
        q0 = HOME_JOINTS
    res = least_squares(
        _residual, q0, args=(np.asarray(target, dtype=float),),
        xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=20000,
    )
    err = float(np.linalg.norm(tip_pos(res.x) - np.asarray(target, dtype=float)))
    return res.x, err, tilt_deg(res.x)


def lerp_tip(start, end, n):
    """n tip positions on the straight segment start -> end, endpoints included."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    return [start + (end - start) * t for t in np.linspace(0.0, 1.0, n)]
