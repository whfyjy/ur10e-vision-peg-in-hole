"""Compute the eye-in-hand camera offset (in wrist_3_link) for a desired world view.

Isaac Lab applies CameraCfg.OffsetCfg as `parent_pose * offset`, converting the
offset rotation from `convention` to OpenGL first (camera/utils +
isaaclab.utils.math.convert_camera_frame_orientation_convention):

  - OpenGL (what USD cameras use): looks down -Z, +Y up, +X right
  - "ros":                         looks down +Z, -Y up, +X right
  and ros -> opengl is implemented as "negate the Y and Z columns of the
  rotation matrix", which is its own inverse.

Rather than hand-deriving a magic quaternion (the reference project's is
documented only as "computed to look at the peg tip" and targets a different
peg geometry), this computes it: give a world camera position and a world
look-at point, and it returns the (pos, rot) pair to put in OffsetCfg with
convention="ros", plus the angles at which the peg tip and the hole fall in the
frame so the framing can be judged before rendering anything.

Run:  python scripts/cam_offset.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kin  # noqa: E402

HOME = kin.HOME_JOINTS
FOCAL_LENGTH = 24.0        # cm, Isaac Lab PinholeCameraCfg default
H_APERTURE = 20.955        # cm, default
HFOV_DEG = 2 * np.degrees(np.arctan(H_APERTURE / (2 * FOCAL_LENGTH)))


def wrist_frame(q=HOME):
    """(pos, R) of wrist_3_link in world at joint vector q -- from the URDF chain."""
    m = np.eye(4)
    for xyz, rpy in kin.FIXED_PRE:
        m = m @ kin._tf(xyz, rpy)
    for i, ((xyz, rpy), a) in enumerate(zip(kin.REVOLUTE, q)):
        m = m @ kin._tf(xyz, rpy, a)
        if i == 5:  # wrist_3
            return m[:3, 3].copy(), m[:3, :3].copy()
    raise RuntimeError("chain is shorter than expected")


def look_at_rotation(cam_pos, target, world_up=(0.0, 0.0, 1.0)):
    """OpenGL-convention camera rotation matrix looking from cam_pos at target.

    OpenGL camera: -Z is forward, +Y is up, +X is right.
    """
    fwd = np.asarray(target, float) - np.asarray(cam_pos, float)
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(world_up, float)
    if abs(fwd @ up) > 0.999:            # looking straight up/down: pick another up
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    return np.column_stack([right, true_up, -fwd])   # x, y, z columns


def quat_from_matrix(R):
    """(w, x, y, z) from a rotation matrix (Shepperd's method)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def offset_from_world(cam_pos_w, look_at_w, q=HOME):
    """OffsetCfg (pos, rot, convention='ros') placing the camera at cam_pos_w."""
    w_pos, w_R = wrist_frame(q)
    R_cam = look_at_rotation(cam_pos_w, look_at_w)          # opengl
    R_off_gl = w_R.T @ R_cam
    # opengl -> ros: negate the Y and Z columns (the same operation both ways)
    R_off_ros = R_off_gl.copy()
    R_off_ros[:, 1] = -R_off_ros[:, 1]
    R_off_ros[:, 2] = -R_off_ros[:, 2]
    pos_off = w_R.T @ (np.asarray(cam_pos_w, float) - w_pos)
    return pos_off, quat_from_matrix(R_off_ros)


def frame_angles(cam_pos_w, look_at_w, points):
    """Angle of each world point away from the camera's optical axis, in degrees."""
    R = look_at_rotation(cam_pos_w, look_at_w)
    axis = -R[:, 2]                                   # opengl forward
    out = {}
    for name, p in points.items():
        v = np.asarray(p, float) - np.asarray(cam_pos_w, float)
        d = np.linalg.norm(v)
        out[name] = (np.degrees(np.arccos(np.clip(v / d @ axis, -1, 1))), d)
    return out


def main():
    tip = kin.tip_pos(HOME)
    hole = np.array([0.1637, 0.7054, 0.1807])
    w_pos, w_R = wrist_frame(HOME)
    wx, wy, wz = w_R[:, 0], w_R[:, 1], w_R[:, 2]

    print(f"wrist_3_link at HOME: pos={np.round(w_pos, 4)}")
    print(f"  axes in world: x={np.round(wx, 3)}  y={np.round(wy, 3)}  z={np.round(wz, 3)}")
    print(f"peg tip = {np.round(tip, 4)}   hole = {np.round(hole, 4)}")
    print(f"camera HFOV = {HFOV_DEG:.1f} deg (focal {FOCAL_LENGTH} cm, aperture {H_APERTURE} cm)")
    print(f"in the wrist frame the hole sits at {np.round(w_R.T @ (hole - tip), 4)} relative to the tip\n")

    # Candidate: sit beside the tool axis, slightly below the wrist, and look
    # down at a point between the peg tip and the hole so both are in frame.
    for lateral in (0.05, 0.07, 0.09):
        for along in (0.0, 0.03):
            cam = w_pos + (-lateral) * wx + along * wz
            tgt = 0.5 * (tip + hole) + np.array([0.0, 0.0, 0.05])
            pos_off, quat = offset_from_world(cam, tgt)
            ang = frame_angles(cam, tgt, {"tip": tip, "hole": hole})
            ok = all(a < HFOV_DEG / 2 * 0.9 for a, _ in ang.values())
            print(f"lateral={lateral:.2f} along={along:.2f} -> cam_world={np.round(cam, 4)}")
            print(f"    tip {ang['tip'][0]:6.2f} deg (d={ang['tip'][1]:.3f})   "
                  f"hole {ang['hole'][0]:6.2f} deg (d={ang['hole'][1]:.3f})   "
                  f"{'both in frame' if ok else 'OUT OF FRAME'}")
            print(f"    OffsetCfg(pos=({pos_off[0]:.5f}, {pos_off[1]:.5f}, {pos_off[2]:.5f}),")
            print(f"              rot=({quat[0]:.5f}, {quat[1]:.5f}, {quat[2]:.5f}, {quat[3]:.5f}), convention=\"ros\")")


if __name__ == "__main__":
    main()
