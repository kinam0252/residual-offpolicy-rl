import torch
from isaaclab.utils.math import quat_apply, quat_inv, quat_mul

def compute_object_state_in_hand_frame(
    object_pos_w: torch.Tensor, object_quat_w: torch.Tensor,
    object_lin_vel_w: torch.Tensor, object_ang_vel_w: torch.Tensor,
    hand_pos_w: torch.Tensor, hand_quat_w: torch.Tensor,
    hand_lin_vel_w: torch.Tensor, hand_ang_vel_w: torch.Tensor,
    debug: bool = False,
) -> dict:
    """
    Compute the object's pose and velocity relative to the hand frame.

    Args:
        All inputs are batched tensors of shape [N, ...].
        Quaternions are in (w, x, y, z) format.

    Returns:
        Dictionary with the following keys:
            - "pos": [N, 3] position in hand frame
            - "quat": [N, 4] orientation in hand frame (wxyz)
            - "lin_vel": [N, 3] linear velocity in hand frame
            - "ang_vel": [N, 3] angular velocity in hand frame
    """
    hand_quat_inv = quat_inv(hand_quat_w)

    pos_obj_in_hand = quat_apply(hand_quat_inv, object_pos_w - hand_pos_w)
    quat_obj_in_hand = quat_mul(hand_quat_inv, object_quat_w)
    lin_vel_in_hand = quat_apply(hand_quat_inv, object_lin_vel_w - hand_lin_vel_w)
    ang_vel_in_hand = quat_apply(hand_quat_inv, object_ang_vel_w - hand_ang_vel_w)

    if debug: import pdb; pdb.set_trace()
    return {
        "pos": pos_obj_in_hand,
        "quat": quat_obj_in_hand,
        "lin_vel": lin_vel_in_hand,
        "ang_vel": ang_vel_in_hand,
    }
