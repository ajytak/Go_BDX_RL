import numpy as np
import torch
import pickle
import os
from typing import Optional


class MotionLoader:
    """
    Simplified MotionLoader for demos that only contain:
    - joint positions (T, 1+N)
    - joint velocities (T, 1+N)
    where the first column is time.

    Frequency is fixed at 250 Hz.
    """

    def __init__(self, npz_file: str, device: torch.device):
        assert os.path.isfile(npz_file), f"Invalid file path: {npz_file}"

        self.device = device

        # load data
        data = np.load(npz_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.root_pos = torch.tensor(data["body_pos_w"][:,0,:], dtype=torch.float32, device=device)
        self.root_quat = torch.tensor(data["body_quat_w"][:,0,:], dtype=torch.float32, device=device)
        self.root_lin_vel_w = torch.tensor(data["body_lin_vel_w"][:,0,:], dtype=torch.float32, device=device)
        self.root_ang_vel_w = torch.tensor(data["body_ang_vel_w"][:,0,:], dtype=torch.float32, device=device)
        self.left_foot_pos = torch.tensor(data["left_toe_pos"], dtype=torch.float32, device=device)
        self.right_foot_pos = torch.tensor(data["right_toe_pos"], dtype=torch.float32, device=device)
        self.left_foot_vel = torch.tensor(data["left_toe_vel"], dtype=torch.float32, device=device)
        self.right_foot_vel = torch.tensor(data["right_toe_vel"], dtype=torch.float32, device=device)


        # Number of frames
        self.num_frames = self.joint_pos.shape[0]
        self.dt = 1.0 / self.fps
        self.duration = self.dt * (self.num_frames - 1)

        print(f"[MotionLoader] Loaded {npz_file}")
        print(f"Frames: {self.num_frames}, Duration: {self.duration:.3f}s, FPS={self.fps}")

    # -----------------------------------------------------------
    # Interpolation helpers
    # -----------------------------------------------------------

    def _interpolate(self, a, *, b=None, blend=None, start=None, end=None):
        """Linear interpolation between a[start] and a[end]."""
        if start is not None and end is not None:
            a0 = a[start]
            a1 = a[end]
            return self._interpolate(a0, b=a1, blend=blend)
        # expand blend dimensions
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _compute_frame_blend(self, times: np.ndarray):
        """Compute index_0, index_1, blend just like Isaac Lab."""
        phase = np.clip(times / self.duration, 0, 1)
        index_0 = (phase * (self.num_frames - 1)).round().astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = ((times - index_0 * self.dt) / self.dt).clip(0, 1)
        return index_0, index_1, blend

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    # -----------------------------------------------------------
    # Sampling
    # -----------------------------------------------------------

    def sample_times(self, num_samples):
        return self.duration * np.random.uniform(0.0, 1.0, num_samples)

    def sample(self, num_samples, times=None):
        """Returns joint_pos and joint_vel at random times."""
        if times is None:
            times = self.sample_times(num_samples)

        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.root_pos, blend=blend, start=index_0, end=index_1),
            self._slerp(self.root_quat, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.root_lin_vel_w, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.root_ang_vel_w, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.left_foot_pos,blend=blend, start=index_0, end=index_1),
            self._interpolate(self.right_foot_pos,blend=blend, start=index_0, end=index_1),
            self._interpolate(self.left_foot_vel,blend=blend, start=index_0, end=index_1),
            self._interpolate(self.right_foot_vel,blend=blend, start=index_0, end=index_1),
            self._interpolate(self.joint_pos, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.joint_vel, blend=blend, start=index_0, end=index_1),
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)