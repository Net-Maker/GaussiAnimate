"""MLP/GRU baselines from the author's VTO experiment, without cached models."""

import numpy as np
import torch
from torch import nn


class PoseMLP(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        layers = [nn.Linear(69, 512), nn.ReLU()]
        for _ in range(6):
            layers.extend([nn.Linear(512, 512), nn.ReLU(), nn.Dropout(0.1)])
        layers.append(nn.Linear(512, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, poses):
        return self.net(poses)


class PoseGRU(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.cell = nn.GRUCell(69, 512)
        self.output = nn.Linear(512, output_dim)

    def forward(self, poses, hidden=None):
        if hidden is None:
            hidden = poses.new_zeros(len(poses), 512)
        hidden = self.cell(poses, hidden)
        return self.output(hidden), hidden


def train_models(poses, transforms, frame_counts, *, device, steps=3000, seed=42):
    """Original Adam, normalized transform regression, and per-sequence GRU BPTT."""
    if steps < 1:
        raise ValueError("training steps must be positive")
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    poses = np.asarray(poses[:, 3:], dtype=np.float32)
    targets = np.asarray(transforms[:, :, :3, :], dtype=np.float32).reshape(len(poses), -1)
    stats = {"pose_mean": poses.mean(0), "pose_std": poses.std(0) + 1e-8,
             "transform_mean": targets.mean(0), "transform_std": targets.std(0) + 1e-8}
    x = torch.from_numpy((poses - stats["pose_mean"]) / stats["pose_std"]).to(device)
    y = torch.from_numpy((targets - stats["transform_mean"]) / stats["transform_std"]).to(device)
    offsets = np.cumsum([0] + list(frame_counts))
    if offsets[-1] != len(poses) or any(count < 1 for count in frame_counts):
        raise ValueError("frame_counts must partition the training frames")
    gru = PoseGRU(targets.shape[1]).to(device)
    optimizer = torch.optim.Adam(gru.parameters(), lr=1e-4)
    for _ in range(steps):
        index = rng.randint(len(frame_counts))
        start, stop = offsets[index:index + 2]
        hidden = None
        loss = 0
        for frame in range(start, stop):
            pred, hidden = gru(x[frame:frame + 1], hidden)
            loss = loss + ((pred - y[frame:frame + 1]) ** 2).mean()
        optimizer.zero_grad()
        (loss / (stop - start)).backward()
        optimizer.step()
    mlp = PoseMLP(targets.shape[1]).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    for _ in range(steps):
        indices = rng.choice(len(x), min(256, len(x)), replace=False)
        loss = ((mlp(x[indices]) - y[indices]) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return {"gru": gru.eval(), "mlp": mlp.eval()}, stats


def predict_transforms(model, poses, stats, method, device):
    values = np.asarray(poses[:, 3:], dtype=np.float32)
    values = (values - stats["pose_mean"]) / stats["pose_std"]
    values = torch.from_numpy(values).to(device)
    with torch.no_grad():
        if method == "gru":
            frames, hidden = [], None
            for frame in values:
                pred, hidden = model(frame[None], hidden)
                frames.append(pred)
            prediction = torch.cat(frames)
        else:
            prediction = model(values)
    prediction = prediction.cpu().numpy() * stats["transform_std"] + stats["transform_mean"]
    transforms = np.tile(np.eye(4, dtype=np.float32), (len(values), prediction.shape[1] // 12, 1, 1))
    transforms[:, :, :3, :] = prediction.reshape(len(values), -1, 3, 4)
    return transforms
