# agents/RLRouterAgent.py
# 行为克隆 (BC) 路由器：从 runs/trajectories/*.jsonl 读取样本训练 MLP，
# 与 langgraph_rag.py 的 decision_state 字段一一对应。

import os, json, glob
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# =========================
# 路径 & 常量
# =========================
TRAJ_DIR = os.getenv("TRAJ_DIR", os.path.join(os.path.dirname(__file__), "..", "runs", "trajectories"))
POLICY_SAVE_PATH = os.path.join(os.path.dirname(__file__), "router_policy.pt")

# 与 langgraph_rag.py 的 decision_state 顺序保持一致
FEATURE_KEYS = [
    "context_precision",     # ctxP
    "context_recall",        # ctxR
    "faithfulness_score",
    "response_relevancy",
    "noise_sensitivity",
    "semantic_f1_score",
]

ACTION2IDX = {"end": 0, "requery": 1, "regenerate": 2}
IDX2ACTION = {v: k for k, v in ACTION2IDX.items()}


# =========================
# 工具函数
# =========================
def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def _coalesce_keys(rec: Dict[str, Any], keys: List[str], default=0.0) -> float:
    """从多个可能键名里取一个值（兼容不同写法/层级）"""
    for k in keys:
        if k in rec:
            return _safe_float(rec[k], default)
    return default

def _extract_action(rec: Dict[str, Any]) -> Optional[int]:
    """从轨迹记录中抽取动作；兼容 router_action / action_taken / action 等字段"""
    a = rec.get("router_action") or rec.get("action_taken") or rec.get("action")
    if not a:
        return None
    a = str(a).lower().strip()
    # 兼容“stop/end/finish”等表达
    if a in ("stop", "finish", "done"):
        a = "end"
    return ACTION2IDX.get(a, ACTION2IDX["end"])

def _minmax_norm(x: torch.Tensor, fmin: torch.Tensor, fmax: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(fmax - fmin, min=1e-6)
    return (x - fmin) / denom

def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


# =========================
# 数据集：从 JSONL 轨迹抽样
# =========================
class RouterTrajectoryDataset(Dataset):
    """
    从 runs/trajectories/*.jsonl 里逐条读取轨迹；
    每条样本使用“最新评估指标 + 最终路由动作”作为训练对。
    """
    def __init__(self, traj_dir: str = TRAJ_DIR):
        self.samples: List[Tuple[List[float], int]] = []
        self._load(traj_dir)

    def _load(self, traj_dir: str):
        if not os.path.isdir(traj_dir):
            raise FileNotFoundError(f"Trajectory directory not found: {traj_dir}")

        paths = sorted(glob.glob(os.path.join(traj_dir, "*.jsonl")))
        if not paths:
            raise FileNotFoundError(f"No JSONL files under: {traj_dir}")

        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    container = rec.get("eval", rec)

                    # 指标抽取（全部尽量落在 0..1，缺省给 0/1 合理值）
                    ctxP  = _coalesce_keys(container, ["ctxP", "context_precision"], default=0.0)
                    ctxR  = _coalesce_keys(container, ["ctxR", "context_recall"],    default=0.0)
                    faith = _coalesce_keys(container, ["faith", "faithfulness", "faithfulness_score"], default=0.0)
                    rel   = _coalesce_keys(container, ["response_relevancy", "answer_relevancy"], default=0.0)
                    noise = _coalesce_keys(container, ["noise_sensitivity", "noise"], default=1.0)
                    semf1 = _coalesce_keys(container, ["semantic_f1", "semantic_f1_score"], default=0.0)

                    feats = [ctxP, ctxR, faith, rel, noise, semf1]
                    action_idx = _extract_action(rec)
                    if action_idx is None:
                        continue

                    self.samples.append((feats, action_idx))

        if not self.samples:
            raise RuntimeError("No valid samples with router_action found; run the pipeline to generate trajectories first.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feats, action_idx = self.samples[idx]
        return torch.tensor(feats, dtype=torch.float32), torch.tensor(action_idx, dtype=torch.long)


# =========================
# MLP 策略网络
# =========================
class RouterPolicyNet(nn.Module):
    def __init__(self, input_dim: int = len(FEATURE_KEYS), hidden_dim: int = 32, num_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 训练（行为克隆）
# =========================
def train_router(
    traj_dir: str = TRAJ_DIR,
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    class_balance: bool = True,          # 类别不平衡时自动 reweight
    save_path: str = POLICY_SAVE_PATH,
    device: str = "cpu"
):
    dataset = RouterTrajectoryDataset(traj_dir)

    # 统计类别分布（用于日志）
    labels_all = torch.tensor([dataset[i][1] for i in range(len(dataset))], dtype=torch.long)
    class_counts = torch.bincount(labels_all, minlength=len(ACTION2IDX)).clamp(min=1)
    print(f"[train] class counts: " + ", ".join(f"{IDX2ACTION[i]}={int(c)}" for i, c in enumerate(class_counts)))

    # 计算特征 min/max（用于保存到 ckpt，推理时复用）
    all_feats = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0)  # [N, D]
    feat_min = all_feats.min(dim=0, keepdim=True).values
    feat_max = all_feats.max(dim=0, keepdim=True).values
    norm_feats = _minmax_norm(all_feats, feat_min, feat_max)

    # 用归一化后的特征重建 dataset
    class _NormedDS(Dataset):
        def __init__(self, feats: torch.Tensor, labels: torch.Tensor):
            self._feats = feats
            self._labels = labels
        def __len__(self): return self._feats.size(0)
        def __getitem__(self, idx): return self._feats[idx], self._labels[idx]

    ds = _NormedDS(norm_feats, labels_all)

    # 类别不平衡采样
    if class_balance:
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels_all]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(ds), replacement=True)
        loader = DataLoader(ds, batch_size=batch_size, sampler=sampler)
    else:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = RouterPolicyNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for feats, actions in loader:
            feats, actions = feats.to(device), actions.to(device)
            logits = model(feats)
            loss = criterion(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)
            # 训练准确率
            pred = torch.argmax(logits, dim=-1)
            total_correct += (pred == actions).sum().item()
            total_seen += actions.numel()

        avg_loss = total_loss / max(1, total_seen)
        acc = total_correct / max(1, total_seen)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feat_min": feat_min.cpu(),
            "feat_max": feat_max.cpu(),
            "feature_keys": FEATURE_KEYS,
            "action2idx": ACTION2IDX,
        },
        save_path
    )
    print(f"✅ Trained policy saved to {save_path}")


# =========================
# 推理封装
# =========================
class RLRouterAgent:
    def __init__(self, policy_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.policy = RouterPolicyNet().to(self.device)

        # 归一化参数（来自训练集）
        self.feat_min: Optional[torch.Tensor] = None  # [1, D]
        self.feat_max: Optional[torch.Tensor] = None  # [1, D]

        if policy_path and os.path.exists(policy_path):
            ckpt = torch.load(policy_path, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            self.policy.load_state_dict(state_dict, strict=False)

            # 读取训练时的归一化参数
            if "feat_min" in ckpt and "feat_max" in ckpt:
                self.feat_min = ckpt["feat_min"].to(self.device).float()
                self.feat_max = ckpt["feat_max"].to(self.device).float()
            else:
                print("⚠️ No feat_min/feat_max in checkpoint; will fall back to clamp(0..1).")

            print(f"Loaded router policy from {policy_path}")
        else:
            print("⚠️ No trained router policy found; using randomly initialized policy (will mostly output 'end').")

        self.policy.eval()

    def _featurize(self, state: Dict[str, float]) -> torch.Tensor:
        """按 FEATURE_KEYS 顺序取特征，并做与训练一致的归一化。"""
        vals: List[float] = []
        for k in FEATURE_KEYS:
            if k in ("context_precision", "context_recall"):
                vals.append(_safe_float(state.get(k, state.get({"context_precision": "ctxP", "context_recall": "ctxR"}[k], 0.0))))
            else:
                vals.append(_safe_float(state.get(k, 0.0)))
        x = torch.tensor(vals, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, D]

        if self.feat_min is not None and self.feat_max is not None:
            x = _minmax_norm(x, self.feat_min, self.feat_max)
        else:
            x = _clamp01(x)
        return x

    def decide(self, state: Dict[str, float], greedy: bool = True, temperature: float = 1.0) -> str:
        with torch.no_grad():
            x = self._featurize(state)
            logits = self.policy(x)

            if greedy or temperature <= 0:
                action_idx = torch.argmax(logits, dim=-1).item()
            else:
                # 软采样（可用于探索）
                probs = torch.softmax(logits / max(1e-6, float(temperature)), dim=-1).squeeze(0).cpu().numpy()
                action_idx = int(probs.argmax())

        return IDX2ACTION.get(action_idx, "end")


# =========================
# 命令行训练入口（可选）
# =========================
if __name__ == "__main__":
    # 直接：python agents/RLRouterAgent.py
    train_router()
