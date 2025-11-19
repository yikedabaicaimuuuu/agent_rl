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
    """从轨迹记录中抽取动作；兼容 router_action / action_taken / action 等字段，并做标准化映射"""
    a = rec.get("router_action") or rec.get("action_taken") or rec.get("action")
    if not a:
        return None
    a = str(a).lower().strip()

    # 统一别名/细粒度到 3 类
    if a in ("stop", "finish", "done", "end", "finalize"):
        a_std = "end"
    elif a in ("retry_retrieval", "requery", "retry", "retrieve_again"):
        a_std = "requery"
    elif a in ("regenerate", "regenerate_answer", "regenerate_answer_lower_temp", "regen", "rewrite_answer"):
        a_std = "regenerate"
    else:
        # 未知动作全部归并为 end
        a_std = "end"

    return ACTION2IDX.get(a_std, ACTION2IDX["end"])


# >>> CHANGE: 增加教师规则（用于自动打标签 & 推理回退）
def _teacher_rule_action(metrics_like: Dict[str, float]) -> str:
    """
    输入可以是 rec 或 rec['eval'] 或你组装的 metrics。
    只依赖 6 个特征：ctxP, ctxR, faith, rel, noise, semf1
    """
    def f(*names, default=0.0):
        for n in names:
            if n in metrics_like:
                return _safe_float(metrics_like[n], default)
        return default

    ctxR  = f("context_recall", "ctxR", default=0.0)
    rel   = f("response_relevancy", "answer_relevancy", default=0.0)
    faith = f("faithfulness_score", "faith", "faithfulness", default=0.0)
    noise = f("noise_sensitivity", "noise", default=1.0)
    semf1 = f("semantic_f1_score", "semantic_f1", default=0.0)

    # 规则与我们前面讨论保持一致（你可以按需微调）：
    if ctxR <= 1e-6:
        return "requery"
    if rel < 0.40:
        return "requery"            # 你的三动作集合里没有“改写后重试”，这里统一归到 requery
    if (faith < 0.55) or (noise > 0.60):
        return "regenerate"
    if semf1 >= 0.70:
        return "end"
    return "regenerate"

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

                    # 1) 轨迹里已有动作 → 用它
                    action_idx = _extract_action(rec)
                    # >>> CHANGE: 2) 否则自动用教师规则打标签（不再丢样本）
                    if action_idx is None:
                        a = _teacher_rule_action(container)
                        action_idx = ACTION2IDX.get(a, ACTION2IDX["end"])

                    self.samples.append((feats, action_idx))

        if not self.samples:
            raise RuntimeError("No valid samples found; run the pipeline to generate trajectories first.")

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
    def __init__(self, policy_path: Optional[str] = None, device: str = "cpu", logger=None):
        self.device = device
        self.logger = logger
        self.policy = RouterPolicyNet().to(self.device)

        self.feat_min: Optional[torch.Tensor] = None  # [1, D]
        self.feat_max: Optional[torch.Tensor] = None  # [1, D]
        self.policy_path: Optional[str] = policy_path
        self._has_policy: bool = False                 # ✅ 新增：记录是否已成功加载

        # ==== 新增：用环境变量控制模式 ====
        # ROUTER_MODE 可选： "off" / "teacher" / "bc"
        self.mode = os.getenv("ROUTER_MODE", "bc").lower()
        if self.mode not in ("off", "teacher", "bc"):
            self.mode = "bc"

        # ---- off 模式：router 存在，但永远直接 end ----
        if self.mode == "off":
            print("🧠 RLRouterAgent in OFF mode (ROUTER_MODE=off): always choose 'end'.")
            self.policy.eval()
            return

        # ---- teacher 模式：完全不用 BC policy，只走教师规则 ----
        if self.mode == "teacher":
            print("🧠 RLRouterAgent in TEACHER-RULE mode (ROUTER_MODE=teacher); ignore policy file.")
            self.policy.eval()
            return

        # ---- bc 模式：正常加载 policy, 失败则回退 teacher rule----
        if policy_path and os.path.exists(policy_path):
            ckpt = torch.load(policy_path, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)
            self.policy.load_state_dict(state_dict, strict=False)

            if "feat_min" in ckpt and "feat_max" in ckpt:
                self.feat_min = ckpt["feat_min"].to(self.device).float()
                self.feat_max = ckpt["feat_max"].to(self.device).float()
            else:
                print("⚠️ No feat_min/feat_max in checkpoint; will fall back to clamp(0..1).")

            print(f"Loaded router policy from {policy_path}")
            self._has_policy = True                     # ✅ 成功加载
        else:
            # 不存在或未提供 policy_path
            if policy_path:
                print(f"⚠️ Router policy not found at: {policy_path}; will fallback to teacher rules.")
            else:
                print("⚠️ No trained router policy found; will fallback to teacher rules.")
        self.policy.eval()

    def _featurize(self, state: Dict[str, float]) -> torch.Tensor:
        """按 FEATURE_KEYS 顺序取特征，并做与训练一致的归一化。"""
        vals: List[float] = []
        # 兼容 ctxP/ctxR 命名
        cp = state.get("context_precision", state.get("ctxP", 0.0))
        cr = state.get("context_recall", state.get("ctxR", 0.0))
        vals.append(_safe_float(cp))
        vals.append(_safe_float(cr))
        vals.append(_safe_float(state.get("faithfulness_score", state.get("faith", 0.0))))
        vals.append(_safe_float(state.get("response_relevancy", state.get("answer_relevancy", 0.0))))
        vals.append(_safe_float(state.get("noise_sensitivity", state.get("noise", 1.0))))
        vals.append(_safe_float(state.get("semantic_f1_score", state.get("semantic_f1", 0.0))))

        x = torch.tensor(vals, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, D]
        if self.feat_min is not None and self.feat_max is not None:
            x = _minmax_norm(x, self.feat_min, self.feat_max)
        else:
            x = _clamp01(x)
        return x

    def decide(self, state: Dict[str, float], greedy: bool = True, temperature: float = 1.0) -> str:
        # 如果没模型，直接教师规则；这样 3 个 case 也能完整走通
        # ✅ 用加载结果判断，而非硬读全局常量路径
        # 1) off 模式：永远直接 end，相当于“有 graph 但 router 不干预”
        if getattr(self, "mode", "bc") == "off":
            action = "end"
            print("[router.decide] ROUTER_MODE=off → always 'end'")
            if self.logger:
                self.logger.set_router_action(action)
            return action

        # 2) teacher 模式 或 没有成功加载 policy：用教师规则
        if self.mode == "teacher" or not getattr(self, "_has_policy", False):
            if self.mode == "teacher":
                print("[router.decide] ROUTER_MODE=teacher → teacher_rule")
            else:
                print("[router.decide] no loaded policy → fallback teacher_rule")
            a = _teacher_rule_action(state)
            if self.logger:
                self.logger.set_router_action(a)
            return a

        # 3) bc 模式 + 已有 policy：用 MLP policy
        with torch.no_grad():
            x = self._featurize(state)
            logits = self.policy(x)

            if greedy or temperature <= 0:
                action_idx = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits / max(1e-6, float(temperature)), dim=-1).squeeze(0).cpu().numpy()
                action_idx = int(probs.argmax())

        a = IDX2ACTION.get(action_idx, "end")
        print(f"[router.decide] ROUTER_MODE=bc + BC POLICY action={a} (idx={action_idx})")
        if self.logger:
            self.logger.set_router_action(a)
        return a


# =========================
# 命令行训练入口（可选）
# =========================
if __name__ == "__main__":
    # 直接：python agents/RLRouterAgent.py
    train_router()
