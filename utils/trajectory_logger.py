# utils/trajectory_logger.py
# 通用轨迹记录器：把一次问答的 plan/retrieval/generation/eval/router 动作记录到 JSONL

from __future__ import annotations
import json, os, time, hashlib, re
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "1.1"

def _now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def safe_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

# ---- 安全工具 ----
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
def _clean_text(s: str, max_len: int = 2000) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = _CONTROL_RE.sub(" ", s).strip()
    if max_len and len(s) > max_len:
        s = s[:max_len]
    return s

def _to_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        # list → 取均值（更稳）
        if isinstance(x, list):
            nums = []
            for v in x:
                try:
                    nums.append(float(v))
                except Exception:
                    pass
            return float(sum(nums)/len(nums)) if nums else default
        return float(x)
    except Exception:
        # 有些对象有 .value
        try:
            return float(getattr(x, "value"))
        except Exception:
            return default

# ---- 评估键名规范化映射 ----
_EVAL_KEY_MAP = {
    # faithfulness
    "faith": "faith",
    "faithfulness": "faith",
    "faithfulness_score": "faith",
    # relevancy
    "response_relevancy": "response_relevancy",
    "answer_relevancy": "response_relevancy",
    # noise
    "noise_sensitivity": "noise_sensitivity",
    "noise": "noise_sensitivity",
    # context precision / recall
    "context_precision": "context_precision",
    "ctxp": "context_precision",
    "context_recall": "context_recall",
    "ctxr": "context_recall",
    # semantic f1
    "semantic_f1": "semantic_f1",
    "semantic_f1_score": "semantic_f1",
}

def _normalize_eval_dict(d: Dict[str, Any]) -> Dict[str, float]:
    """把各种别名键合到标准键；数值化；不在映射表内的键会原样返回（可选保留）。"""
    out: Dict[str, float] = {}
    passthrough: Dict[str, float] = {}
    for k, v in (d or {}).items():
        k_norm = _EVAL_KEY_MAP.get(str(k).lower(), None)
        if k_norm:
            out[k_norm] = _to_float(v, 0.0)
        else:
            # 其他键：尽量数值化（如果需要，也可以选择丢弃）
            try:
                passthrough[k] = _to_float(v, 0.0)
            except Exception:
                pass
    # 只保留我们关心的 6 个标准键，其他非必要键不进入 out（避免干扰）
    return out

@dataclass
class TrajectoryRecord:
    qid: str
    query_raw: str
    started_at: str = field(default_factory=_now_iso)
    schema_version: str = SCHEMA_VERSION
    reason_steps: List[str] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)   # 建议只放摘要/哈希
    generations: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    eval: Dict[str, Any] = field(default_factory=dict)      # 规范化后的关键指标
    router_action: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)      # 预留：模型名、版本、超参等
    finished_at: Optional[str] = None

class TrajectoryLogger:
    def __init__(self, out_dir: str = "runs/trajectories"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.rec: Optional[TrajectoryRecord] = None

    # -------- 生命周期 --------
    def start(self, qid: str, query_raw: str, **meta):
        """开启一条新轨迹；可同时传 meta，例如 model/base_url 等。"""
        self.rec = TrajectoryRecord(qid=qid, query_raw=_clean_text(query_raw, 4000))
        if meta:
            self.rec.meta.update(meta)

    def commit(self, out_path: Optional[str] = None):
        assert self.rec is not None, "commit() called before start()"
        self.rec.finished_at = _now_iso()
        data = asdict(self.rec)
        if out_path is None:
            out_path = os.path.join(self.out_dir, f"{self.rec.qid}.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        self.rec = None

    # -------- 记录内容 --------
    def add_reason(self, text: str):
        assert self.rec is not None, "add_reason() called before start()"
        self.rec.reason_steps.append(_clean_text(text, 2000))

    def add_tool_call(self, **kwargs):
        """
        例：
          type='retrieval', query='...', topk=5, latency_ms=..., hits=[{'doc_id':..., 'score':...}, ...]
        """
        assert self.rec is not None, "add_tool_call() called before start()"
        # 去除过长/不可见字符
        safe_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, str):
                safe_kwargs[k] = _clean_text(v, 2000)
            else:
                safe_kwargs[k] = v
        self.rec.tools.append(safe_kwargs)

    def add_observation(self, text_or_summary: str, do_hash: bool = True, max_len: int = 500):
        """默认只写入 hash/摘要，避免泄漏原文；当 do_hash=False 时也会做清洗与限长。"""
        assert self.rec is not None, "add_observation() called before start()"
        if do_hash:
            self.rec.observations.append(f"[OBS:{safe_hash(text_or_summary)}]")
        else:
            self.rec.observations.append(_clean_text(text_or_summary, max_len))

    def add_generation(self, attempt: int, prompt_id: str, answer: str):
        assert self.rec is not None, "add_generation() called before start()"
        self.rec.generations.append({
            "attempt": int(attempt),
            "prompt_id": _clean_text(prompt_id, 128),
            "answer": _clean_text(answer, 2000)
        })

    def set_final_answer(self, answer: str):
        assert self.rec is not None, "set_final_answer() called before start()"
        self.rec.final_answer = _clean_text(answer, 4000)

    def add_eval(self, **metrics):
        """
        例：add_eval(faith=..., response_relevancy=..., noise_sensitivity=..., semantic_f1=..., ctxP=..., ctxR=...)
        - 自动将别名规范化为固定 6 个键；
        - 自动做数值化（float）；
        - 与已有 self.rec.eval 合并（同键覆盖）。
        """
        assert self.rec is not None, "add_eval() called before start()"
        norm = _normalize_eval_dict(metrics)
        if not self.rec.eval:
            self.rec.eval = {}
        # 合并更新
        self.rec.eval.update(norm)

    def set_router_action(self, action: str):
        assert self.rec is not None, "set_router_action() called before start()"
        # 统一小写，去首尾空白
        self.rec.router_action = _clean_text(str(action).lower().strip(), 64)
