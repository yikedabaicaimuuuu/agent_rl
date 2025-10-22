# agents/generation_agent.py
from typing import Optional, Dict, Any, List, Union
import os, time, sys
import numpy as np
import dspy
from langchain_openai import ChatOpenAI

# 允许从项目根导入 utils
import os as _os
sys.path.append(_os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))

# 工具
from utils.text_utils import safe_trim_prompt, trim_text_to_token_limit
from utils.trajectory_logger import TrajectoryLogger


NumberLike = Union[int, float, np.floating, np.generic, str, List[Any]]


def extract_scalar(val: NumberLike) -> float:
    """将 ragas/评估返回的各种类型稳健转成 float。"""
    if val is None:
        return 0.0
    # list: 取均值（更稳），如果需要取首元素改成 float(val[0])
    if isinstance(val, list):
        nums = []
        for v in val:
            try:
                nums.append(float(v))
            except Exception:
                pass
        return float(np.mean(nums)) if nums else 0.0
    # numpy / int / float / str
    try:
        return float(val)
    except Exception:
        # dict.value 兜底
        if hasattr(val, "value"):
            try:
                return float(getattr(val, "value"))
            except Exception:
                return 0.0
        return 0.0


def _get_doc_text(d) -> str:
    """从任意 doc 抽文本内容。"""
    if d is None:
        return ""
    txt = getattr(d, "page_content", None)
    if isinstance(txt, str):
        return txt
    if isinstance(d, dict):
        for k in ("page_content", "text", "content"):
            v = d.get(k)
            if isinstance(v, str):
                return v
    if isinstance(d, str):
        return d
    return ""


class GenerationAgent:
    def __init__(
        self,
        model_name: Optional[str] = None,
        llm: Optional[ChatOpenAI] = None,
        semantic_f1_metric=None,
        logger: Optional[TrajectoryLogger] = None,
        max_ctx_tokens: int = 1400,
        max_gen_tokens: int = 512
    ):
        """
        - 云端优先：OPENAI_API_KEY_REAL → 官方端点；否则回落本地 OPENAI_API_BASE。
        - 控制上下文/生成长度，适配轻量资源。
        """
        self.logger = logger
        self.semantic_f1_metric = semantic_f1_metric

        # 轻量模式可通过环境变量覆盖
        LIGHT_MODE = os.getenv("LIGHT_MODE", "0") == "1"
        if LIGHT_MODE:
            max_ctx_tokens = int(os.getenv("GEN_MAX_CTX_TOKENS", max_ctx_tokens))
            max_gen_tokens = int(os.getenv("GEN_MAX_GEN_TOKENS", max_gen_tokens))
        self.max_ctx_tokens = max_ctx_tokens
        self.max_gen_tokens = max_gen_tokens

        if llm is not None:
            self.llm = llm
        else:
            # 云端优先
            real_key = os.getenv("OPENAI_API_KEY_REAL")
            if real_key:
                base_url = "https://api.openai.com/v1"
                api_key = real_key
            else:
                base_url = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
                api_key = os.getenv("OPENAI_API_KEY", "sk-fake")

            default_model = "gpt-3.5-turbo"
            model_name = model_name or os.getenv("GEN_LLM_MODEL", default_model)

            self.llm = ChatOpenAI(
                model=model_name,                 # 新版用 model=
                base_url=base_url,
                api_key=api_key,
                temperature=0.0,
                max_tokens=self.max_gen_tokens,
                top_p=1.0,
                max_retries=int(os.getenv("LC_MAX_RETRIES", "1")),
                timeout=float(os.getenv("LC_TIMEOUT", "60")),
            )

        # === 更稳健的 LLM 标识（避免 None） ===  [新增]
        def _llm_ident(llm_obj):
            name = None
            base = None
            try:
                # 常见字段
                name = getattr(llm_obj, "model_name", None) or getattr(llm_obj, "model", None)
                base = getattr(llm_obj, "base_url", None)
                # LangChain 新版常把 client 藏里面
                if base is None and hasattr(llm_obj, "client"):
                    base = getattr(llm_obj.client, "base_url", None)
            except Exception:
                pass
            # 兜底：repr 截断
            if name is None:
                try:
                    name = repr(llm_obj)[:60]
                except Exception:
                    name = "<unknown>"
            if base is None:
                base = os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL") or "<unknown>"
            return name, base

        # 用更稳健的识别信息打印  [替换你原来的 print(...)]
        try:
            _name, _base = _llm_ident(self.llm)
            print(f"[GenerationAgent] model={_name} base={_base} "
                  f"ctx={self.max_ctx_tokens} gen={self.max_gen_tokens}")
        except Exception:
            print(f"[GenerationAgent] ctx={self.max_ctx_tokens} gen={self.max_gen_tokens}")

    # ---- 综合分（按需调整权重）----
    def _compute_combined_score(self, faith: float, relevancy: float, noise: float) -> float:
        # 防御：空值/越界
        f = float(faith or 0.0)
        r = float(relevancy or 0.0)
        n = float(noise if noise is not None else 1.0)
        n = min(max(n, 0.0), 1.0)
        return 0.65 * f + 0.25 * r + 0.10 * (1.0 - n)

    def _safe_semantic_f1(self, gold: str, pred: str) -> float:
        """
        优先调用 self.semantic_f1_metric（若可用），失败则回退到字符串级 token F1。
        支持多答案切分；拒答短语→0 分。
        """
        def _looks_like_refusal(t: str) -> bool:
            if not t: return False
            s = t.strip().lower()
            return any(kw in s for kw in [
                "not enough information", "cannot answer", "can't answer",
                "insufficient context", "unknown", "抱歉", "无法", "没有足够信息"
            ])

        def _normalize_text(s: str) -> str:
            import re
            t = (s or "").strip().lower()
            t = (t.replace("“","\"").replace("”","\"")
                .replace("’","'").replace("‘","'")
                .replace("—","-"))
            t = re.sub(r"[^\w\s]", " ", t)
            t = re.sub(r"\s+", " ", t).strip()
            return t

        def _split_multi(s: str) -> list:
            import re
            if not s: return []
            parts = re.split(r"\s*(?:,|;|/|\bor\b|或)\s*", str(s), flags=re.I)
            return [p for p in parts if p.strip()]

        def _token_f1(g: str, p: str) -> float:
            from collections import Counter
            g_norm, p_norm = _normalize_text(g), _normalize_text(p)
            if not g_norm or not p_norm:
                return 0.0
            g_toks, p_toks = g_norm.split(), p_norm.split()
            if not g_toks or not p_toks:
                return 0.0
            g_c, p_c = Counter(g_toks), Counter(p_toks)
            overlap = sum((g_c & p_c).values())
            if overlap <= 0:
                return 0.0
            precision = overlap / len(p_toks)
            recall    = overlap / len(g_toks)
            return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        def _best_token_f1(gold_s: str, pred_s: str) -> float:
            gold_list = _split_multi(gold_s) or [gold_s]
            pred_list = _split_multi(pred_s) or [pred_s]
            best = 0.0
            for g in gold_list:
                for p in pred_list:
                    g_norm, p_norm = _normalize_text(g), _normalize_text(p)
                    if g_norm and (g_norm in p_norm or p_norm in g_norm):
                        return 1.0
                    best = max(best, _token_f1(g, p))
            return best

        gold = "" if gold is None else str(gold).strip()
        pred = "" if pred is None else str(pred).strip()
        if not gold or not pred:
            return 0.0
        if _looks_like_refusal(pred):
            return 0.0

        metric = getattr(self, "semantic_f1_metric", None)
        if callable(metric):
            # 1) 直接字符串
            try:
                val = metric(gold, pred)
                try:
                    return float(val)
                except Exception:
                    pass
                if isinstance(val, dict):
                    for k in ("f1", "score", "semantic_f1"):
                        if k in val:
                            return float(val[k])
                for attr in ("score", "f1", "value"):
                    if hasattr(val, attr):
                        return float(getattr(val, attr))
            except Exception:
                pass
            # 2) Example 风格（惰性导入，环境无 dspy 也不会报错）
            try:
                import importlib
                dspy_mod = importlib.import_module("dspy")
                Example = getattr(dspy_mod, "Example", None)
            except Exception:
                Example = None
            if Example is not None:
                for fld in ("answer", "output", "response", "prediction"):
                    try:
                        e1, e2 = Example(**{fld: gold}), Example(**{fld: pred})
                        val = metric(e1, e2)
                        try:
                            return float(val)
                        except Exception:
                            if isinstance(val, dict):
                                for k in ("f1", "score", "semantic_f1"):
                                    if k in val:
                                        return float(val[k])
                            for attr in ("score", "f1", "value"):
                                if hasattr(val, attr):
                                    return float(getattr(val, attr))
                    except Exception:
                        pass

        # 3) 兜底：字符串 token F1
        return _best_token_f1(gold, pred)


    # ---- 上下文拼接 + 截断 ----
    def _trim_context(self, docs: List[Any], max_tokens: Optional[int] = None) -> str:
        max_tokens = max_tokens or self.max_ctx_tokens
        parts: List[str] = []
        for i, d in enumerate(docs, start=1):
            txt = _get_doc_text(d)
            if txt:
                parts.append(f"<Document {i}> {txt}")
        combined = "\n".join(parts)
        # 这里的 model 名只影响估算分词器选择；与你用的实际生成模型可以不同
        return trim_text_to_token_limit(combined, max_tokens=max_tokens, model="gpt-3.5-turbo")

    # ---- 生成提示 ----
    def _build_prompt(self, question: str, context: str, attempt: int) -> str:
        instructions = """
You are an AI assistant that answers questions strictly based on the retrieved documents.

Instructions:
1) Extract key facts from the retrieved context.
2) Summarize the points that directly answer the question.
3) Write the final answer explicitly and concisely, mirroring key entities in the question.
4) If the context is insufficient, say: "The provided context does not contain enough information to answer this question."
Do NOT mention the retrieval process or speculate beyond the context.
""".strip()
        if attempt > 0:
            instructions += f"\n(Note: This is attempt #{attempt + 1}. Improve faithfulness and concision.)"

        prompt = f"""
{instructions}

Question:
{question}

Retrieved Context:
{context}

Answer strictly based on the retrieved context above:
""".strip()

        return safe_trim_prompt(prompt, model="gpt-3.5-turbo")

    # ---- 主流程 ----

    def answer(
        self,
        question: str,
        docs: List[Any],
        evaluation_agent,
        ground_truth: Optional[str] = None,
        max_attempts: int = 2,
        prompt_id: str = "gen_v1"
    ) -> Dict[str, Any]:
        """
        多次尝试 → 评估 → 早停；在评测失效/默认值时也会刷新 best_answer，避免返回空串。
        """
        # ---- 上下文兜底 ----
        context = self._trim_context(docs) if docs else "<NO_RETRIEVED_CONTEXT>"

        # ==== A1. 先评估一次检索，拿 precision / recall_like（优先真 recall，否则 weak_recall） ====
        try:
            retrieval_res = evaluation_agent.evaluate_retrieval(
                user_query=question, retrieved_docs=docs, reference=ground_truth
            ) or {}
        except Exception:
            retrieval_res = {}

        retr_prec = float(retrieval_res.get("context_precision", 0.0) or 0.0)
        retr_rec  = retrieval_res.get("context_recall", None)
        weak_rec  = retrieval_res.get("weak_recall", None)
        recall_like = float((retr_rec if retr_rec is not None else (weak_rec if weak_rec is not None else 0.0)) or 0.0)

        # ---- 日志增强（E）----
        if self.logger:
            ctx_dbg = [ (getattr(d, "page_content", "") or "")[:120] for d in (docs or []) ][:2]
            self.logger.add_reason(
                f"[retr.dbg] k={len(docs or [])} prec={retr_prec:.2f} rec_like={recall_like:.2f} "
                f"ctx0={ctx_dbg[0] if ctx_dbg else ''}"
            )

        best_answer = ""
        best_combined_score = -1.0  # 初始很低，便于第一次有效答案覆盖
        best_metrics = {
            "faithfulness_score": 0.0,
            "response_relevancy": 0.0,
            "answer_relevancy": 0.0,   # 镜像，方便下游读取
            "noise_sensitivity": 1.0,
            "semantic_f1_score": 0.0
        }
        best_eval_result = None
        best_latency_ms = 0.0

        # 记录被用到的文档（hash 摘要由 logger 负责）
        if self.logger:
            for d in docs or []:
                snippet = getattr(d, "page_content", "")[:200]
                if snippet:
                    self.logger.add_observation(snippet, do_hash=True)

        # ==== C. 语义 F1 的安全计算（拒答→0；字段对称） ====



        for attempt in range(max_attempts):
            prompt = self._build_prompt(question, context, attempt)

            # ---- 生成（健壮抽取 + 非空兜底）----
            try:
                t0 = time.time()
                msg = self.llm.invoke(prompt)         # LangChain ChatOpenAI -> AIMessage
                gen_latency_ms = (time.time() - t0) * 1000.0

                answer_text = None
                # 常规
                if hasattr(msg, "content"):
                    answer_text = msg.content
                # 兼容其它字段
                if (not answer_text) and hasattr(msg, "message"):
                    answer_text = getattr(msg, "message")
                if (not answer_text) and hasattr(msg, "text"):
                    answer_text = getattr(msg, "text")
                # 兜底到字符串
                if not answer_text:
                    answer_text = str(msg)

                # 可能是 list/块结构
                if isinstance(answer_text, (list, tuple)):
                    parts = []
                    for p in answer_text:
                        if isinstance(p, str):
                            parts.append(p)
                        elif isinstance(p, dict):
                            if "text" in p and isinstance(p["text"], str):
                                parts.append(p["text"])
                            elif "content" in p and isinstance(p["content"], str):
                                parts.append(p["content"])
                    answer_text = " ".join(parts).strip()

                answer_text = (answer_text or "").strip()
                if not answer_text:
                    answer_text = "[NO_ANSWER_GENERATED]"
            except Exception as e:
                gen_latency_ms = 0.0
                answer_text = "[NO_ANSWER_GENERATED]"
                if self.logger:
                    self.logger.add_reason(f"[gen.error] {type(e).__name__}: {e}")

            # 生成日志
            if self.logger:
                self.logger.add_generation(attempt=attempt + 1, prompt_id=prompt_id, answer=answer_text[:800])
                self.logger.add_eval(gen_latency_ms=round(gen_latency_ms, 2), attempt=attempt + 1)

            # ---- 评估（传 reference；拿分+status）----
            try:
                eval_result = evaluation_agent.evaluate_generation(
                    user_query=question,
                    retrieved_docs=docs,
                    response=answer_text,
                    reference=ground_truth
                ) or {}
            except Exception as e:
                eval_result = {}
                if self.logger:
                    self.logger.add_reason(f"[eval.error] {type(e).__name__}: {e}")

            # 分数
            faithfulness_score = extract_scalar(eval_result.get("faithfulness", 0.0))
            relevancy_score_raw = eval_result.get("response_relevancy", None)
            if relevancy_score_raw is None:
                relevancy_score_raw = eval_result.get("answer_relevancy", 0.0)
            relevancy_score = extract_scalar(relevancy_score_raw)

            # status（默认missing，便于判断）
            faith_st = str(eval_result.get("faithfulness_status", "missing"))
            rel_st   = str(
                eval_result.get(
                    "response_relevancy_status",
                    eval_result.get("answer_relevancy_status", "missing")
                )
            )
            noise_st = "missing"

            # noise 兼容键名
            noise_sensitivity = None
            for k, v in eval_result.items():
                if "noise_sensitivity" in str(k):
                    noise_sensitivity = extract_scalar(v)
                    noise_st = str(
                        eval_result.get(f"{k}_status",
                        eval_result.get("noise_sensitivity_status", "ok"))
                    )
                    break
            if noise_sensitivity is None:
                noise_sensitivity, noise_st = 1.0, "missing"

            # ==== C. 语义 F1（修复版） ====
            # 可选：语义 F1
            if self.semantic_f1_metric and ground_truth:
                try:
                    semantic_f1_score = self._safe_semantic_f1(str(ground_truth), str(answer_text))
                except Exception as e:
                    print(f"⚠️ Semantic F1 failed: {e}")
                    semantic_f1_score = 0.0
            else:
                semantic_f1_score = 0.0


            # 评估日志（便于外部排查）
            if self.logger:
                self.logger.add_eval(
                    faith=faithfulness_score, response_relevancy=relevancy_score,
                    noise_sensitivity=noise_sensitivity, semantic_f1=semantic_f1_score,
                    faith_status=faith_st, relevancy_status=rel_st, noise_status=noise_st,
                    attempt=attempt + 1
                )
                # 调试打印一行（可留可去）
                print(f"🔧 eval_result -> faith={faithfulness_score:.4f}({faith_st}), "
                    f"rel={relevancy_score:.4f}({rel_st}), noise={noise_sensitivity:.4f}({noise_st}), "
                    f"ans[:60]={answer_text[:60]!r}")

            # ---- 早停 or 刷新最佳 ----
            combined_score = self._compute_combined_score(
                faithfulness_score, relevancy_score, noise_sensitivity,
            )

            # 检索差 → 降权
            if retr_prec < 0.2 or recall_like < 0.2:
                combined_score *= 0.6

            # “有用分数”：任一 status 为 ok 即视为有效评测
            has_any_valid = (faith_st == "ok") or (rel_st == "ok") or (noise_st == "ok")

            # ==== B. 早停（检索合格 + 模型分达标） ====
            early_stop = (
                has_any_valid and
                (retr_prec >= 0.5 and recall_like >= 0.5) and
                faithfulness_score >= 0.8 and
                relevancy_score   >= 0.6 and
                noise_sensitivity <= 0.4 and
                semantic_f1_score >= 0.7
            )
            if early_stop:
                if self.logger:
                    self.logger.set_final_answer(answer_text)
                return {
                    "answer": answer_text,
                    "faithfulness_score": faithfulness_score,
                    "response_relevancy": relevancy_score,
                    "answer_relevancy":  relevancy_score,   # 镜像
                    "noise_sensitivity": noise_sensitivity,
                    "semantic_f1_score": semantic_f1_score,
                    "cached_eval_result": eval_result,
                    "eval_result": eval_result,
                    "faith": faithfulness_score,
                    "semantic_f1": semantic_f1_score,
                    "latency_ms": gen_latency_ms
                }

            # 刷新最佳：① 有任何有效分且更优；② 或“尚未写入过 best_answer”（防空串）
            if (has_any_valid and combined_score > best_combined_score) or (not best_answer.strip()):
                if has_any_valid:
                    best_combined_score = combined_score
                best_answer = answer_text
                best_metrics = {
                    "faithfulness_score": faithfulness_score,
                    "response_relevancy": relevancy_score,
                    "answer_relevancy": relevancy_score,  # 镜像，方便下游读取
                    "noise_sensitivity": noise_sensitivity,
                    "semantic_f1_score": semantic_f1_score
                }
                best_eval_result = eval_result
                best_latency_ms = gen_latency_ms

        # ---- 达最大次数，返回最佳 ----
        if self.logger:
            self.logger.set_final_answer(best_answer)

        return {
            "answer": best_answer,
            "faithfulness_score": best_metrics.get("faithfulness_score", 0.0),
            "response_relevancy": best_metrics.get("response_relevancy", 0.0),
            "answer_relevancy":   best_metrics.get("answer_relevancy", best_metrics.get("response_relevancy", 0.0)),
            "noise_sensitivity": best_metrics.get("noise_sensitivity", 1.0),
            "semantic_f1_score": best_metrics.get("semantic_f1_score", 0.0),
            "cached_eval_result": best_eval_result,
            "eval_result": best_eval_result,
            "faith": best_metrics.get("faithfulness_score", 0.0),
            "semantic_f1": best_metrics.get("semantic_f1_score", 0.0),
            "latency_ms": best_latency_ms
        }
