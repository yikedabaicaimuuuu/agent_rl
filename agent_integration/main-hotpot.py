# main-hotpot.py

import os
from tqdm import tqdm
import numpy as np
from decimal import Decimal

# ====== Agents ======
from agents.reasoning_agent import ReasoningAgent
from agents.retrieval_agent import RetrievalAgent
from agents.evaluation_agent import EvaluationAgent
from agents.generation_agent import GenerationAgent

# ====== run_rag_pipeline 导入（优先根目录，其次 agents/）======
try:
    from langgraph_rag import run_rag_pipeline  # 项目根目录
except Exception:
    from agents.langgraph_rag import run_rag_pipeline  # 兼容旧路径

# ====== Vector & Embeddings ======
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# ====== LLMs / DSPy ======
import dspy
from dspy.evaluate import SemanticF1
from langchain_openai import ChatOpenAI


# =========================
# 基础环境（默认本地 llama.cpp 兼容端点）
# =========================
os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")  # llama.cpp 不校验，但需要占位
os.environ.setdefault("EVAL_MAX_TOKENS", "1024")     # 原来 256 太小，建议 1024~2048
os.environ.setdefault("DSPY_MAX_TOKENS", "384")      # dspy 的也略增一点（避免 Reasoning 截断）
os.environ.setdefault("GEN_MAX_GEN_TOKENS", "1024")   # 生成 LLM 的也略增一点

# 控制项
USE_LOCAL_EMB = os.getenv("USE_LOCAL_EMB", "0") == "1"          # 本地嵌入（需重建索引）
USE_ROUTER    = os.getenv("USE_ROUTER", "0") == "1"             # 是否启用 Router/LangGraph
TESTSET_SIZE  = int(os.getenv("TESTSET_SIZE", "5"))             # 测试样本数
EVAL_MODE     = os.getenv("EVAL_MODE", "hybrid")                # strict / lenient / hybrid

# api keys/base
OPENAI_API_BASE      = os.getenv("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY       = os.getenv("OPENAI_API_KEY", "EMPTY")
OPENAI_API_KEY_REAL  = os.getenv("OPENAI_API_KEY_REAL")  # 真实 key（若用云端）


# 🔧 标量抽取（兼容 list/np/Decimal）
def extract_scalar(val):
    import numpy as _np
    if val is None:
        return 0.0
    # list / tuple / np array 取平均或第一个（你喜欢哪个就固定一个策略）
    if isinstance(val, (list, tuple)):
        try:
            xs = [float(x) for x in val if x is not None]
            return float(sum(xs) / len(xs)) if xs else 0.0
        except Exception:
            return 0.0
    # numpy 标量
    if isinstance(val, (_np.floating, _np.integer, _np.generic)):
        return float(val)
    # 普通数字
    if isinstance(val, (int, float)):
        return float(val)
    # 可转 float 的字符串
    try:
        return float(val)
    except Exception:
        # 兼容 ragas MetricResult 对象：有 .score / .value
        s = getattr(val, "score", None)
        if s is not None:
            try: return float(s)
            except: pass
        v = getattr(val, "value", None)
        if v is not None:
            try: return float(v)
            except: pass
        # 兼容 dict
        if isinstance(val, dict):
            for k in ("score", "value", "mean", "avg"):
                if k in val:
                    try: return float(val[k])
                    except: pass
        return 0.0



def _final_metric(result: dict, name: str, default: float = 0.0) -> float:
    """
    优先取 result['metrics'][name]（通常是重试后最终值），
    没有再回退到顶层 result[name]，最后给默认值。
    """
    m = (result.get("metrics") or {}).get(name, None)
    if m is None:
        m = result.get(name, None)
    return extract_scalar(m if m is not None else default)

def is_success(result: dict) -> bool:
    """统一判断是否通过（容错 ragas 指标缺失/解析失败）"""
    mode   = EVAL_MODE.lower()

    faith  = extract_scalar(result.get("faithfulness_score", 0.0))
    rel    = extract_scalar(result.get("response_relevancy", 0.0))
    noise  = extract_scalar(result.get("noise_sensitivity", 1.0))
    sem_f1 = extract_scalar(result.get("semantic_f1_score", result.get("semantic_f1", 0.0)))
    # 用最终值（可能是检索重试后的）
    recall = _final_metric(result, "context_recall", 0.0)

    # 读取状态（若 evaluate_generation 返回了这些字段）
    ev           = result.get("eval_result") or {}
    faith_st     = str(ev.get("faithfulness_status", "ok"))
    rel_st       = str(ev.get("response_relevancy_status", "ok"))
    noise_st     = str(ev.get("noise_sensitivity_status", "ok"))
    any_unreliable = (faith_st != "ok") or (rel_st != "ok") or (noise_st != "ok")

    # 规则
    ruleA = (faith >= 0.7 and rel >= 0.7 and noise <= 0.4 and sem_f1 >= 0.7)
    ruleB = (sem_f1 >= 0.85 and recall >= 0.7)  # 回退：答案对+召回够

    if mode == "strict":
        passed = ruleA
    elif mode == "lenient":
        passed = (sem_f1 >= 0.75 and recall >= 0.7)
    else:  # hybrid
        passed = (ruleA or ruleB) if not any_unreliable else ruleB

    # 可选调试打印：确保你看见用来判定的“最终 recall”
    print(f"🧪 pass_check | ruleA={ruleA} ruleB={ruleB} any_unreliable={any_unreliable} | "
          f"F1={sem_f1:.2f} faith={faith:.2f} rel={rel:.2f} noise={noise:.2f} ctxR(final)={recall:.2f} ({mode})")
    return passed



def main():
    # ------------------------
    # (1) 配置 DSPy 使用本地/云端（优先真实 key；否则退到本地）
    # ------------------------
    dspy.configure(
        lm=dspy.LM(
            model=os.getenv("DSPY_MODEL", "gpt-3.5-turbo"),
            api_base=OPENAI_API_BASE,
            api_key=OPENAI_API_KEY_REAL or OPENAI_API_KEY or "EMPTY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(os.getenv("DSPY_MAX_TOKENS", "256")),
            timeout=int(os.getenv("DSPY_TIMEOUT", "30")),
        )
    )

    print(f"🧭 USE_ROUTER={USE_ROUTER} | USE_LOCAL_EMB={USE_LOCAL_EMB} | EVAL_MODE={EVAL_MODE}")

    # ------------------------
    # (2) 构建评估/生成用的 LLM
    # ------------------------
    # 生成 LLM：按你的需求，默认走本地端点（也可改成云端）
    gen_llm = ChatOpenAI(
        model=os.getenv("GEN_LLM_MODEL", "gpt-3.5-turbo"),
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY or "EMPTY",
        temperature=0.0,
        max_tokens=int(os.getenv("GEN_MAX_GEN_TOKENS", "256")),
        timeout=float(os.getenv("LC_TIMEOUT", "60")),
    )

    # 评估 LLM：若有真实 key，则走“官方云端”更稳定；否则退到本地端点
    if OPENAI_API_KEY_REAL:
        eval_llm = ChatOpenAI(
            model=os.getenv("EVAL_LLM_MODEL", "gpt-3.5-turbo"),
            base_url="https://api.openai.com/v1",
            api_key=OPENAI_API_KEY_REAL,
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(os.getenv("EVAL_MAX_TOKENS", "256")),
            timeout=float(os.getenv("EVAL_TIMEOUT", "60")),
            max_retries=int(os.getenv("EVAL_MAX_RETRIES", "0")),
        )
        print("🔎 Evaluation LLM: using OpenAI cloud endpoint.")
    else:
        eval_llm = ChatOpenAI(
            model=os.getenv("EVAL_LLM_MODEL", "gpt-3.5-turbo"),
            base_url=OPENAI_API_BASE,
            api_key=OPENAI_API_KEY or "EMPTY",
            temperature=0.0,
            top_p=1.0,
            max_tokens=int(os.getenv("EVAL_MAX_TOKENS", "1024")),
            timeout=float(os.getenv("EVAL_TIMEOUT", "90")),
            max_retries=int(os.getenv("EVAL_MAX_RETRIES", "0")),
        )
        print("🔎 Evaluation LLM: no real key found, falling back to local endpoint.")

    # ------------------------
    # (3) Embeddings & FAISS
    # ------------------------
    if USE_LOCAL_EMB:
        # ⚠️ 必须用该模型**重建**向量库，否则会维度不匹配！
        print("🧠 Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2, dim=384)")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_path = os.getenv("FAISS_PATH_LOCAL", "vectorstore-hotpot/hf-miniLM-faiss")
    else:
        print("🧠 Embeddings: OpenAI (text-embedding-ada-002, dim=1536) — must match your existing FAISS index.")
        embeddings = OpenAIEmbeddings(
            model=os.getenv("EMB_MODEL", "text-embedding-ada-002"),
            api_key=OPENAI_API_KEY_REAL or os.getenv("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
        )
        vectorstore_path = os.getenv("FAISS_PATH_OPENAI", "vectorstore-hotpot/hotpotqa_faiss")

    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # ------------------------
    # (4) 初始化各 Agent
    # ------------------------
    semantic_f1_metric = SemanticF1(decompositional=True)

    reasoning_agent = ReasoningAgent()                 # 内部已配置 dspy
    evaluation_agent = EvaluationAgent(llm=eval_llm)   # 评估走 eval_llm（云端优先）
    retrieval_agent = RetrievalAgent(
        vectorstore=vectorstore,
        evaluation_agent=evaluation_agent,
        top_k=int(os.getenv("RETR_TOP_K", "3"))
    )
    generation_agent = GenerationAgent(
        llm=gen_llm,
        semantic_f1_metric=semantic_f1_metric
    )

    # ------------------------
    # (5) 取测试集并循环评测
    # ------------------------
    # ReasoningAgent.load_dataset() 在 __init__ 里已跑；数据集路径请保证存在
    total = min(TESTSET_SIZE, len(reasoning_agent.testset))
    subset_testset = reasoning_agent.testset[:total]

    print("\n🧪 Running test set evaluation...")
    correct = 0
    faithfulness_scores = []
    semantic_f1_scores = []
    failed_cases = []

    for i, example in enumerate(tqdm(subset_testset, desc="Evaluating test set")):
        question = example.question
        ground_truth = example.response
        print(f"\n🔍 Test {i+1}/{total}\nQ: {question}")
        print(f"📖 Ground Truth: {ground_truth}")  # 新增这一行

        result = run_rag_pipeline(
            question=question,
            retrieval_agent=retrieval_agent,
            reasoning_agent=reasoning_agent,
            generation_agent=generation_agent,
            evaluation_agent=evaluation_agent,
            reference=ground_truth,   # 传入 GT 以便 ctxR / semanticF1
            visualize=False,
            use_router=USE_ROUTER
        )

        ctxR_used = _final_metric(result, "context_recall", 0.0)
        ctxP_used = _final_metric(result, "context_precision", 0.0)
        print(f"🔎 Used-for-pass (final): ctxR={ctxR_used:.4f} ctxP={ctxP_used:.4f}")

        predicted_answer   = result.get("answer", "")
        faithfulness_score = extract_scalar(result.get("faithfulness_score", 0.0))
        relevancy_score    = extract_scalar(result.get("response_relevancy", 0.0))
        noise_score        = extract_scalar(result.get("noise_sensitivity", 1.0))
        semantic_f1_score  = extract_scalar(result.get("semantic_f1_score", 0.0))

        faithfulness_scores.append(faithfulness_score)
        semantic_f1_scores.append(semantic_f1_score)

        if is_success(result):
            correct += 1
        else:
            failed_cases.append({
                "question": question,
                "ground_truth": ground_truth,
                "predicted_answer": predicted_answer,
                "faithfulness_score": faithfulness_score,
                "relevancy_score": relevancy_score,
                "noise_score": noise_score,
                "semantic_f1_score": semantic_f1_score,
                "context_recall": ctxR_used,       # 最终
                "context_precision": ctxP_used     # 最终
            })

    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    avg_f1 = sum(semantic_f1_scores) / len(semantic_f1_scores) if semantic_f1_scores else 0.0
    accuracy = (correct / total * 100.0) if total > 0 else 0.0

    print(f"\n✅ Test accuracy: {correct}/{total} ({accuracy:.2f}%)")
    print(f"📊 Average faithfulness score: {avg_faithfulness:.2f}")
    print(f"🧮 Average Semantic F1 score: {avg_f1:.2f}")

    if failed_cases:
        print("\n⚠️ Failed cases (head 5):")
        for case in failed_cases[:5]:
            print("\n🔍 Question:", case["question"])
            print("📖 Standard answer:", case["ground_truth"])
            print("📝 Predicted answer:", case["predicted_answer"])
            print(f"📊 Faithfulness: {case['faithfulness_score']:.2f}")
            print(f"🎯 Relevancy: {case['relevancy_score']:.2f}")
            print(f"🔊 Noise sensitivity: {case['noise_score']:.2f}")
            print(f"✅ Semantic F1: {case['semantic_f1_score']:.2f}")


if __name__ == "__main__":
    main()
