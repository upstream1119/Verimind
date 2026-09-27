"""
run_cross_judge.py
对 10 个策略的生成结果做三评判器统一打分：
- Qwen-Plus
- GLM-4.5-Air
- DeepSeek-V3.2

输入：
- 默认读取 csv/experiment_results_sci_v3.csv（由 run_sci_experiment_main.py 产出）

输出：
- csv/cross_judge_results.csv
"""
import argparse
import os
import re
import time
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from llama_index.llms.dashscope import DashScope
from openai import OpenAI
from zhipuai import ZhipuAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
INPUT_DEFAULT = os.path.join(
    PROJECT_ROOT, "revision", "artifacts", "1_2_baseline", "1_2_1_main", "experiment_results_sci_v3.csv"
)
OUTPUT_DEFAULT = os.path.join(
    PROJECT_ROOT, "revision", "artifacts", "1_2_baseline", "1_2_2_cross_judge", "cross_judge_results.csv"
)

JUDGES_DEFAULT = ["qwen", "glm", "deepseek"]

QWEN_LLM = DashScope(
    model_name="qwen-plus",
    api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
    max_tokens=64,
    temperature=0.01,
)
GLM_CLIENT = ZhipuAI(api_key=os.environ.get("ZHIPU_API_KEY", ""))
DEEPSEEK_CLIENT = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com",
)

REFUSAL_FLAGS = [
    "information not found",
    "not found",
    "cannot find",
    "unable to determine",
    "rejected",
    "无证据",
    "未找到",
]


def make_judge_prompt(answer: str, context: str) -> str:
    return (
        "Evaluate if the [Generated Answer] is strictly derived from the [Retrieved Context].\n"
        "Output ONLY an integer score from 0 to 10.\n"
        "10 = completely faithful; 0 = completely hallucinated.\n\n"
        f"[Context]: {(context or '')[:3000]}\n"
        f"[Answer]: {answer}\n"
        "Score:"
    )


def parse_score(text: str) -> int:
    m = re.search(r"\b(10|[0-9])\b", text or "")
    return int(m.group(1)) if m else 6


def judge_with_qwen(prompt: str) -> int:
    raw = QWEN_LLM.complete(prompt).text.strip()
    return parse_score(raw)


def judge_with_glm(prompt: str) -> int:
    resp = GLM_CLIENT.chat.completions.create(
        model="glm-4.5-air",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.01,
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = (resp.choices[0].message.content or "").strip()
    return parse_score(raw)


def judge_with_deepseek(prompt: str) -> int:
    resp = DEEPSEEK_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.01,
    )
    raw = (resp.choices[0].message.content or "").strip()
    return parse_score(raw)


def run_single_judge(judge: str, answer: str, context: str, retries: int = 3) -> Tuple[float, float, str]:
    if any(flag in (answer or "").lower() for flag in REFUSAL_FLAGS):
        return 10.0, 0.0, "refusal_shortcut"

    prompt = make_judge_prompt(answer, context)
    last_err = ""
    for attempt in range(retries):
        try:
            if judge == "qwen":
                score = judge_with_qwen(prompt)
            elif judge == "glm":
                score = judge_with_glm(prompt)
            elif judge == "deepseek":
                score = judge_with_deepseek(prompt)
            else:
                raise ValueError(f"Unsupported judge: {judge}")
            hal = 1.0 if score < 7 else 0.0
            return float(score), hal, "ok"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    return 6.0, 0.0, f"failed:{last_err}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-judge evaluation for 10 strategies.")
    parser.add_argument("--input_csv", default=INPUT_DEFAULT, help="Input generation CSV.")
    parser.add_argument("--output_csv", default=OUTPUT_DEFAULT, help="Output cross-judge CSV.")
    parser.add_argument("--limit", type=int, default=0, help="Limit base rows for smoke test.")
    parser.add_argument(
        "--judges",
        default=",".join(JUDGES_DEFAULT),
        help="Comma-separated judges, e.g. qwen,glm,deepseek",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judges: List[str] = [j.strip() for j in args.judges.split(",") if j.strip()]

    df = pd.read_csv(args.input_csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    required_cols = ["question_id", "strategy", "generated_answer"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV 缺少必要列: {missing}")

    if "judge_context" not in df.columns:
        df["judge_context"] = ""
    if "retrieval_hit" not in df.columns:
        df["retrieval_hit"] = 0
    if "routed_granularity" not in df.columns:
        df["routed_granularity"] = "CONCEPT"
    if "retrieval_path" not in df.columns:
        df["retrieval_path"] = "DEFAULT"
    if "crag_score" not in df.columns:
        df["crag_score"] = -1.0
    if "trust_score" not in df.columns:
        df["trust_score"] = 0.0

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)

    completed = set()
    if os.path.exists(args.output_csv):
        old = pd.read_csv(args.output_csv)
        for _, row in old.iterrows():
            completed.add((int(row["question_id"]), row["strategy"], row["judge"]))
        print(f"检测到历史结果，已完成 {len(completed)} 条 judge 记录。")

    total = len(df) * len(judges)
    print(f"总任务：{len(df)} base rows × {len(judges)} judges = {total}")

    for idx, row in df.iterrows():
        qid = int(row["question_id"])
        strategy = row["strategy"]
        answer = str(row.get("generated_answer", ""))
        context = str(row.get("judge_context", ""))
        question = str(row.get("question", ""))

        for judge in judges:
            key = (qid, strategy, judge)
            if key in completed:
                continue

            score, hal, judge_status = run_single_judge(judge, answer, context)

            out = {
                "question_id": qid,
                "question": question,
                "strategy": strategy,
                "judge": judge,
                "faith_score": score,
                "hallucination_rate": hal,
                "retrieval_hit": int(row.get("retrieval_hit", 0)),
                "routed_granularity": str(row.get("routed_granularity", "")),
                "retrieval_path": str(row.get("retrieval_path", "")),
                "crag_score": float(row.get("crag_score", -1.0)),
                "trust_score": float(row.get("trust_score", 0.0)),
                "category": str(row.get("category", "")),
                "difficulty": str(row.get("difficulty", "")),
                "judge_status": judge_status,
            }
            pd.DataFrame([out]).to_csv(
                args.output_csv,
                mode="a",
                header=not os.path.exists(args.output_csv),
                index=False,
                encoding="utf-8-sig",
            )
            print(
                f"[{idx+1}/{len(df)}] {strategy:<22} | judge={judge:<8} "
                f"| score={score:.1f} | hal={hal:.0f} | {judge_status}"
            )

    print(f"\n完成：结果已保存至 {args.output_csv}")


if __name__ == "__main__":
    main()
