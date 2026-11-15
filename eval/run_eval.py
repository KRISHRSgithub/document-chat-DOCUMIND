"""Evaluate DocuMind on eval/questions.json.

Metrics
  - Hit@k        : is the expected page among the retrieved chunks?          (no LLM needed)
  - Guard acc.   : are out-of-scope questions refused by the similarity guard? (no LLM needed)
  - Keyword recall + citation rate: only when an LLM provider is given.

Usage
  python eval/run_eval.py                          # retrieval-only, free and fast
  python eval/run_eval.py --provider ollama        # full answer-quality check
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import GUARD_THRESHOLD, DocuMind, load_embeddings  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=str(ROOT / "data" / "sample.pdf"))
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.json"))
    ap.add_argument("--provider", default="none", help="none | ollama | openai | gemini")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    eng = DocuMind(load_embeddings(), args.provider)
    name = Path(args.pdf).name
    eng.add_pdf(args.pdf, name)
    questions = json.loads(Path(args.questions).read_text())

    rows, hits, in_scope, guarded, out_scope, kw_scores, cited = [], 0, 0, 0, 0, [], 0
    for q in questions:
        docs, conf = eng._search(q["question"], args.k, None)
        row = {"question": q["question"], "confidence": round(conf, 3)}

        if q.get("out_of_scope"):
            out_scope += 1
            refused = conf < GUARD_THRESHOLD
            guarded += refused
            row["result"] = "refused (correct)" if refused else "ANSWERED (should refuse)"
        else:
            in_scope += 1
            hit = any(d.metadata["page"] == q["expected_page"] for d in docs)
            hits += hit
            row["result"] = f"hit@{args.k}" if hit else f"MISS (got pages {[d.metadata['page'] for d in docs]})"

            if args.provider != "none":
                res = eng.ask(q["question"], [], k=args.k)
                answer = "".join(res.stream)
                kws = q["expected_keywords"]
                score = sum(k.lower() in answer.lower() for k in kws) / len(kws)
                kw_scores.append(score)
                cited += bool(re.search(r"\[[^\]]+p\.\d+\]", answer))
                row["keyword_recall"] = round(score, 2)
        rows.append(row)

    for r in rows:
        print(f"- {r['question'][:55]:<55} conf={r['confidence']:<6} {r['result']}")

    summary = {
        f"hit@{args.k}": f"{hits}/{in_scope}",
        "guard_accuracy": f"{guarded}/{out_scope}",
    }
    if kw_scores:
        summary["avg_keyword_recall"] = round(sum(kw_scores) / len(kw_scores), 2)
        summary["citation_rate"] = f"{cited}/{in_scope}"
    print("\nSUMMARY", json.dumps(summary, indent=2))

    (ROOT / "eval" / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
