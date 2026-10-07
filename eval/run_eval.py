"""Evaluate retrieval and answers on a small labeled question set.

    python -m eval.run_eval                     # all retrieval modes, default LLM
    python -m eval.run_eval --provider anthropic
    python -m eval.run_eval --modes hybrid --no-rerank

Retrieval metrics (per mode):
    hit@k   share of questions where a relevant chunk is in the top k
    MRR     mean reciprocal rank of the first relevant chunk

Answer metrics (hybrid mode, chosen provider):
    keyword recall   share of expected key facts that appear in the answer
    groundedness     share of answer sentences whose content words mostly
                     appear in the retrieved sources (a cheap hallucination check)
    refusal          the unanswerable question should get the not-found reply
"""
import argparse
import json
import re
import tempfile
from pathlib import Path

from docusense.config import Settings
from docusense.pipeline import RAGPipeline
from docusense.prompts import NOT_FOUND_MESSAGE
from docusense.text import split_sentences, tokenize

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
QUESTIONS = Path(__file__).resolve().parent / "questions.jsonl"


def is_relevant(text: str, needles) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def groundedness(answer: str, sources) -> float:
    source_tokens = set(tokenize(" ".join(sources)))
    sentences = [s for s in split_sentences(re.sub(r"\[\d+\]", "", answer)) if len(tokenize(s)) >= 3]
    if not sentences:
        return 1.0
    grounded = sum(
        1 for s in sentences if len(set(tokenize(s)) & source_tokens) / len(set(tokenize(s))) >= 0.6
    )
    return grounded / len(sentences)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["bm25", "vector", "hybrid"])
    ap.add_argument("--provider", default="auto")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results.json"))
    args = ap.parse_args()

    questions = [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]
    answerable = [q for q in questions if q["doc"]]

    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(data_dir=Path(tmp), rerank=not args.no_rerank, top_k=args.top_k)
        pipe = RAGPipeline(settings)
        for f in sorted(SAMPLES.iterdir()):
            pipe.ingest(f)

        print(f"Embedder: {pipe.embedder.name} | reranker: {pipe.reranker.name if pipe.reranker else 'off'}")
        print(f"{len(pipe.kb.chunks)} chunks from {len(pipe.kb.documents)} docs, {len(answerable)} labeled questions\n")

        results = {"embedder": pipe.embedder.name, "reranker": pipe.reranker.name if pipe.reranker else None,
                   "top_k": args.top_k, "retrieval": {}, "answers": {}}

        k_label = f"hit@{args.top_k}"
        print(f"{'mode':<8} {'hit@1':>6} {'hit@3':>6} {k_label:>6} {'MRR':>6}   paraphrase {k_label}")
        for mode in args.modes:
            hits1 = hits3 = hitsk = mrr = 0.0
            para = [0, 0]
            for q in answerable:
                got = pipe.retrieve(q["question"], top_k=args.top_k, mode=mode)
                ranks = [i for i, rc in enumerate(got, 1)
                         if rc.chunk.filename == q["doc"] and is_relevant(rc.chunk.text, q["relevant"])]
                first = ranks[0] if ranks else None
                hits1 += first == 1
                hits3 += bool(first and first <= 3)
                hitsk += first is not None
                mrr += 1 / first if first else 0
                if q["style"] == "paraphrase":
                    para[0] += first is not None
                    para[1] += 1
            n = len(answerable)
            row = {"hit@1": hits1 / n, "hit@3": hits3 / n, f"hit@{args.top_k}": hitsk / n, "mrr": mrr / n,
                   "paraphrase_hit": para[0] / para[1] if para[1] else None}
            results["retrieval"][mode] = row
            print(f"{mode:<8} {row['hit@1']:>6.2f} {row['hit@3']:>6.2f} {row[f'hit@{args.top_k}']:>6.2f} "
                  f"{row['mrr']:>6.2f}   {para[0]}/{para[1]}")

        provider = pipe.get_provider(args.provider)
        print(f"\nAnswers with {provider.name} ({provider.model}), hybrid retrieval:")
        recall_total = ground_total = 0.0
        per_question = []
        refusal_ok = None
        for q in questions:
            res = pipe.query(q["question"], provider=args.provider, top_k=args.top_k)
            if not q["doc"]:
                refusal_ok = res.answer.strip().startswith(NOT_FOUND_MESSAGE[:30])
                per_question.append({"question": q["question"], "answer": res.answer, "refused": refusal_ok})
                continue
            found = [k for k in q["answer_keywords"] if k.lower() in res.answer.lower()]
            recall = 1.0 if found else 0.0  # any expected key fact counts as answered
            g = groundedness(res.answer, [c.text for c in res.citations])
            recall_total += recall
            ground_total += g
            per_question.append({"question": q["question"], "answer": res.answer, "keyword_hit": recall,
                                 "groundedness": round(g, 2), "provider": res.provider})
        n = len(answerable)
        results["answers"] = {"provider": provider.name, "model": provider.model,
                              "keyword_recall": recall_total / n, "groundedness": ground_total / n,
                              "refuses_unanswerable": refusal_ok, "per_question": per_question}
        print(f"  keyword recall  {recall_total / n:.2f}")
        print(f"  groundedness    {ground_total / n:.2f}")
        print(f"  refuses unanswerable question: {'yes' if refusal_ok else 'no'}")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nDetails written to {args.out}")


if __name__ == "__main__":
    main()
