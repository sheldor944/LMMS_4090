import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import List, Dict


def load_data(data_path: str) -> List[Dict]:
    path = Path(data_path)

    if path.suffix == ".jsonl":
        data = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    elif path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)

    elif path.suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def run_batch_judging(
    model,
    tokenizer,
    data: List[Dict],
    output_path: str = "judge_results.jsonl",
    use_swap: bool = True,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    resume: bool = True,
):
    from judge_engine import judge_single, judge_with_swap

    output_file = Path(output_path)

    processed_ids = set()
    if resume and output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    processed_ids.add(entry.get("id"))
        print(f"Resuming: {len(processed_ids)} already processed")

    stats = {"total": 0, "winner_1": 0, "winner_2": 0, "parse_fail": 0, "inconsistent": 0}

    with open(output_file, "a") as f_out:
        for item in tqdm(data, desc="Judging"):
            item_id = item.get("id", str(stats["total"]))

            if item_id in processed_ids:
                continue

            question = item["question"]
            answer = item["answer"]
            desc1 = item["description_1"]
            desc2 = item["description_2"]

            try:
                if use_swap:
                    result = judge_with_swap(
                        model, tokenizer,
                        question, answer, desc1, desc2,
                        max_new_tokens, temperature,
                    )
                    winner = result["final_winner"]
                    consistent = result["consistent"]

                    output_entry = {
                        "id": item_id,
                        "question": question,
                        "answer": answer,
                        "final_winner": winner,
                        "consistent": consistent,
                        "original_response": result["original_order"]["raw_response"],
                        "swapped_response": result["swapped_order"]["raw_response"],
                        "original_winner": result["original_order"]["winner"],
                        "swapped_winner_mapped": result["swapped_winner_mapped"],
                        "part1_analysis": result["original_order"]["part1_analysis"],
                        "part2_analysis": result["original_order"]["part2_analysis"],
                        "input_tokens": result["original_order"]["input_tokens"],
                        "output_tokens": result["original_order"]["output_tokens"],
                        "timestamp": datetime.now().isoformat(),
                    }

                    if not consistent:
                        stats["inconsistent"] += 1

                else:
                    result = judge_single(
                        model, tokenizer,
                        question, answer, desc1, desc2,
                        max_new_tokens, temperature,
                    )
                    winner = result["winner"]

                    output_entry = {
                        "id": item_id,
                        "question": question,
                        "answer": answer,
                        "final_winner": winner,
                        "raw_response": result["raw_response"],
                        "part1_analysis": result["part1_analysis"],
                        "part2_analysis": result["part2_analysis"],
                        "input_tokens": result["input_tokens"],
                        "output_tokens": result["output_tokens"],
                        "timestamp": datetime.now().isoformat(),
                    }

                stats["total"] += 1
                if winner == 1:
                    stats["winner_1"] += 1
                elif winner == 2:
                    stats["winner_2"] += 1
                else:
                    stats["parse_fail"] += 1

                f_out.write(json.dumps(output_entry) + "\n")
                f_out.flush()

            except Exception as e:
                print(f"\nError on item {item_id}: {e}")
                error_entry = {
                    "id": item_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
                f_out.write(json.dumps(error_entry) + "\n")
                f_out.flush()
                stats["parse_fail"] += 1
                stats["total"] += 1
                continue

            if stats["total"] % 50 == 0:
                print(f"\n--- Stats after {stats['total']} items ---")
                print(f"  Winner 1: {stats['winner_1']} | Winner 2: {stats['winner_2']}")
                print(f"  Parse fails: {stats['parse_fail']} | Inconsistent: {stats['inconsistent']}")

    return stats
