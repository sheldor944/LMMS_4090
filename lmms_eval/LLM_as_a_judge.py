import json
import os
import time
import re
import argparse
from collections import Counter
from datetime import datetime

# ====== Install these if needed ======
# pip install openai google-genai python-dotenv
# =====================================

from dotenv import load_dotenv
import openai
from google import genai


def load_jsonl(filepath):
    """Load a JSONL file (one JSON object per line)."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_winner(response_text):
    if not response_text:
        return None

    patterns = [
        r'Winner:\s*(\d)',
        r'Winner:\s*Model\s*(\d)',
        r'Winner:\s*Description\s*(\d)',
        r'\*\*Winner:\s*(\d)\*\*',
        r'\*\*Winner:\s*Model\s*(\d)\*\*',
        r'\*\*Winner:\s*Description\s*(\d)\*\*',
        r'Winner:\s*\*\*(\d)\*\*',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            winner = int(match.group(1))
            if winner in [1, 2]:
                return winner

    return None


class GeminiJudge:
    def __init__(self, api_key, model_name="gemini-3-flash-preview"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def judge(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                return response.text
            except Exception as e:
                print(f"\n  Gemini attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  All retries exhausted.")
                    return None


class OpenAIJudge:
    def __init__(self, api_key, model_name="gpt-5.4-2026-03-05"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name

    def judge(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a strict, impartial judge for evaluating AI model outputs."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_completion_tokens=1024,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"\n  OpenAI attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** (attempt + 1)
                    print(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"  All retries exhausted.")
                    return None


def run_judging(judge_prompts, judge_instance, judge_name, output_dir, rate_limit_delay=1.0):
    results = []
    total = len(judge_prompts)

    print(f"\n{'='*80}")
    print(f"Running {judge_name} judge on {total} prompts")
    print(f"{'='*80}\n")

    for i, item in enumerate(judge_prompts):
        qid = item["question_id"]
        prompt = item["judge_prompt"]

        print(f"[{i+1}/{total}] Judging question: {qid} ...", end=" ", flush=True)

        raw_response = judge_instance.judge(prompt)
        winner = extract_winner(raw_response) if raw_response else None

        result = {
            "question_id": qid,
            "question": item.get("question", ""),
            "answer_letter": item.get("answer_letter", ""),
            "answer_full": item.get("answer_full", ""),
            "judge_name": judge_name,
            "raw_response": raw_response,
            "winner": winner,
        }
        results.append(result)

        if winner:
            print(f"Winner: {winner}")
        else:
            print(f"Could not parse winner")

        if i < total - 1:
            time.sleep(rate_limit_delay)

    return results


def calculate_statistics(results, judge_name):
    winners = [r["winner"] for r in results if r["winner"] is not None]
    total_judged = len(results)
    total_parsed = len(winners)
    unparsed = total_judged - total_parsed

    counter = Counter(winners)
    model1_wins = counter.get(1, 0)
    model2_wins = counter.get(2, 0)

    model1_pct = (model1_wins / total_parsed * 100) if total_parsed > 0 else 0
    model2_pct = (model2_wins / total_parsed * 100) if total_parsed > 0 else 0

    stats = {
        "judge_name": judge_name,
        "total_questions": total_judged,
        "total_parsed": total_parsed,
        "unparsed": unparsed,
        "model1_wins": model1_wins,
        "model2_wins": model2_wins,
        "model1_win_rate": round(model1_pct, 2),
        "model2_win_rate": round(model2_pct, 2),
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS: {judge_name}")
    print(f"{'='*60}")
    print(f"  Total questions judged:    {total_judged}")
    print(f"  Successfully parsed:       {total_parsed}")
    print(f"  Failed to parse:           {unparsed}")
    print(f"{'─'*60}")
    print(f"  Model 1 (Description 1) wins:  {model1_wins}  ({model1_pct:.2f}%)")
    print(f"  Model 2 (Description 2) wins:  {model2_wins}  ({model2_pct:.2f}%)")
    print(f"{'='*60}\n")

    return stats


def save_results(results, stats, judge_name, output_dir, is_test=False):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "TEST_" if is_test else ""

    raw_file = os.path.join(output_dir, f"{prefix}{judge_name}_raw_results_{timestamp}.jsonl")
    with open(raw_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Raw results saved to: {raw_file}")

    stats_file = os.path.join(output_dir, f"{prefix}{judge_name}_stats_{timestamp}.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Statistics saved to:  {stats_file}")

    summary_file = os.path.join(output_dir, f"{prefix}{judge_name}_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Judge: {judge_name}\n")
        f.write(f"Mode: {'TEST' if is_test else 'FULL'}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total questions: {stats['total_questions']}\n")
        f.write(f"Parsed:          {stats['total_parsed']}\n")
        f.write(f"Unparsed:        {stats['unparsed']}\n\n")
        f.write(f"Model 1 wins:    {stats['model1_wins']}  ({stats['model1_win_rate']}%)\n")
        f.write(f"Model 2 wins:    {stats['model2_wins']}  ({stats['model2_win_rate']}%)\n\n")
        f.write(f"{'='*60}\n")
        f.write(f"Per-question breakdown:\n")
        f.write(f"{'='*60}\n\n")
        for r in results:
            f.write(f"Question ID: {r['question_id']}\n")
            q = r.get('question', '')
            f.write(f"Question:    {q[:100]}...\n" if len(q) > 100 else f"Question:    {q}\n")
            f.write(f"Winner:      {r['winner']}\n")
            f.write(f"Raw Response:\n{r.get('raw_response', 'N/A')}\n")
            f.write(f"{'─'*40}\n")
    print(f"  Summary saved to:     {summary_file}")

    return raw_file, stats_file, summary_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM judge on description comparison prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python judge_runner.py --test
  python judge_runner.py --test --test-count 10
  python judge_runner.py --gemini
  python judge_runner.py --openai
  python judge_runner.py --gemini --openai
  python judge_runner.py --test --env-file keys.env
        """
    )

    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--test-count", type=int, default=5)
    parser.add_argument("--gemini", action="store_true", default=False)
    parser.add_argument("--openai", action="store_true", default=False)
    parser.add_argument("--gemini-model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--openai-model", type=str, default="gpt-5.4-2026-03-05")
    parser.add_argument("--input", type=str, default="results/VMME_Description/judge_prompts_updated.jsonl")
    parser.add_argument("--output", type=str, default="results/VMME_Description/judge_outputs_updated")
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--env-file", type=str, default=".env")

    args = parser.parse_args()

    if not args.gemini and not args.openai:
        args.gemini = False
        args.openai = True

    return args


def main():
    args = parse_args()

    # ── Find and load .env file ──
    # Try multiple locations to find the .env file
    env_file = args.env_file
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Search order:
    # 1. Exact path provided via --env-file
    # 2. .env in current working directory
    # 3. .env in the same directory as this script
    search_paths = [
        env_file,
        os.path.join(os.getcwd(), ".env"),
        os.path.join(script_dir, ".env"),
    ]

    env_loaded = False
    env_path_used = None
    for path in search_paths:
        if os.path.exists(path):
            load_dotenv(dotenv_path=path, override=True)
            env_loaded = True
            env_path_used = os.path.abspath(path)
            print(f"✓ Loaded .env from: {env_path_used}")

            # Debug: print raw file contents (keys only, not values)
            with open(path, 'r') as f:
                lines = f.readlines()
                print(f"  .env file has {len(lines)} lines:")
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key = line.split('=')[0].strip()
                        print(f"    - {key} = {'*' * 8}...")
            break

    if not env_loaded:
        print(f"⚠ No .env file found. Searched:")
        for path in search_paths:
            print(f"    {os.path.abspath(path)}")
        print(f"  Falling back to system environment variables.\n")

    # ── Read API keys ──
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

    # ── Debug: verify keys are loaded ──
    print(f"\n  DEBUG - GEMINI_API_KEY from os.environ: {repr(GEMINI_API_KEY[:20]) + '...' if GEMINI_API_KEY else 'None'}")
    print(f"  DEBUG - OPENAI_API_KEY from os.environ: {repr(OPENAI_API_KEY[:20]) + '...' if OPENAI_API_KEY else 'None'}")

    # ── Print configuration ──
    print(f"\n{'#'*70}")
    print(f"  JUDGE RUNNER CONFIGURATION")
    print(f"{'#'*70}")
    print(f"  Mode:            {'TEST (first {})'.format(args.test_count) if args.test else 'FULL'}")
    print(f"  Input:           {args.input}")
    print(f"  Output:          {args.output}")
    print(f"  .env file:       {env_path_used if env_loaded else 'NOT FOUND'}")
    print(f"  Run Gemini:      {args.gemini}  (model: {args.gemini_model})")
    print(f"  Run OpenAI:      {args.openai}  (model: {args.openai_model})")
    print(f"  API delay:       {args.delay}s")
    print(f"  GEMINI_API_KEY:  {'✓ Loaded (' + GEMINI_API_KEY[:8] + '...)' if GEMINI_API_KEY else '✗ NOT SET'}")
    print(f"  OPENAI_API_KEY:  {'✓ Loaded (' + OPENAI_API_KEY[:8] + '...)' if OPENAI_API_KEY else '✗ NOT SET'}")
    print(f"{'#'*70}\n")

    # ── Load judge prompts ──
    print(f"Loading judge prompts from: {args.input}")
    judge_prompts = load_jsonl(args.input)
    print(f"Loaded {len(judge_prompts)} prompts total")

    if args.test:
        judge_prompts = judge_prompts[:args.test_count]
        print(f"TEST MODE: Using only first {len(judge_prompts)} prompts\n")
    else:
        print(f"FULL MODE: Using all {len(judge_prompts)} prompts\n")

    all_stats = {}

    # ── Run Gemini Judge ──
    if args.gemini:
        if GEMINI_API_KEY:
            gemini_judge = GeminiJudge(api_key=GEMINI_API_KEY, model_name=args.gemini_model)
            judge_label = f"gemini_{args.gemini_model.replace('-', '_').replace('.', '_')}"

            gemini_results = run_judging(
                judge_prompts, gemini_judge, judge_label,
                args.output, rate_limit_delay=args.delay
            )
            gemini_stats = calculate_statistics(gemini_results, judge_label)
            save_results(gemini_results, gemini_stats, judge_label, args.output, is_test=args.test)
            all_stats["gemini"] = gemini_stats
        else:
            print("⚠  Skipping Gemini: GEMINI_API_KEY not found.")
            print("   Make sure your .env file contains: GEMINI_API_KEY=your_key_here")
            print(f"   Current .env location: {env_path_used if env_loaded else 'NOT FOUND'}\n")

    # ── Run OpenAI Judge ──
    if args.openai:
        if OPENAI_API_KEY:
            openai_judge = OpenAIJudge(api_key=OPENAI_API_KEY, model_name=args.openai_model)
            judge_label = f"openai_{args.openai_model.replace('-', '_')}"

            openai_results = run_judging(
                judge_prompts, openai_judge, judge_label,
                args.output, rate_limit_delay=args.delay
            )
            openai_stats = calculate_statistics(openai_results, judge_label)
            save_results(openai_results, openai_stats, judge_label, args.output, is_test=args.test)
            all_stats["openai"] = openai_stats
        else:
            print("⚠  Skipping OpenAI: OPENAI_API_KEY not found.")
            print("   Make sure your .env file contains: OPENAI_API_KEY=your_key_here\n")

    # ── Combined Summary ──
    if all_stats:
        print(f"\n{'#'*70}")
        print(f"  COMBINED SUMMARY {'(TEST MODE)' if args.test else '(FULL RUN)'}")
        print(f"{'#'*70}\n")
        for key, stats in all_stats.items():
            print(f"  {stats['judge_name']}:")
            print(f"    Total judged:  {stats['total_questions']}")
            print(f"    Model 1 wins:  {stats['model1_wins']} ({stats['model1_win_rate']}%)")
            print(f"    Model 2 wins:  {stats['model2_wins']} ({stats['model2_win_rate']}%)")
            print(f"    Unparsed:      {stats['unparsed']}")
            print()

        os.makedirs(args.output, exist_ok=True)
        prefix = "TEST_" if args.test else ""
        combined_file = os.path.join(args.output, f"{prefix}combined_stats.json")
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"  Combined stats saved to: {combined_file}")
    else:
        print("\n⚠  No judges were run. Check your .env file and flags.")

    print("\nDone!")


if __name__ == "__main__":
    main()