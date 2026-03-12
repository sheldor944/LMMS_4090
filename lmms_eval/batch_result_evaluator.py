import json
import os
import re
import argparse
from collections import Counter
from datetime import datetime


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
    """Extract winner (1 or 2) from the LLM response text."""
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


def extract_content_from_batch_response(batch_item):
    """Extract the assistant's content text from a batch API response item."""
    try:
        response = batch_item.get("response", {})
        if response.get("status_code") != 200:
            error = batch_item.get("error")
            print(f"  ⚠ Non-200 status for {batch_item.get('custom_id', '?')}: "
                  f"status={response.get('status_code')}, error={error}")
            return None

        body = response.get("body", {})
        choices = body.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", None)
        return None
    except Exception as e:
        print(f"  ⚠ Error extracting content from batch item: {e}")
        return None


def parse_custom_id(custom_id):
    """
    Parse the custom_id to extract question_id and prompt variant.
    Expected format: '001-1', '001-2', 'q42-1', etc.
    Returns (question_id_base, variant_number) e.g. ('001', 1)
    """
    if not custom_id:
        return custom_id, None

    # Try to split on the last hyphen to get base and variant
    parts = custom_id.rsplit('-', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return custom_id, None


def calculate_statistics(results, judge_name):
    """Calculate win statistics from the results."""
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


def calculate_per_question_statistics(results, judge_name):
    """
    When there are multiple prompt variants per question (e.g., swapped order),
    calculate per-question aggregated results.
    """
    from collections import defaultdict

    question_results = defaultdict(list)
    for r in results:
        base_id = r.get("question_id_base", r.get("custom_id", "unknown"))
        question_results[base_id].append(r)

    if all(len(v) == 1 for v in question_results.values()):
        # No multi-variant structure detected
        return None

    print(f"\n{'='*60}")
    print(f"  PER-QUESTION AGGREGATED RESULTS: {judge_name}")
    print(f"{'='*60}")

    consistent = 0
    inconsistent = 0
    per_q_stats = []

    for qid, variants in sorted(question_results.items()):
        variant_winners = []
        for v in variants:
            variant_num = v.get("variant_number")
            winner = v.get("winner")
            variant_winners.append((variant_num, winner))

        # Check consistency: if order was swapped, winner=1 in variant 1
        # should correspond to winner=2 in variant 2 (same underlying model wins)
        is_consistent = None
        if len(variant_winners) == 2:
            w1 = variant_winners[0][1]
            w2 = variant_winners[1][1]
            if w1 is not None and w2 is not None:
                # In swapped evaluation: if both agree on the same model,
                # variant 1 winner=1 + variant 2 winner=2 means consistent
                # variant 1 winner=2 + variant 2 winner=1 means consistent
                # Same winner in both means inconsistent (positional bias)
                if w1 != w2:
                    is_consistent = True
                    consistent += 1
                else:
                    is_consistent = False
                    inconsistent += 1

        per_q_stats.append({
            "question_id": qid,
            "variants": variant_winners,
            "consistent": is_consistent,
        })

    total_checkable = consistent + inconsistent
    if total_checkable > 0:
        print(f"  Questions with both variants parsed: {total_checkable}")
        print(f"  Consistent (no positional bias):     {consistent}  ({consistent/total_checkable*100:.1f}%)")
        print(f"  Inconsistent (positional bias):      {inconsistent}  ({inconsistent/total_checkable*100:.1f}%)")
    print(f"{'='*60}\n")

    return {
        "total_questions_with_variants": len(question_results),
        "total_checkable": total_checkable,
        "consistent": consistent,
        "inconsistent": inconsistent,
        "consistency_rate": round(consistent / total_checkable * 100, 2) if total_checkable > 0 else None,
        "per_question": per_q_stats,
    }


def save_results(results, stats, judge_name, output_dir, is_batch=True,
                 per_question_stats=None, usage_stats=None):
    """Save all output files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "BATCH_" if is_batch else ""

    # Save raw results
    raw_file = os.path.join(output_dir, f"{prefix}{judge_name}_raw_results_{timestamp}.jsonl")
    with open(raw_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Raw results saved to: {raw_file}")

    # Save statistics
    stats_output = {"overall": stats}
    if per_question_stats:
        stats_output["per_question_analysis"] = {
            k: v for k, v in per_question_stats.items() if k != "per_question"
        }
    if usage_stats:
        stats_output["usage"] = usage_stats

    stats_file = os.path.join(output_dir, f"{prefix}{judge_name}_stats_{timestamp}.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_output, f, indent=2, ensure_ascii=False)
    print(f"  Statistics saved to:  {stats_file}")

    # Save human-readable summary
    summary_file = os.path.join(output_dir, f"{prefix}{judge_name}_summary_{timestamp}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Judge: {judge_name}\n")
        f.write(f"Source: Batch API Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total questions: {stats['total_questions']}\n")
        f.write(f"Parsed:          {stats['total_parsed']}\n")
        f.write(f"Unparsed:        {stats['unparsed']}\n\n")
        f.write(f"Model 1 wins:    {stats['model1_wins']}  ({stats['model1_win_rate']}%)\n")
        f.write(f"Model 2 wins:    {stats['model2_wins']}  ({stats['model2_win_rate']}%)\n\n")

        if per_question_stats and per_question_stats.get("total_checkable", 0) > 0:
            f.write(f"{'='*60}\n")
            f.write(f"Positional Bias Analysis (swapped variants):\n")
            f.write(f"{'='*60}\n")
            f.write(f"  Checkable pairs:   {per_question_stats['total_checkable']}\n")
            f.write(f"  Consistent:        {per_question_stats['consistent']}  "
                    f"({per_question_stats['consistency_rate']}%)\n")
            f.write(f"  Inconsistent:      {per_question_stats['inconsistent']}\n\n")

        if usage_stats:
            f.write(f"{'='*60}\n")
            f.write(f"Token Usage:\n")
            f.write(f"{'='*60}\n")
            f.write(f"  Total prompt tokens:     {usage_stats['total_prompt_tokens']}\n")
            f.write(f"  Total completion tokens:  {usage_stats['total_completion_tokens']}\n")
            f.write(f"  Total tokens:            {usage_stats['total_tokens']}\n\n")

        f.write(f"{'='*60}\n")
        f.write(f"Per-response breakdown:\n")
        f.write(f"{'='*60}\n\n")
        for r in results:
            f.write(f"Custom ID:   {r.get('custom_id', 'N/A')}\n")
            f.write(f"Question ID: {r.get('question_id_base', r.get('custom_id', 'N/A'))}\n")
            if r.get('variant_number') is not None:
                f.write(f"Variant:     {r['variant_number']}\n")
            f.write(f"Winner:      {r['winner']}\n")
            f.write(f"Model:       {r.get('model', 'N/A')}\n")
            raw = r.get('raw_response', 'N/A')
            f.write(f"Raw Response:\n{raw}\n")
            f.write(f"{'─'*40}\n")

    print(f"  Summary saved to:     {summary_file}")

    return raw_file, stats_file, summary_file


def process_batch_results(batch_data, judge_prompts_data=None):
    """
    Process batch API response data, extract content, parse winners.
    Optionally merge with original judge prompts for richer output.
    """
    # Build a lookup from original judge prompts if provided
    prompt_lookup = {}
    if judge_prompts_data:
        for item in judge_prompts_data:
            qid = item.get("question_id") or item.get("custom_id")
            if qid:
                prompt_lookup[qid] = item

    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    print(f"\nProcessing {len(batch_data)} batch responses...\n")

    for i, batch_item in enumerate(batch_data):
        custom_id = batch_item.get("custom_id", f"unknown_{i}")
        batch_id = batch_item.get("id", "")

        # Extract content
        content = extract_content_from_batch_response(batch_item)
        winner = extract_winner(content) if content else None

        # Parse custom_id for question grouping
        base_id, variant_num = parse_custom_id(custom_id)

        # Extract model name and usage from response body
        response_body = batch_item.get("response", {}).get("body", {})
        model_name = response_body.get("model", "unknown")
        usage = response_body.get("usage", {})

        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)
        total_tokens += usage.get("total_tokens", 0)

        # Look up original prompt data if available
        original = prompt_lookup.get(custom_id, {})

        result = {
            "custom_id": custom_id,
            "question_id_base": base_id,
            "variant_number": variant_num,
            "batch_request_id": batch_id,
            "model": model_name,
            "question": original.get("question", ""),
            "answer_letter": original.get("answer_letter", ""),
            "answer_full": original.get("answer_full", ""),
            "raw_response": content,
            "winner": winner,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "status_code": batch_item.get("response", {}).get("status_code"),
            "error": batch_item.get("error"),
        }
        results.append(result)

        status = f"Winner: {winner}" if winner else "Could not parse winner"
        print(f"  [{i+1}/{len(batch_data)}] {custom_id} → {status}")

    usage_stats = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
    }

    return results, usage_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate batch API results: extract LLM responses, parse winners, compute accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with batch output file
  python batch_result_evaluator.py --batch-file results/batch_output.jsonl

  # With original judge prompts for richer metadata
  python batch_result_evaluator.py --batch-file results/batch_output.jsonl --prompts-file results/judge_prompts.jsonl

  # Custom output directory and judge label
  python batch_result_evaluator.py --batch-file results/batch_output.jsonl --output results/eval_output --judge-name gpt5_batch

  # Only process first N items for testing
  python batch_result_evaluator.py --batch-file results/batch_output.jsonl --test --test-count 10
        """
    )

    parser.add_argument("--batch-file", type=str, default="results/LV_Description/batch_judge_prompts_updated_result.jsonl",
                        help="Path to the batch API output JSONL file.")
    parser.add_argument("--prompts-file", type=str, default="results/LV_Description/judge_prompts_updated.jsonl",
                        help="(Optional) Path to the original judge prompts JSONL file for metadata enrichment.")
    parser.add_argument("--output", type=str, default="results/LV_Description/batch_result/batch_eval_output_updated",
                        help="Output directory for results.")
    parser.add_argument("--judge-name", type=str, default="GPT-5.4",
                        help="Label for the judge. Auto-detected from model name if not provided.")
    parser.add_argument("--test", action="store_true", default=False,
                        help="Test mode: only process first N items.")
    parser.add_argument("--test-count", type=int, default=5,
                        help="Number of items to process in test mode (default: 5).")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Print configuration ──
    print(f"\n{'#'*70}")
    print(f"  BATCH RESULT EVALUATOR")
    print(f"{'#'*70}")
    print(f"  Batch file:      {args.batch_file}")
    print(f"  Prompts file:    {args.prompts_file or 'Not provided'}")
    print(f"  Output dir:      {args.output}")
    print(f"  Judge name:      {args.judge_name or 'Auto-detect'}")
    print(f"  Mode:            {'TEST (first {})'.format(args.test_count) if args.test else 'FULL'}")
    print(f"{'#'*70}\n")

    # ── Validate input files ──
    if not os.path.exists(args.batch_file):
        print(f"✗ Batch file not found: {args.batch_file}")
        return

    # ── Load batch results ──
    print(f"Loading batch results from: {args.batch_file}")
    batch_data = load_jsonl(args.batch_file)
    print(f"Loaded {len(batch_data)} batch responses")

    if args.test:
        batch_data = batch_data[:args.test_count]
        print(f"TEST MODE: Using only first {len(batch_data)} items\n")

    # ── Load original prompts (optional) ──
    judge_prompts_data = None
    if args.prompts_file:
        if os.path.exists(args.prompts_file):
            print(f"Loading original judge prompts from: {args.prompts_file}")
            judge_prompts_data = load_jsonl(args.prompts_file)
            print(f"Loaded {len(judge_prompts_data)} original prompts")
        else:
            print(f"⚠ Prompts file not found: {args.prompts_file} — proceeding without metadata enrichment")

    # ── Process batch results ──
    results, usage_stats = process_batch_results(batch_data, judge_prompts_data)

    # ── Determine judge name ──
    if args.judge_name:
        judge_name = args.judge_name
    else:
        # Auto-detect from model name in the first result
        model_name = results[0].get("model", "unknown") if results else "unknown"
        judge_name = f"batch_{model_name.replace('-', '_').replace('.', '_')}"
    print(f"\nJudge label: {judge_name}")

    # ── Calculate statistics ──
    stats = calculate_statistics(results, judge_name)

    # ── Per-question aggregation (for swapped-order variants) ──
    per_question_stats = calculate_per_question_statistics(results, judge_name)

    # ── Print usage statistics ──
    print(f"{'='*60}")
    print(f"  TOKEN USAGE")
    print(f"{'='*60}")
    print(f"  Total prompt tokens:      {usage_stats['total_prompt_tokens']:,}")
    print(f"  Total completion tokens:  {usage_stats['total_completion_tokens']:,}")
    print(f"  Total tokens:             {usage_stats['total_tokens']:,}")
    print(f"{'='*60}\n")

    # ── Save everything ──
    print(f"Saving results...")
    save_results(
        results, stats, judge_name, args.output,
        is_batch=True,
        per_question_stats=per_question_stats,
        usage_stats=usage_stats,
    )

    # ── Error summary ──
    errors = [r for r in results if r.get("status_code") != 200 or r.get("error")]
    if errors:
        print(f"\n⚠ {len(errors)} responses had errors:")
        for e in errors:
            print(f"    {e['custom_id']}: status={e.get('status_code')}, error={e.get('error')}")

    unparsed = [r for r in results if r["winner"] is None and r.get("status_code") == 200]
    if unparsed:
        print(f"\n⚠ {len(unparsed)} responses could not be parsed for a winner:")
        for u in unparsed:
            snippet = (u.get("raw_response") or "")[:150]
            print(f"    {u['custom_id']}: \"{snippet}...\"")

    # ── Final combined summary ──
    print(f"\n{'#'*70}")
    print(f"  FINAL SUMMARY {'(TEST MODE)' if args.test else '(FULL RUN)'}")
    print(f"{'#'*70}")
    print(f"  Judge:           {judge_name}")
    print(f"  Total responses: {stats['total_questions']}")
    print(f"  Parsed:          {stats['total_parsed']}")
    print(f"  Unparsed:        {stats['unparsed']}")
    print(f"  Model 1 wins:    {stats['model1_wins']}  ({stats['model1_win_rate']}%)")
    print(f"  Model 2 wins:    {stats['model2_wins']}  ({stats['model2_win_rate']}%)")
    if per_question_stats and per_question_stats.get("consistency_rate") is not None:
        print(f"  Consistency:     {per_question_stats['consistent']}/{per_question_stats['total_checkable']}  "
              f"({per_question_stats['consistency_rate']}%)")
    print(f"  Tokens used:     {usage_stats['total_tokens']:,}")
    print(f"{'#'*70}")

    print("\nDone!")


if __name__ == "__main__":
    main()