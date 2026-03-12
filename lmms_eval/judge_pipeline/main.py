import json
import os
import re
import argparse
import time
import gc
import torch
from collections import Counter
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_jsonl(filepath):
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


def load_model(model_id):
    try:
        import flash_attn
        attn_impl = "flash_attention_2"
        print(f"  Using Flash Attention {flash_attn.__version__}")
    except ImportError:
        attn_impl = "sdpa"
        print(f"  Using PyTorch SDPA")

    gc.collect()
    torch.cuda.empty_cache()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"  Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left",
    )

    print(f"  Loading model: {model_id} (4-bit NF4)")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Print context length info
    max_pos = getattr(model.config, "max_position_embeddings", "unknown")
    print(f"  Max context length: {max_pos}")

    allocated = torch.cuda.memory_allocated(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  GPU Memory: {allocated:.1f}GB / {total:.1f}GB ({total - allocated:.1f}GB free)")

    return model, tokenizer


def judge_single(model, tokenizer, prompt, max_new_tokens=2048):
    messages = [
        {"role": "system", "content": "You are a strict, impartial judge for evaluating AI model outputs."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    del inputs, outputs, generated_ids
    torch.cuda.empty_cache()

    return response_text, input_length


def run_judging(model, tokenizer, judge_prompts, judge_name, max_new_tokens=2048):
    results = []
    total = len(judge_prompts)

    print(f"\n{'='*80}")
    print(f"Running {judge_name} judge on {total} prompts")
    print(f"{'='*80}\n")

    for i, item in enumerate(judge_prompts):
        qid = item["question_id"]
        prompt = item["judge_prompt"]

        print(f"[{i+1}/{total}] Judging question: {qid} ...", end=" ", flush=True)

        try:
            raw_response, input_tokens = judge_single(model, tokenizer, prompt, max_new_tokens)
            winner = extract_winner(raw_response)
        except Exception as e:
            print(f"ERROR: {e}")
            raw_response = None
            winner = None
            input_tokens = 0
            torch.cuda.empty_cache()

        result = {
            "question_id": qid,
            "question": item.get("question", ""),
            "answer_letter": item.get("answer_letter", ""),
            "answer_full": item.get("answer_full", ""),
            "judge_name": judge_name,
            "raw_response": raw_response,
            "winner": winner,
            "input_tokens": input_tokens,
        }
        results.append(result)

        if winner:
            print(f"Winner: {winner} (input: {input_tokens} tokens)")
        else:
            print(f"Could not parse winner (input: {input_tokens} tokens)")

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

    input_tokens_list = [r["input_tokens"] for r in results if r.get("input_tokens", 0) > 0]
    avg_input = sum(input_tokens_list) / len(input_tokens_list) if input_tokens_list else 0
    max_input = max(input_tokens_list) if input_tokens_list else 0

    stats = {
        "judge_name": judge_name,
        "total_questions": total_judged,
        "total_parsed": total_parsed,
        "unparsed": unparsed,
        "model1_wins": model1_wins,
        "model2_wins": model2_wins,
        "model1_win_rate": round(model1_pct, 2),
        "model2_win_rate": round(model2_pct, 2),
        "avg_input_tokens": round(avg_input),
        "max_input_tokens": max_input,
    }

    print(f"\n{'='*60}")
    print(f"  RESULTS: {judge_name}")
    print(f"{'='*60}")
    print(f"  Total questions judged:    {total_judged}")
    print(f"  Successfully parsed:       {total_parsed}")
    print(f"  Failed to parse:           {unparsed}")
    print(f"  Avg input tokens:          {round(avg_input)}")
    print(f"  Max input tokens:          {max_input}")
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
            f.write(f"Input tokens: {r.get('input_tokens', 'N/A')}\n")
            f.write(f"Raw Response:\n{r.get('raw_response', 'N/A')}\n")
            f.write(f"{'─'*40}\n")
    print(f"  Summary saved to:     {summary_file}")


def parse_args():
    parser = argparse.ArgumentParser(description="Local LLM Judge")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--test-count", type=int, default=5)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Max output tokens (default 2048, increase for reasoning models)")
    parser.add_argument("--input", type=str, default="results/VMME_Description/judge_prompts.jsonl")
    parser.add_argument("--output", type=str, default="results/VMME_Description/judge_outputs")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    model_short = args.model.split("/")[-1].replace("-", "_").replace(".", "_")
    judge_name = f"local_{model_short}_4bit"

    print(f"\n{'#'*70}")
    print(f"  LOCAL LLM JUDGE")
    print(f"{'#'*70}")
    print(f"  Mode:            {'TEST (first {})'.format(args.test_count) if args.test else 'FULL'}")
    print(f"  Model:           {args.model}")
    print(f"  Quantization:    4-bit NF4 + double quant")
    print(f"  Max new tokens:  {args.max_new_tokens}")
    print(f"  Input:           {args.input}")
    print(f"  Output:          {args.output}")
    print(f"  GPU:             {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'#'*70}\n")

    print(f"Loading judge prompts from: {args.input}")
    judge_prompts = load_jsonl(args.input)
    print(f"Loaded {len(judge_prompts)} prompts total")

    if args.test:
        judge_prompts = judge_prompts[:args.test_count]
        print(f"TEST MODE: Using only first {len(judge_prompts)} prompts\n")
    else:
        print(f"FULL MODE: Using all {len(judge_prompts)} prompts\n")

    print(f"Loading model...")
    start_time = time.time()
    model, tokenizer = load_model(args.model)
    load_time = time.time() - start_time
    print(f"  Model loaded in {load_time:.1f}s\n")

    start_time = time.time()
    results = run_judging(model, tokenizer, judge_prompts, judge_name, args.max_new_tokens)
    judge_time = time.time() - start_time

    stats = calculate_statistics(results, judge_name)
    stats["model_id"] = args.model
    stats["model_load_time_s"] = round(load_time, 1)
    stats["judging_time_s"] = round(judge_time, 1)
    stats["avg_time_per_question_s"] = round(judge_time / len(judge_prompts), 1) if judge_prompts else 0

    save_results(results, stats, judge_name, args.output, is_test=args.test)

    print(f"\n{'#'*70}")
    print(f"  DONE")
    print(f"{'#'*70}")
    print(f"  Total time:       {judge_time:.1f}s")
    print(f"  Avg per question: {stats['avg_time_per_question_s']}s")
    print(f"  Model 1 wins:     {stats['model1_wins']} ({stats['model1_win_rate']}%)")
    print(f"  Model 2 wins:     {stats['model2_wins']} ({stats['model2_win_rate']}%)")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
