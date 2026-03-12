import torch
import re
from typing import Optional, Tuple


def judge_single(
    model,
    tokenizer,
    question: str,
    answer: str,
    description_1: str,
    description_2: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    from judge_prompt import build_judge_prompt

    messages = build_judge_prompt(question, answer, description_1, description_2)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs[0][input_length:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    winner = parse_winner(response_text)
    part1_analysis, part2_analysis = parse_analyses(response_text)

    return {
        "raw_response": response_text,
        "winner": winner,
        "part1_analysis": part1_analysis,
        "part2_analysis": part2_analysis,
        "input_tokens": input_length,
        "output_tokens": len(generated_ids),
    }


def parse_winner(response: str) -> Optional[int]:
    patterns = [
        r"Winner:\s*(\d)",
        r"Winner:\s*Description\s*(\d)",
        r"Winner:\s*Model\s*(\d)",
        r"\*\*Winner:\s*(\d)\*\*",
        r"\*\*Winner\*\*:\s*(\d)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            winner = int(match.group(1))
            if winner in [1, 2]:
                return winner
    return None


def parse_analyses(response: str) -> Tuple[str, str]:
    part1 = ""
    part2 = ""

    part1_match = re.search(
        r"Part 1 analysis:\s*(.+?)(?=Part 2 analysis:|Winner:|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if part1_match:
        part1 = part1_match.group(1).strip()

    part2_match = re.search(
        r"Part 2 analysis:\s*(.+?)(?=Winner:|$)",
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if part2_match:
        part2 = part2_match.group(1).strip()

    return part1, part2


def judge_with_swap(
    model,
    tokenizer,
    question: str,
    answer: str,
    description_1: str,
    description_2: str,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    result_original = judge_single(
        model, tokenizer, question, answer,
        description_1, description_2,
        max_new_tokens, temperature,
    )

    result_swapped = judge_single(
        model, tokenizer, question, answer,
        description_2, description_1,
        max_new_tokens, temperature,
    )

    swapped_winner = result_swapped["winner"]
    if swapped_winner == 1:
        swapped_winner_mapped = 2
    elif swapped_winner == 2:
        swapped_winner_mapped = 1
    else:
        swapped_winner_mapped = None

    original_winner = result_original["winner"]
    consistent = (original_winner == swapped_winner_mapped)

    if consistent:
        final_winner = original_winner
    else:
        final_winner = original_winner

    return {
        "final_winner": final_winner,
        "consistent": consistent,
        "original_order": result_original,
        "swapped_order": result_swapped,
        "swapped_winner_mapped": swapped_winner_mapped,
    }
