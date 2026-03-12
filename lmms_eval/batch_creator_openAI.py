import json

INPUT_FILE = "results/LV_Description/judge_prompts_updated.jsonl"
OUTPUT_FILE = "results/LV_Description/batch_judge_prompts_updated.jsonl"
MODEL = "gpt-5.4-2026-03-05"


def create_batch_file():

    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
    
        for i, line in enumerate(infile):

            item = json.loads(line)

            qid = str(item["question_id"])  # <-- Ensure custom_id is a string
            prompt = item["judge_prompt"]

            batch_entry = {
                "custom_id": qid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a strict, impartial judge for evaluating AI model outputs."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "temperature": 0,
                    "max_completion_tokens": 1024
                }
            }

            outfile.write(json.dumps(batch_entry, ensure_ascii=False) + "\n")
    print(f"Batch request file created: {OUTPUT_FILE}")


if __name__ == "__main__":
    create_batch_file()