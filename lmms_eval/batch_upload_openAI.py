import time
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# ===== CONFIG =====
INPUT_BATCH_FILE = "results/LV_Description/batch_judge_prompts_updated.jsonl"
OUTPUT_RESULTS_FILE = "results/LV_Description/batch_judge_prompts_updated_result.jsonl"
CHECK_INTERVAL_SECONDS = 1800   # 1 hour
MODEL_ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
# ==================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def upload_batch_file(filepath):
    print("Uploading batch file...")
    file = client.files.create(
        file=open(filepath, "rb"),
        purpose="batch"
    )
    print(f"Uploaded file_id: {file.id}")
    return file.id


def create_batch(file_id):
    print("Creating batch job...")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint=MODEL_ENDPOINT,
        completion_window=COMPLETION_WINDOW
    )
    print(f"Batch created: {batch.id}")
    return batch.id


def wait_for_completion(batch_id):
    print("\nWaiting for batch completion...")
    print(f"Checking every {CHECK_INTERVAL_SECONDS/3600} hour(s)\n")

    while True:
        batch = client.batches.retrieve(batch_id)

        status = batch.status
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{now}] Batch status: {status}")

        if status == "completed":
            print("Batch completed!")
            return batch

        if status in ["failed", "cancelled", "expired"]:
            raise RuntimeError(f"Batch ended with status: {status}")

        print(f"Sleeping for {CHECK_INTERVAL_SECONDS} seconds...\n")
        time.sleep(CHECK_INTERVAL_SECONDS)


def download_results(output_file_id):
    print("Downloading results...")

    content = client.files.content(output_file_id)

    with open(OUTPUT_RESULTS_FILE, "wb") as f:
        f.write(content.read())

    print(f"Results saved to: {OUTPUT_RESULTS_FILE}")


def main():

    file_id = upload_batch_file(INPUT_BATCH_FILE)

    batch_id = create_batch(file_id)

    batch = wait_for_completion(batch_id)

    output_file_id = batch.output_file_id

    if not output_file_id:
        raise RuntimeError("Batch completed but no output_file_id found")

    download_results(output_file_id)

    print("\nAll done.")


if __name__ == "__main__":
    main()