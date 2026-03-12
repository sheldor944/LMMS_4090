import time
import json
import argparse
from datetime import datetime
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# ===== CONFIG =====
INPUT_BATCH_FILE = "results/LV_Description/batch_judge_prompts_updated.jsonl"
OUTPUT_RESULTS_FILE = "results/LV_Description/batch_judge_prompts_updated_result.jsonl"
TEMP_DIR = "results/LV_Description/temp_chunks/"
CHECK_INTERVAL_SECONDS = 60  # Check every 60 seconds (chunks are smaller, finish faster)
MODEL_ENDPOINT = "/v1/chat/completions"
COMPLETION_WINDOW = "24h"
# ==================

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def load_and_chunk(input_file, chunk_size, max_entries=None):
    """Load the batch JSONL file and split into chunks."""
    entries = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            entries.append(line.strip())
            if max_entries and len(entries) >= max_entries:
                break

    chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]
    print(f"Total entries loaded: {len(entries)}")
    print(f"Split into {len(chunks)} chunks of up to {chunk_size} entries each\n")
    return chunks


def write_chunk_file(chunk, chunk_index):
    """Write a chunk to a temporary JSONL file."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    chunk_file = os.path.join(TEMP_DIR, f"chunk_{chunk_index}.jsonl")
    with open(chunk_file, "w", encoding="utf-8") as f:
        for line in chunk:
            f.write(line + "\n")
    print(f"  Chunk file written: {chunk_file} ({len(chunk)} entries)")
    return chunk_file


def upload_batch_file(filepath):
    """Upload batch file to OpenAI."""
    print("  Uploading batch file...")
    with open(filepath, "rb") as f:
        file = client.files.create(file=f, purpose="batch")
    print(f"  Uploaded file_id: {file.id}")
    return file.id


def create_batch(file_id, chunk_index):
    """Create a batch job."""
    print("  Creating batch job...")
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint=MODEL_ENDPOINT,
        completion_window=COMPLETION_WINDOW,
        metadata={"description": f"chunk_{chunk_index}"}
    )
    print(f"  Batch created: {batch.id}")
    return batch.id


def wait_for_completion(batch_id, poll_interval):
    """Poll until batch completes or fails."""
    print(f"  Waiting for batch completion (polling every {poll_interval}s)...")

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        completed = batch.request_counts.completed
        failed = batch.request_counts.failed
        total = batch.request_counts.total

        print(f"  [{now}] Status: {status} | {completed}/{total} completed, {failed} failed")

        if status == "completed":
            print("  ✅ Batch completed!")
            return batch

        if status in ["failed", "cancelled", "expired"]:
            error_msg = f"Batch ended with status: {status}"
            if batch.errors:
                for error in batch.errors.data:
                    error_msg += f"\n    Error: {error.code} - {error.message}"
            raise RuntimeError(error_msg)

        time.sleep(poll_interval)


def download_chunk_results(batch, chunk_index):
    """Download results for a completed chunk."""
    output_file_id = batch.output_file_id
    if not output_file_id:
        print(f"  ⚠️ No output_file_id for chunk {chunk_index}")
        return None

    print("  Downloading results...")
    content = client.files.content(output_file_id)

    result_file = os.path.join(TEMP_DIR, f"result_chunk_{chunk_index}.jsonl")
    with open(result_file, "wb") as f:
        f.write(content.read())

    print(f"  Results saved: {result_file}")
    return result_file


def merge_results(num_chunks):
    """Merge all chunk results into the final output file."""
    print(f"\nMerging {num_chunks} chunk results...")
    total_lines = 0

    with open(OUTPUT_RESULTS_FILE, "w", encoding="utf-8") as outfile:
        for i in range(num_chunks):
            chunk_result = os.path.join(TEMP_DIR, f"result_chunk_{i}.jsonl")
            try:
                with open(chunk_result, "r", encoding="utf-8") as infile:
                    for line in infile:
                        outfile.write(line)
                        total_lines += 1
            except FileNotFoundError:
                print(f"  ⚠️ Missing: {chunk_result}")

    print(f"📦 All results merged into: {OUTPUT_RESULTS_FILE} ({total_lines} entries)")


def run(chunk_size, max_entries=None, poll_interval=60):
    """Main execution loop."""

    # Load and chunk
    chunks = load_and_chunk(INPUT_BATCH_FILE, chunk_size, max_entries)

    # Process each chunk sequentially
    for idx, chunk in enumerate(chunks):
        print(f"{'=' * 60}")
        print(f"CHUNK {idx + 1}/{len(chunks)} ({len(chunk)} entries)")
        print(f"{'=' * 60}")

        # Write temp file
        chunk_file = write_chunk_file(chunk, idx)

        # Upload
        file_id = upload_batch_file(chunk_file)

        # Submit batch
        batch_id = create_batch(file_id, idx)

        # Wait for completion
        batch = wait_for_completion(batch_id, poll_interval)

        # Download results
        download_chunk_results(batch, idx)

        print()

    # Merge everything
    merge_results(len(chunks))
    print("\n✅ All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunked OpenAI Batch API Runner")
    parser.add_argument(
        "--mode",
        choices=["test", "full"],
        default="full",
        help="'test' = 100 entries, 10 per chunk | 'full' = all entries, 500 per chunk"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size"
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Override max entries to process"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        help="Seconds between status checks"
    )
    args = parser.parse_args()

    # Set defaults based on mode
    if args.mode == "test":
        chunk_size = args.chunk_size or 10
        max_entries = args.max_entries or 100
        poll_interval = args.poll_interval or 30      # Faster polling for test
    else:
        chunk_size = args.chunk_size or 700
        max_entries = args.max_entries                 # None = all entries
        poll_interval = args.poll_interval or 60

    print(f"🚀 Mode: {args.mode}")
    print(f"   Chunk size: {chunk_size}")
    print(f"   Max entries: {max_entries or 'ALL'}")
    print(f"   Poll interval: {poll_interval}s")
    print(f"   Input: {INPUT_BATCH_FILE}")
    print(f"   Output: {OUTPUT_RESULTS_FILE}\n")

    run(
        chunk_size=chunk_size,
        max_entries=max_entries,
        poll_interval=poll_interval
    )