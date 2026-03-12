import json
import os
import time
import re
import shutil
import argparse
import inspect
from collections import Counter
from datetime import datetime
from typing import List, Dict

from dotenv import load_dotenv
import openai
from google import genai
from google.genai import types


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


def upload_file_with_mime(client, local_file, mime_type="application/jsonl"):
    """
    Upload a file to Gemini File API, trying multiple approaches
    to set mime_type since the SDK is picky about .jsonl files.
    """
    errors = []

    # Approach 1: types.UploadFileConfig
    try:
        if hasattr(types, 'UploadFileConfig'):
            result = client.files.upload(
                file=local_file,
                config=types.UploadFileConfig(mime_type=mime_type),
            )
            return result
    except Exception as e:
        errors.append(f"UploadFileConfig: {e}")

    # Approach 2: dict config
    try:
        result = client.files.upload(
            file=local_file,
            config={"mime_type": mime_type},
        )
        return result
    except Exception as e:
        errors.append(f"dict config: {e}")

    # Approach 3: direct mime_type kwarg
    try:
        result = client.files.upload(
            file=local_file,
            mime_type=mime_type,
        )
        return result
    except Exception as e:
        errors.append(f"mime_type kwarg: {e}")

    # Approach 4: Rename to .txt (universally recognized)
    try:
        txt_file = local_file + ".txt"
        shutil.copy2(local_file, txt_file)
        result = client.files.upload(file=txt_file)
        try:
            os.remove(txt_file)
        except:
            pass
        return result
    except Exception as e:
        errors.append(f".txt rename: {e}")

    # Approach 5: Rename to .json
    try:
        json_file = local_file.rsplit(".", 1)[0] + ".json"
        shutil.copy2(local_file, json_file)
        result = client.files.upload(file=json_file)
        try:
            os.remove(json_file)
        except:
            pass
        return result
    except Exception as e:
        errors.append(f".json rename: {e}")

    # Approach 6: Use pathlib with explicit suffix
    try:
        from pathlib import Path
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode='w') as tmp:
            with open(local_file, 'r') as src:
                tmp.write(src.read())
            tmp_path = tmp.name
        result = client.files.upload(file=tmp_path)
        try:
            os.remove(tmp_path)
        except:
            pass
        return result
    except Exception as e:
        errors.append(f"tempfile .txt: {e}")

    # All failed
    raise RuntimeError(
        f"Could not upload file after 6 attempts:\n" +
        "\n".join(f"  {i+1}) {err}" for i, err in enumerate(errors))
    )


class GeminiJudge:
    """Original sequential Gemini judge — unchanged."""
    def __init__(self, api_key, model_name="gemini-3-flash-preview"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def judge(self, prompt, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name, contents=prompt,
                )
                return response.text
            except Exception as e:
                print(f"\n  Gemini attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    return None


class GeminiBatchJudge:
    """
    Gemini Native Batch API judge using File API.

    Pipeline:
      1. Write prompts → JSONL file (with custom_id per line)
      2. Upload JSONL → Gemini File API
      3. Submit batch job: client.batches.create(model=..., src=file_name)
      4. Poll until complete
      5. Retrieve output file from job.dest
      6. Parse results using custom_id mapping
    """

    def __init__(
        self,
        api_key=None, model_name="gemini-3-flash-preview",
        gcs_bucket=None, gcs_prefix="batch_jobs",
        project_id=None, location="us-central1",
        use_vertex=False,
    ):
        self.model_name = model_name
        self.gcs_bucket = gcs_bucket
        self.gcs_prefix = gcs_prefix
        self.project_id = project_id
        self.location = location
        self.use_vertex = use_vertex
        self.api_key = api_key

        if use_vertex and project_id:
            self.client = genai.Client(vertexai=True, project=project_id, location=location)
            print(f"  ✓ Vertex AI client (project={project_id})")
        elif api_key:
            self.client = genai.Client(api_key=api_key)
            print(f"  ✓ Gemini client with API key")
        else:
            raise ValueError("Provide api_key or (use_vertex + project_id)")

    def _build_batch_jsonl(self, judge_prompts, output_file):
        """Write JSONL in Gemini Batch API format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in judge_prompts:
                req = {
                    "custom_id": str(item["question_id"]),
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [{"text": item["judge_prompt"]}]
                            }
                        ],
                        "generation_config": {
                            "temperature": 0.0,
                            "max_output_tokens": 1024,
                        }
                    }
                }
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        print(f"  📄 JSONL written: {output_file} ({len(judge_prompts)} requests)")

    def submit_batch(self, judge_prompts, local_input_dir="batch_files"):
        """Submit batch job via File API (API key) or GCS (Vertex AI)."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(local_input_dir, exist_ok=True)
        local_file = os.path.join(local_input_dir, f"batch_input_{timestamp}.jsonl")

        print(f"\n{'='*80}")
        print(f"  📦 SUBMITTING NATIVE BATCH JOB")
        print(f"     Model:    {self.model_name}")
        print(f"     Prompts:  {len(judge_prompts)}")
        print(f"     Mode:     {'Vertex AI (GCS)' if self.use_vertex else 'API Key (File API)'}")
        print(f"{'='*80}\n")

        self._build_batch_jsonl(judge_prompts, local_file)

        if self.use_vertex:
            if not self.gcs_bucket:
                raise ValueError("--gcs-bucket required for Vertex AI.")
            from google.cloud import storage as gcs_storage
            gcs_client = gcs_storage.Client(project=self.project_id)
            bucket = gcs_client.bucket(self.gcs_bucket)
            gcs_in = f"{self.gcs_prefix}/{timestamp}/input.jsonl"
            gcs_out = f"{self.gcs_prefix}/{timestamp}/output/"
            bucket.blob(gcs_in).upload_from_filename(local_file)
            src_uri = f"gs://{self.gcs_bucket}/{gcs_in}"
            dst_uri = f"gs://{self.gcs_bucket}/{gcs_out}"
            print(f"  ☁️  GCS input:  {src_uri}")
            print(f"  ☁️  GCS output: {dst_uri}")
            batch_job = self.client.batches.create(
                model=self.model_name, src=src_uri, dest=dst_uri,
            )
        else:
            print(f"  📤 Uploading to Gemini File API...")
            uploaded_file = upload_file_with_mime(self.client, local_file)
            print(f"  ✅ Uploaded: {uploaded_file.name}")
            if hasattr(uploaded_file, 'uri'):
                print(f"     URI: {uploaded_file.uri}")

            batch_job = self.client.batches.create(
                model=self.model_name,
                src=uploaded_file.name,
            )

        job_name = batch_job.name
        print(f"\n  ✅ Batch job submitted!")
        print(f"     Job name: {job_name}")
        print(f"     State:    {batch_job.state}")
        return job_name

    def poll_until_complete(self, job_name, poll_interval=60, max_wait_minutes=120):
        """Poll until terminal state."""
        print(f"\n{'='*80}")
        print(f"  ⏳ POLLING: {job_name}")
        print(f"     Interval: {poll_interval}s | Max: {max_wait_minutes}m")
        print(f"{'='*80}\n")

        start = time.time()
        max_secs = max_wait_minutes * 60

        while True:
            elapsed = time.time() - start
            if elapsed > max_secs:
                print(f"\n  ⏰ TIMEOUT!")
                return None

            try:
                job = self.client.batches.get(name=job_name)
            except Exception as e:
                print(f"  ⚠ Poll error: {e}")
                time.sleep(poll_interval)
                continue

            state_str = str(getattr(job, 'state', 'UNKNOWN'))
            now_str = datetime.now().strftime("%H:%M:%S")
            mins = elapsed / 60

            progress = ""
            cs = getattr(job, 'completion_stats', None)
            if cs:
                s = getattr(cs, 'succeeded_count', '?')
                f_ = getattr(cs, 'failed_count', '?')
                t = getattr(cs, 'total_count', '?')
                progress = f" | {s}/{t} done, {f_} failed"

            print(f"  [{now_str}] ({mins:.1f}m) {state_str}{progress}")

            if "SUCCEEDED" in state_str:
                print(f"\n  ✅ COMPLETE!")
                return job
            elif "FAILED" in state_str:
                print(f"\n  ❌ FAILED!")
                if hasattr(job, 'error') and job.error:
                    print(f"     Error: {job.error}")
                return job
            elif "CANCELLED" in state_str:
                print(f"\n  🚫 CANCELLED!")
                return job

            time.sleep(poll_interval)

    def retrieve_results(self, job_name):
        """Retrieve results from completed job."""
        print(f"\n  📥 Retrieving: {job_name}")
        job = self.client.batches.get(name=job_name)
        print(f"     State: {getattr(job, 'state', '?')}")

        dest = getattr(job, 'dest', None)
        if dest:
            for attr in ['file_name', 'gcs_uri', 'inlined_responses']:
                val = getattr(dest, attr, None)
                if val:
                    if isinstance(val, list):
                        print(f"     dest.{attr}: {len(val)} items")
                    else:
                        print(f"     dest.{attr}: {val}")

        # Try inlined_responses
        if dest and getattr(dest, 'inlined_responses', None):
            r = list(dest.inlined_responses)
            print(f"  ✅ {len(r)} inlined responses")
            return r

        # Try output file
        if dest and getattr(dest, 'file_name', None):
            return self._download_api_file(dest.file_name)

        # Try GCS
        if dest and getattr(dest, 'gcs_uri', None):
            return self._download_gcs(dest.gcs_uri)

        # Dump for debug
        print(f"  ⚠ No results found. Job details:")
        for attr in dir(job):
            if not attr.startswith('_'):
                try:
                    val = getattr(job, attr)
                    if not callable(val) and val is not None:
                        print(f"    {attr}: {str(val)[:300]}")
                except:
                    pass
        return []

    def _download_api_file(self, file_name):
        """Download output from Gemini File API."""
        print(f"  📥 Downloading: {file_name}")
        try:
            file_info = self.client.files.get(name=file_name)
            uri = getattr(file_info, 'uri', None)
            print(f"     URI: {uri}")
            print(f"     State: {getattr(file_info, 'state', '?')}")

            if uri:
                import urllib.request
                # Try with API key
                for url in [f"{uri}?key={self.api_key}", uri]:
                    try:
                        data = urllib.request.urlopen(url).read().decode('utf-8')
                        results = [json.loads(l) for l in data.strip().split('\n') if l.strip()]
                        print(f"  ✅ {len(results)} results from file")
                        return results
                    except Exception:
                        continue

            # Try SDK download
            if hasattr(self.client.files, 'download'):
                content = self.client.files.download(name=file_name)
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                results = [json.loads(l) for l in content.strip().split('\n') if l.strip()]
                print(f"  ✅ {len(results)} results")
                return results

        except Exception as e:
            print(f"  ❌ Download error: {e}")
        return []

    def _download_gcs(self, gcs_uri):
        """Download from GCS."""
        from google.cloud import storage as gcs_storage
        print(f"  ☁️  Downloading: {gcs_uri}")
        parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name, prefix = parts[0], parts[1] if len(parts) > 1 else ""
        client = gcs_storage.Client(project=self.project_id)
        results = []
        for blob in client.bucket(bucket_name).list_blobs(prefix=prefix):
            if blob.name.endswith((".jsonl", ".json")):
                for line in blob.download_as_text().strip().split("\n"):
                    if line.strip():
                        results.append(json.loads(line))
        print(f"  ✅ {len(results)} results from GCS")
        return results

    def parse_batch_results(self, raw_results, judge_prompts, judge_name):
        """Parse batch results back to standard format using custom_id."""
        print(f"\n  🔍 Parsing {len(raw_results)} results...")
        lookup = {str(p["question_id"]): p for p in judge_prompts}
        parsed = []
        ok = 0
        fail = 0

        for idx, raw in enumerate(raw_results):
            cid = None
            text = None

            try:
                if isinstance(raw, dict):
                    cid = raw.get("custom_id")
                    resp = raw.get("response", raw)
                    if isinstance(resp, dict):
                        cands = resp.get("candidates", [])
                        if cands:
                            parts = cands[0].get("content", {}).get("parts", [])
                            if parts:
                                text = parts[0].get("text", "")
                        if not text:
                            text = resp.get("text")
                else:
                    cid = getattr(raw, 'custom_id', None)
                    resp = getattr(raw, 'response', raw)
                    if hasattr(resp, 'text'):
                        text = resp.text
                    elif hasattr(resp, 'candidates') and resp.candidates:
                        c = resp.candidates[0]
                        if hasattr(c, 'content') and c.content and c.content.parts:
                            text = c.content.parts[0].text
                    if not text and hasattr(raw, 'response'):
                        r2 = raw.response
                        if hasattr(r2, 'text'):
                            text = r2.text
            except Exception as e:
                print(f"    ⚠ Parse error #{idx}: {e}")

            if cid and str(cid) in lookup:
                orig = lookup[str(cid)]
                qid = str(cid)
            elif idx < len(judge_prompts):
                orig = judge_prompts[idx]
                qid = str(orig["question_id"])
            else:
                orig = {}
                qid = f"extra_{idx}"

            winner = extract_winner(text) if text else None
            parsed.append({
                "question_id": qid,
                "question": orig.get("question", ""),
                "answer_letter": orig.get("answer_letter", ""),
                "answer_full": orig.get("answer_full", ""),
                "judge_name": judge_name,
                "raw_response": text,
                "winner": winner,
            })
            if winner:
                ok += 1
            else:
                fail += 1

        returned = {str(r["question_id"]) for r in parsed}
        missing = 0
        for p in judge_prompts:
            qid = str(p["question_id"])
            if qid not in returned:
                missing += 1
                parsed.append({
                    "question_id": qid, "question": p.get("question", ""),
                    "answer_letter": p.get("answer_letter", ""),
                    "answer_full": p.get("answer_full", ""),
                    "judge_name": judge_name, "raw_response": None, "winner": None,
                })

        print(f"    ✅ {ok} with winner, {fail} without" + (f", {missing} missing" if missing else ""))
        return parsed

    def submit_and_wait(self, judge_prompts, judge_name,
                        poll_interval=60, max_wait_minutes=120,
                        local_input_dir="batch_files"):
        """Full pipeline: submit → poll → retrieve → parse."""
        job_name = self.submit_batch(judge_prompts, local_input_dir)

        job_info = {
            "job_name": job_name, "model": self.model_name,
            "num_prompts": len(judge_prompts),
            "submitted_at": datetime.now().isoformat(),
            "prompt_order": [str(p["question_id"]) for p in judge_prompts],
        }
        os.makedirs(local_input_dir, exist_ok=True)
        jf = os.path.join(local_input_dir, f"job_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(jf, 'w') as f:
            json.dump(job_info, f, indent=2)
        print(f"\n  💾 Job info: {jf}")
        print(f"     Retrieve later: --batch-retrieve '{job_name}'\n")

        completed = self.poll_until_complete(job_name, poll_interval, max_wait_minutes)
        if completed is None:
            print(f"\n  ⏰ Timed out. Use: --batch-retrieve '{job_name}'")
            return []

        state = str(getattr(completed, 'state', ''))
        if "SUCCEEDED" not in state:
            print(f"\n  ❌ State: {state}")
            return []

        raw = self.retrieve_results(job_name)
        if not raw:
            return []

        return self.parse_batch_results(raw, judge_prompts, judge_name)


class OpenAIJudge:
    def __init__(self, api_key, model_name="gpt-4.1-2025-04-14"):
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
                    temperature=0, max_completion_tokens=1024,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"\n  OpenAI attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** (attempt + 1))
                else:
                    return None


def run_judging(judge_prompts, judge_instance, judge_name, output_dir, rate_limit_delay=1.0):
    results = []
    total = len(judge_prompts)
    print(f"\n{'='*80}")
    print(f"Running {judge_name} on {total} prompts (sequential)")
    print(f"{'='*80}\n")
    for i, item in enumerate(judge_prompts):
        qid = item["question_id"]
        print(f"[{i+1}/{total}] {qid} ...", end=" ", flush=True)
        raw = judge_instance.judge(item["judge_prompt"])
        winner = extract_winner(raw) if raw else None
        results.append({
            "question_id": qid, "question": item.get("question", ""),
            "answer_letter": item.get("answer_letter", ""),
            "answer_full": item.get("answer_full", ""),
            "judge_name": judge_name, "raw_response": raw, "winner": winner,
        })
        print(f"Winner: {winner}" if winner else "Could not parse")
        if i < total - 1:
            time.sleep(rate_limit_delay)
    return results


def calculate_statistics(results, judge_name):
    winners = [r["winner"] for r in results if r["winner"] is not None]
    total = len(results)
    parsed = len(winners)
    c = Counter(winners)
    m1, m2 = c.get(1, 0), c.get(2, 0)
    m1p = (m1 / parsed * 100) if parsed else 0
    m2p = (m2 / parsed * 100) if parsed else 0
    stats = {
        "judge_name": judge_name, "total_questions": total,
        "total_parsed": parsed, "unparsed": total - parsed,
        "model1_wins": m1, "model2_wins": m2,
        "model1_win_rate": round(m1p, 2), "model2_win_rate": round(m2p, 2),
    }
    print(f"\n{'='*60}")
    print(f"  RESULTS: {judge_name}")
    print(f"  Judged: {total} | Parsed: {parsed} | Unparsed: {total-parsed}")
    print(f"  Model 1: {m1} ({m1p:.2f}%) | Model 2: {m2} ({m2p:.2f}%)")
    print(f"{'='*60}\n")
    return stats


def save_results(results, stats, judge_name, output_dir, is_test=False):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    px = "TEST_" if is_test else ""

    rf = os.path.join(output_dir, f"{px}{judge_name}_raw_{ts}.jsonl")
    with open(rf, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Raw: {rf}")

    sf = os.path.join(output_dir, f"{px}{judge_name}_stats_{ts}.json")
    with open(sf, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Stats: {sf}")

    sm = os.path.join(output_dir, f"{px}{judge_name}_summary_{ts}.txt")
    with open(sm, 'w', encoding='utf-8') as f:
        f.write(f"Judge: {judge_name}\nMode: {'TEST' if is_test else 'FULL'}\n{'='*60}\n\n")
        f.write(f"Total: {stats['total_questions']} | Parsed: {stats['total_parsed']} | Unparsed: {stats['unparsed']}\n")
        f.write(f"Model 1: {stats['model1_wins']} ({stats['model1_win_rate']}%)\n")
        f.write(f"Model 2: {stats['model2_wins']} ({stats['model2_win_rate']}%)\n\n")
        for r in results:
            q = r.get('question', '')
            f.write(f"ID: {r['question_id']} | Winner: {r['winner']}\n")
            f.write(f"Q: {q[:100]}{'...' if len(q)>100 else ''}\n")
            f.write(f"Response:\n{r.get('raw_response', 'N/A')}\n{'─'*40}\n")
    print(f"  Summary: {sm}")


def parse_args():
    p = argparse.ArgumentParser(
        description="LLM Judge with Gemini Native Batch API support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python LLM_as_a_judge_batch.py --test --test-count 10 --batch
  python LLM_as_a_judge_batch.py --gemini --batch
  python LLM_as_a_judge_batch.py --gemini --batch --batch-submit-only
  python LLM_as_a_judge_batch.py --batch-retrieve "batches/xxxx"
  python LLM_as_a_judge_batch.py --batch-list
  python LLM_as_a_judge_batch.py --test                 # sequential
        """
    )
    p.add_argument("--test", action="store_true")
    p.add_argument("--test-count", type=int, default=5)
    p.add_argument("--gemini", action="store_true")
    p.add_argument("--openai", action="store_true")
    p.add_argument("--gemini-model", default="gemini-3-flash-preview")
    p.add_argument("--openai-model", default="gpt-4.1-2025-04-14")
    p.add_argument("--input", default="results/VMME_Description/judge_prompts.jsonl")
    p.add_argument("--output", default="results/VMME_Description/judge_outputs")
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--env-file", default=".env")

    b = p.add_argument_group("Batch")
    b.add_argument("--batch", action="store_true", help="Use native Batch API")
    b.add_argument("--batch-submit-only", action="store_true")
    b.add_argument("--batch-retrieve", default=None)
    b.add_argument("--batch-poll-interval", type=int, default=60)
    b.add_argument("--batch-max-wait", type=int, default=120)
    b.add_argument("--batch-list", action="store_true")

    g = p.add_argument_group("GCP")
    g.add_argument("--use-vertex", action="store_true")
    g.add_argument("--gcp-project", default=None)
    g.add_argument("--gcp-location", default="us-central1")
    g.add_argument("--gcs-bucket", default=None)
    g.add_argument("--gcs-prefix", default="batch_jobs")

    a = p.parse_args()
    if not a.gemini and not a.openai and not a.batch_retrieve and not a.batch_list:
        a.gemini = True
    return a


def main():
    args = parse_args()

    # Load .env
    sd = os.path.dirname(os.path.abspath(__file__))
    for path in [args.env_file, os.path.join(os.getcwd(), ".env"), os.path.join(sd, ".env")]:
        if os.path.exists(path):
            load_dotenv(dotenv_path=path, override=True)
            print(f"✓ .env: {os.path.abspath(path)}")
            with open(path) as f:
                for l in f:
                    l = l.strip()
                    if l and not l.startswith('#'):
                        print(f"    - {l.split('=')[0].strip()} = ********...")
            break

    GK = os.environ.get("GEMINI_API_KEY")
    OK = os.environ.get("OPENAI_API_KEY")
    GP = args.gcp_project or os.environ.get("GOOGLE_CLOUD_PROJECT")

    print(f"\n  GEMINI_API_KEY: {repr(GK[:20])+'...' if GK else 'None'}")
    print(f"  OPENAI_API_KEY: {repr(OK[:20])+'...' if OK else 'None'}")

    try:
        import google.genai as gm
        print(f"  SDK: google-genai {getattr(gm, '__version__', '?')}")
    except:
        pass

    print(f"\n{'#'*70}")
    print(f"  CONFIG")
    print(f"{'#'*70}")
    print(f"  Mode:    {'TEST({})'.format(args.test_count) if args.test else 'FULL'}")
    print(f"  Input:   {args.input}")
    print(f"  Output:  {args.output}")
    print(f"  Gemini:  {args.gemini} ({args.gemini_model})")
    print(f"  OpenAI:  {args.openai} ({args.openai_model})")
    print(f"  Batch:   {'✓' if args.batch else '✗'}")
    if args.batch:
        print(f"  Submit only: {args.batch_submit_only}")
        print(f"  Poll: {args.batch_poll_interval}s | Max: {args.batch_max_wait}m")
    print(f"  Keys:    G={'✓' if GK else '✗'} O={'✓' if OK else '✗'}")
    print(f"{'#'*70}\n")

    # --batch-list
    if args.batch_list:
        print("📋 Batch jobs:\n")
        if not GK:
            print("❌ Need GEMINI_API_KEY"); return
        c = genai.Client(api_key=GK)
        n = 0
        for j in c.batches.list():
            print(f"  {getattr(j,'name','?')} | {getattr(j,'state','?')} | {getattr(j,'model','?')}")
            n += 1
        print(f"\n  Total: {n}" if n else "  None.")
        return

    # --batch-retrieve
    if args.batch_retrieve:
        print(f"📥 Retrieve: {args.batch_retrieve}\n")
        if not GK:
            print("❌ Need key"); return
        bj = GeminiBatchJudge(api_key=GK, model_name=args.gemini_model,
                              gcs_bucket=args.gcs_bucket, project_id=GP, use_vertex=args.use_vertex)
        j = bj.client.batches.get(name=args.batch_retrieve)
        st = str(getattr(j, 'state', ''))
        print(f"  State: {st}")
        if "SUCCEEDED" not in st:
            print("  ⚠ Not done."); return
        prompts = load_jsonl(args.input)
        if args.test:
            prompts = prompts[:args.test_count]
        label = f"gemini_{args.gemini_model.replace('-','_').replace('.','_')}"
        raw = bj.retrieve_results(args.batch_retrieve)
        if not raw:
            print("  ⚠ No results."); return
        parsed = bj.parse_batch_results(raw, prompts, label)
        stats = calculate_statistics(parsed, label)
        save_results(parsed, stats, label, args.output, args.test)
        return

    # Load prompts
    print(f"Loading: {args.input}")
    prompts = load_jsonl(args.input)
    print(f"Loaded {len(prompts)}")
    if args.test:
        prompts = prompts[:args.test_count]
        print(f"TEST: {len(prompts)} prompts\n")
    else:
        print(f"FULL: {len(prompts)} prompts\n")

    all_stats = {}

    # Gemini
    if args.gemini:
        if not GK and not (args.use_vertex and GP):
            print("⚠ No Gemini key.")
        else:
            label = f"gemini_{args.gemini_model.replace('-','_').replace('.','_')}"

            if args.batch:
                print(f"📦 BATCH MODE\n")
                bj = GeminiBatchJudge(
                    api_key=GK, model_name=args.gemini_model,
                    gcs_bucket=args.gcs_bucket, gcs_prefix=args.gcs_prefix,
                    project_id=GP, location=args.gcp_location,
                    use_vertex=args.use_vertex,
                )
                if args.batch_submit_only:
                    jn = bj.submit_batch(prompts)
                    print(f"\n  📦 Submitted: {jn}")
                    print(f"  Retrieve: --batch-retrieve '{jn}'")
                    return
                results = bj.submit_and_wait(
                    prompts, label,
                    poll_interval=args.batch_poll_interval,
                    max_wait_minutes=args.batch_max_wait,
                )
                if not results:
                    return
            else:
                print(f"📝 SEQUENTIAL\n")
                gj = GeminiJudge(api_key=GK, model_name=args.gemini_model)
                results = run_judging(prompts, gj, label, args.output, args.delay)

            s = calculate_statistics(results, label)
            save_results(results, s, label, args.output, args.test)
            all_stats["gemini"] = s

    # OpenAI
    if args.openai:
        if not OK:
            print("⚠ No OpenAI key.")
        else:
            label = f"openai_{args.openai_model.replace('-','_')}"
            oj = OpenAIJudge(api_key=OK, model_name=args.openai_model)
            results = run_judging(prompts, oj, label, args.output, args.delay)
            s = calculate_statistics(results, label)
            save_results(results, s, label, args.output, args.test)
            all_stats["openai"] = s

    if all_stats:
        print(f"\n{'#'*70}")
        print(f"  SUMMARY {'(TEST)' if args.test else ''}")
        print(f"{'#'*70}")
        for s in all_stats.values():
            print(f"  {s['judge_name']}: M1={s['model1_wins']}({s['model1_win_rate']}%) "
                  f"M2={s['model2_wins']}({s['model2_win_rate']}%) Unparsed={s['unparsed']}")
        os.makedirs(args.output, exist_ok=True)
        cf = os.path.join(args.output, f"{'TEST_' if args.test else ''}combined.json")
        with open(cf, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"  Saved: {cf}")

    print("\nDone!")


if __name__ == "__main__":
    main()