import os
import json
from dotenv import load_dotenv
from google import genai

# 1. Load the .env file
load_dotenv()

# 2. Get the API key from environment
# (The SDK often looks for GEMINI_API_KEY or GOOGLE_API_KEY automatically)
api_key = os.getenv("GEMINI_API_KEY")

# 3. Initialize the client
client = genai.Client(api_key=api_key)

# Your specific job name
job_name = "batches/tw91gvzztfaf30m7cjvyklgoex8q8efhdroa" 

# 4. Get the latest status
job = client.batches.get(name=job_name)

print(f"Job State: {job.state}")

if job.state == "SUCCEEDED":
    # If the job succeeded, job.output will contain the name of the result file
    # This result is typically a JSONL file stored in the File API
    output_file = job.output
    print(f"Results are ready! Output file name: {output_file}")
    
    # Optional: Download and read the first few results
    # content = client.files.download(name=output_file)
    # print("Downloaded results successfully.")

elif job.state == "FAILED":
    print(f"Job failed. Error: {job.error}")
    
elif job.state in ["PENDING", "RUNNING"]:
    print(f"Job is still {job.state.lower()}. Please check back in a few hours.")
    
else:
    print(f"Current Status: {job.state}")