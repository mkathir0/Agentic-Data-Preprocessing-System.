import httpx
import time
import os
import sys

API_URL = "http://127.0.0.1:8000/api/jobs"
HEADERS = {"X-API-Key": "default_secret_key"}

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python verify_backend.py <path_to_csv>")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' does not exist.")
        sys.exit(1)

    print(f"Uploading {csv_file}...")
    with open(csv_file, "rb") as f:
        # Use only the filename for the upload field, not the full path
        filename = os.path.basename(csv_file)
        files = {"file": (filename, f, "text/csv")}
        response = httpx.post(f"{API_URL}/upload", files=files, headers=HEADERS)
        
    if response.status_code != 200:
        print(f"Upload failed: {response.text}")
        return
        
    job_data = response.json()
    job_id = job_data["id"]
    print(f"Upload successful. Job ID: {job_id}")

    # Polling for status
    print("Polling job status...")
    while True:
        response = httpx.get(f"{API_URL}/{job_id}", headers=HEADERS)
        if response.status_code != 200:
            print(f"Failed to get status: {response.text}")
            break
            
        status_data = response.json()
        status = status_data["status"]
        step = status_data.get("current_step", "N/A")
        progress = status_data.get("progress", 0)
        
        print(f"Status: {status} | Step: {step} | Progress: {progress}%")
        
        if status == "COMPLETED":
            print("Job finished successfully!")
            break
        elif status == "FAILED":
            print("Job failed.")
            break
            
        time.sleep(2)
        
    # Download the cleaned CSV
    print("Downloading cleaned CSV...")
    response = httpx.get(f"{API_URL}/{job_id}/download", headers=HEADERS)
    if response.status_code == 200:
        output_filename = f"downloaded_cleaned_{filename}"
        with open(output_filename, "wb") as f:
            f.write(response.content)
        print(f"Successfully downloaded cleaned.csv to {output_filename}")
    else:
        print(f"Download failed: {response.text}")

if __name__ == "__main__":
    main()
