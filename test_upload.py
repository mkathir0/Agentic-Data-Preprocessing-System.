import requests

url = "http://localhost:8000/api/jobs/upload"
headers = {"X-API-Key": "default_secret_key"}

files = [
    ("files", ("test1.csv", "name,age\nAlice,30\nBob,25", "text/csv")),
]

response = requests.post(url, headers=headers, files=files)
print(response.status_code)
print(response.text)
