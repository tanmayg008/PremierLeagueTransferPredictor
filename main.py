import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("FOOTBALL_API_KEY")

headers = { "X-Auth-Key": API_KEY }

url="https://api.football-data.org/v4/competitions/PL"

response = requests.get(url, headers=headers)
print(response.status_code)
print(response.json())
data = response.json()

print("Status Code:", response.status_code)

if response.status_code == 200:
    for team in data["teams"]:
        print(team["name"])
else:
    print(data["message"])