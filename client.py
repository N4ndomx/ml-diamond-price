
import requests
body = {
    "carat": 0.53,
    "cut": 2.0,
    "color": 3.2,
    "clarity":6.0,
    "depth": 61.8,
    "table": 56.0,
    "x": 5.19,
    "y": 5.24,
    "z": 3.22
    }
response = requests.post(url = 'http://127.0.0.1:8000/score',
              json = body)
print (response.json())
# output: {'score': 0.866490130600765}

