import json
import math
# Load from file
with open('data_mock.json', 'r') as f:
    data = json.load(f)

# Filter
data_original = [{'time': item['time'], 'consume': item['consume']} for item in data]
filtered_data = [{'time': item['time'], 'consume': item['consume'], 'billing': round(item['consume'] * 1.352, 2)} for item in data]

print(data)