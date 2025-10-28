import json
with open("data/best_kf_params_top5.json", "r") as f:
    data = json.load(f)

print("🔑 Keys in file:")
for k in data.keys():
    print(k)