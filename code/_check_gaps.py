import pandas as pd

r = pd.read_csv("data/q3_relay_job_options.csv")
p = pd.read_csv("data/q3_pattern_comm_gaps.csv")

rgaps = set(r["gap_id"])
pgaps = set(p["gap_id"])

print(f"Relay gaps: {len(rgaps)}, first: {sorted(rgaps)[0]}")
print(f"Pattern gaps: {len(pgaps)}, first: {sorted(pgaps)[0]}")
print(f"Relay gaps IN pattern gaps: {len(rgaps & pgaps)}")
print(f"Relay gaps NOT in pattern gaps: {len(rgaps - pgaps)}")
print(f"Pattern gaps NOT in relay: {len(pgaps - rgaps)}")


t = pd.read_csv("data/q3_task_comm_gaps.csv")
tgaps = set(t["gap_id"])
print(f"\nTask gaps: {len(tgaps)}")
print(f"Relay gaps in task gaps: {len(rgaps & tgaps)}")
print(f"Task gaps not in relay: {len(tgaps - rgaps)}")