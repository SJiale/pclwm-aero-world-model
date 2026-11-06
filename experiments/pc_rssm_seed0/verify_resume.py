import json
from pathlib import Path

p = Path(r"C:\pclwm\pclwm_pcrssm\results\multiseed_results.json")
d = json.loads(p.read_text(encoding="utf-8"))
CFGS = ["E0_baseline", "E1_static", "E2_dynamic", "E3_both"]

print("=== JSON 已有 seed ===")
for k in sorted(d.keys()):
    have = set(d[k].get("cfgs", {}).keys())
    skip = set(CFGS) <= have
    print("  %s: %d/4  complete=%s  cfgs=%s" % (k, len(have), skip, sorted(have)))

print()
print("=== resume 预期行为 ===")
for s in [0, 1, 2, 3, 4]:
    k = "seed%d" % s
    if k in d:
        have = set(d[k].get("cfgs", {}).keys())
        if set(CFGS) <= have:
            print("  seed%d: 跳过（4/4 完成）" % s)
            continue
    print("  seed%d: 训练 4 个配置" % s)
