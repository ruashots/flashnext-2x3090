#!/usr/bin/env python3
"""Grade the objectively checkable items in the quality set.

Only items with a single defensible right answer are graded here. Everything
else goes to a blind reader. Each check states what it looks for, so a stranger
can disagree with the check itself rather than with a score.
"""
import json, re, sys, collections

def words(line):
    return len([w for w in re.split(r"\s+", line.strip()) if w])

def check_instr1(ans):
    """Exactly three lines, of 5, 7 and 5 words."""
    lines = [l for l in ans.strip().splitlines() if l.strip()]
    if len(lines) != 3:
        return False, f"{len(lines)} lines, wanted 3"
    counts = [words(l) for l in lines]
    return counts == [5, 7, 5], f"word counts {counts}, wanted [5, 7, 5]"

def check_instr2(ans):
    """First 10 primes, their sum 129, product of the first four 210."""
    nums = re.findall(r"\d+", ans.replace(",", " "))
    ok_sum = "129" in nums
    ok_prod = "210" in nums
    primes = ["2","3","5","7","11","13","17","19","23","29"]
    ok_list = all(p in nums for p in primes)
    return (ok_sum and ok_prod and ok_list,
            f"list={ok_list} sum129={ok_sum} product210={ok_prod}")

def check_math1(ans):
    """x = -(2 ln5 + ln3) / (2 ln3 - ln5) = -7.3453 to four places."""
    hits = re.findall(r"-?\d+\.\d{2,}", ans)
    good = [h for h in hits if abs(float(h) - (-7.3453)) < 0.005]
    return bool(good), f"found {hits[:6]}, wanted about -7.3453"

def check_reason2(ans):
    """Posterior is 0.0098, that is 0.98 percent."""
    hits = re.findall(r"-?\d+\.?\d*", ans.replace("%", " % "))
    vals = [float(h) for h in hits if h not in ("", ".", "-")]
    good = any(abs(v - 0.98) < 0.06 for v in vals) or any(abs(v - 0.0098) < 0.0006 for v in vals)
    return good, "wanted 0.98 percent (0.0098)"

def check_reason3(ans):
    """Must state that two weighings cannot cover 10, since 3^2 = 9 < 10."""
    low = ans.lower()
    has_nine = "9" in ans or "nine" in low
    has_reason = ("3^2" in ans or "3²" in ans or "three weighings" in low
                  or "9 " in ans or "nine outcomes" in low)
    return has_nine and has_reason, "wanted the 9 outcomes versus 10 balls argument"

def check_edge1(ans):
    """The text to summarise is empty. The answer must say so, not invent one."""
    low = ans.lower()
    said = any(k in low for k in ["empty", "no text", "nothing", "blank",
                                  "not provided", "cannot", "can not", "missing"])
    return said, "wanted an explicit statement that there is no text"

CHECKS = {"instr-1": check_instr1, "instr-2": check_instr2, "math-1": check_math1,
          "reason-2": check_reason2, "reason-3": check_reason3, "edge-1": check_edge1}

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = collections.defaultdict(dict)
for r in rows:
    by[r["label"]][r["id"]] = r

print(f"{'item':10s} " + " ".join(f"{l[:26]:28s}" for l in sorted(by)))
print("-" * (11 + 29 * len(by)))
score = collections.Counter()
for pid, fn in CHECKS.items():
    cells = []
    for lab in sorted(by):
        r = by[lab].get(pid)
        if not r:
            cells.append(f"{'no answer':28s}"); continue
        ok, why = fn(r.get("answer") or "")
        score[lab] += 1 if ok else 0
        cells.append(f"{('PASS' if ok else 'FAIL') + '  ' + why:28.28s}")
    print(f"{pid:10s} " + " ".join(cells))
print("-" * (11 + 29 * len(by)))
print(f"{'total':10s} " + " ".join(f"{str(score[l]) + '/' + str(len(CHECKS)):28s}" for l in sorted(by)))
