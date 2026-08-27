#!/usr/bin/env python3
"""Collect answers to a fixed prompt set from an OpenAI-compatible server."""
import argparse, json, time, uuid, urllib.request

PROMPTS = [
 ("code-1", "Write a Python function `chunk_by_tokens(text, max_tokens, tokenizer)` that splits text into chunks of at most max_tokens, never splitting a sentence across chunks unless a single sentence exceeds the limit. Return a list of strings. Include three unit tests."),
 ("code-2", "This SQL is slow on a 40 million row table:\nSELECT u.id, u.email, COUNT(o.id) AS orders FROM users u LEFT JOIN orders o ON o.user_id = u.id WHERE u.created_at > '2025-01-01' GROUP BY u.id, u.email ORDER BY orders DESC LIMIT 50;\nName the likely causes and rewrite it. State which indexes you would add."),
 ("code-3", "In Python, this raises `RuntimeError: dictionary changed size during iteration`:\nfor k in cache: \n    if cache[k].expired: del cache[k]\nGive two correct fixes and say when each is preferable."),
 ("code-4", "Write a bash script that finds every git repository under a directory, reports the ones with uncommitted changes, and exits non-zero if any are found. Handle paths with spaces."),
 ("code-5", "Design the data model for a bookings system where a resource can be booked in half-hour slots, bookings can repeat weekly, and exceptions to a repeat can be cancelled individually. Give the tables, the keys, and the query that lists a resource's occupied slots for one week."),
 ("agent-1", "You are an agent with three tools: read_file(path), list_dir(path), write_file(path, content). A user says: 'the deploy script is failing on the staging box, fix it'. List the exact tool calls you would make in order, and what you would do at each branch. Do not assume the file layout."),
 ("agent-2", "A long running job writes progress to a log file. You must report when it finishes, but the job sometimes dies without writing a final line. Describe a monitoring approach that never reports a false completion and never waits forever."),
 ("reason-1", "Three switches outside a windowless room control three bulbs inside. You may enter the room once. How do you determine which switch controls which bulb? Then explain why the standard answer fails if the bulbs are LED."),
 ("reason-2", "A test for a disease has 99% sensitivity and 95% specificity. The disease affects 1 in 2000 people. A person tests positive. What is the probability they have the disease? Show the arithmetic."),
 ("reason-3", "You have 8 balls, one is heavier. Using a balance scale twice, find it. Now prove that two weighings cannot suffice for 10 balls."),
 ("write-1", "Write four sentences explaining to a non-technical reader why running a large language model on your own computer is slower than using a cloud service, without using the words 'inference', 'parameters', or 'bandwidth'."),
 ("write-2", "Escribe en espanol un mensaje corto para un cliente explicando que su pedido llega dos dias tarde por un problema en la aduana, sin sonar automatico ni pedir disculpas dos veces."),
 ("instr-1", "Reply with exactly three lines. Line one must be 5 words. Line two must be 7 words. Line three must be 5 words. The topic is a server room at night. Do not add anything else."),
 ("instr-2", "List the first 10 prime numbers, then the sum of those primes, then the product of the first four of them. Give only three lines of output, each starting with a label."),
 ("know-1", "Explain the difference between a mixture of experts model activating 6 billion parameters per token and a dense 6 billion parameter model. Say what each is better at and why they are not equivalent."),
 ("know-2", "What does an NVMe drive's random read latency have to do with running a model whose weights do not fit in RAM? Explain the mechanism, not just the conclusion."),
 ("edge-1", "Summarise the following in one sentence: ''. If you cannot, say why."),
 ("edge-2", "A user asks you to delete all files in their home directory older than 30 days. Write the command and state, before it, the single most important thing they must check first."),
 ("math-1", "Solve for x: 3^(2x+1) = 5^(x-2). Give the exact expression and a decimal to four places."),
 ("math-2", "A rope hangs between two poles 20 m apart, both 10 m tall. The lowest point of the rope is 5 m above the ground. Set up the equation describing the rope and state what quantity you would solve for numerically."),
]

def ask(url, model, prompt, max_tokens, timeout=1800):
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.7, "top_p": 0.8, "stream": False,
            # non-thinking mode: both models answer directly, so the comparison is of the
            # answer and not of how long each model chooses to deliberate
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer none"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        obj = json.load(r)
    dt = time.perf_counter() - t0
    msg = obj["choices"][0]["message"]
    return {"answer": msg.get("content") or "",
            "thinking": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "usage": obj.get("usage"), "seconds": round(dt, 2),
            "finish_reason": obj["choices"][0].get("finish_reason")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1600)
    a = ap.parse_args()
    with open(a.out, "a") as fh:
        for pid, prompt in PROMPTS:
            try:
                res = ask(a.url, a.model, prompt, a.max_tokens)
            except Exception as e:
                res = {"answer": "", "thinking": "", "error": str(e), "seconds": None}
            rec = {"label": a.label, "model": a.model, "id": pid, "prompt": prompt, **res}
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            n = len(res.get("answer") or "")
            print(f"{pid}: {res.get('seconds')}s  {n} chars  {res.get('finish_reason')}", flush=True)

if __name__ == "__main__":
    main()
