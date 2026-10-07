# Engine-timed figures behind the 0.1.40.1 -> 0.1.40.2 upgrade report

`strata-engine-timing-2026-10-07.jsonl` holds every per-request timing line that Strata's engine wrote during two
server runs on the same box and the same config (262K, two at once, split at layer 26, weight trimming, image cap 1024):

- 0.1.40.1: engine started 2026-10-06 12:29:05, run ended 12:44:52. Its engine reports itself as 0.1.40, because 0.1.40.1 changed only the Python server.
- 0.1.40.2: engine started 2026-10-07 10:56:15, run ended 11:09:40.

Each run covers that version's speed test with this repo's harness (bench.py, then the image and two-at-once probes). Requests from the server's other clients in the same window are included. The source is Strata's engine log, `/opt/strata/strata-orca-iq4xs.log` on the server.

The figures come from `scripts/export_engine_timing.py <log> "<engine start time>" <label>`, and its medians match the report:

| | 0.1.40.1 | 0.1.40.2 |
|---|---|---|
| decode time per verify step, 4K prompts | 23.12 ms | 21.93 ms |
| decode time per verify step, 60K prompts | 23.95 ms | 24.18 ms |
| prompt reading, 4K prompts | 651 tok/s | 797 tok/s |
| prompt reading, 60K prompts | 2,171 tok/s | 2,338 tok/s |

How they are counted:

- The step time is the generation time divided by (generated tokens minus accepted drafts). Only requests that generated at least 100 tokens count; that is 10 requests at 4K and 7 at 60K in each run.
- Prompt reading is the tokens read divided by the read time. Only requests that read more than 1,000 tokens count; that is 12 or 13 at 4K and 9 at 60K.
- Each figure is one run per version on a live server, so treat differences of a few percent as within noise.
