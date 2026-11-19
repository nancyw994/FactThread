# Article MVP regression set

These four short articles were used to diagnose and repair extraction and retrieval defects. Keep them as a regression set; do not describe them as an independent evaluation set or tune new rules solely to their wording.

- `articles.json`: saved article-mode inputs and source URLs.
- `gold_cases.json`: three manually selected factual checkpoints per article. This is a targeted set, not exhaustive claim annotation.
- `before/` and `after/`: retrieval-only runs (`judge="none"`); the first after run is retained separately because the configured search endpoint returned transient errors.
- `retrieval_comparison.json`: query, candidate page, and matched passage comparison, including source-page location.
- `extraction_comparison.json`: counts, statuses, provider restrictions, and Apollo control result.
- `diagnostics/attempt1_request_comparison.json`: attempt1's reconstructed SDK parameters and the explicit missing HTTP body/status fields.
- `diagnostics/request_success_failure_comparison.json`: same-configuration short format probe versus Chan quota failure, including raw reply pointer, finish reason, usage, parsed/filtered counts, and full sanitized 429 body.
- `after/sequenced_extractions.json`: ordered request events from the stopped Chan attempt. `diagnostics/run_sequential.py` is prepared to run one same-configuration Apollo format probe before Chan Chan, then proceed through the remaining regression articles only while runs complete.
- `diagnostics/raw/`: raw model replies used to establish the parser failure and successful Apollo control. Request failures have no model reply.

For an independent evaluation after the next successful regression run, choose new short articles from different topics and freeze their annotations before comparing outputs.
