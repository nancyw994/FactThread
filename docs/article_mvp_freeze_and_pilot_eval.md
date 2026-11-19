# Article MVP freeze and pilot evaluation

**Date:** 2026-11-18  
**Status:** Feature scope frozen; defect fixes applied; extraction post-fix verification is blocked by Groq rate limiting.

## Freeze decision

Keep the current article MVP feature scope frozen. This round adds no product capabilities; it fixes extraction failure reporting and query construction based on the pilot. Do not reopen feature work before the next independent evaluation.

The working tree is not a release checkpoint: changes remain uncommitted, and no tag was created.

## Evaluation set

Four short, previously unused articles were selected across distinct topics. Their URLs and topics were checked against the repository's existing development examples and run records; no matching title or subject was found.

- [Chan Chan mausoleum discovery — archaeology](https://www.smithsonianmag.com/smart-news/unsealed-for-the-first-time-in-600-years-this-tomb-in-peru-was-just-declared-the-discovery-of-the-decade-180989497/) — 5,234 input characters.
- [Northeast Pacific marine heatwaves — climate](https://psl.noaa.gov/news/2026/nepmarineheatwaves.html) — 6,885 characters.
- [Lucy flyby of Donaldjohanson — astronomy](https://science.nasa.gov/missions/lucy/nasas-lucy-reveals-wobbling-peanut-shaped-asteroid/) — 7,045 characters.
- [CDC dengue update — public health](https://www.cdc.gov/han/php/notices/han00523.html) — 11,623 characters.

Each article is within the 12,000-character article-mode input limit. Three high-salience, factual reference claims were manually selected from each article (12 total). This is a targeted checkpoint set, not an exhaustive annotation of every checkable claim in each article.

## Extraction root cause and repair

The failed article was Chan Chan, replayed through `ArticleBook.start_extract`. The submitted text was 5,234 characters and was not truncated. The only configured provider was Groq `openai/gpt-oss-120b`; paid and OpenRouter access were both disabled.

The 14,711-character raw reply was a numbered explanation and quote list that repeated article text, then stopped before producing a JSON object. The old parser searched from the first `{` to the last `}` and returned `{"claims":[]}` when that span could not be parsed. The extraction filter consequently saw 0 parsed items, 0 accepted items, and 0 rejected items. Because there were no rejection reasons, it retried with “no claims were returned”; the second malformed reply was again converted to an empty object. Article mode then marked the run `extracted` with no candidates. This was a parse failure misreported as successful empty output, not a valid empty result or a filtering result.

The Apollo control text (203 characters) previously returned a 1,193-character JSON reply; all four parsed claims passed quote validation. This confirms the current parser accepted valid JSON, while the long article response shape triggered the silent failure.

The parser now raises an extraction parse error when no complete JSON object with a `claims` list exists. A valid `{"claims":[]}` is a legal empty result and does not trigger a repair retry. Request errors, parse errors, valid empty results, and all-filtered results have distinct statuses. Each filter attempt records parsed, accepted, rejected counts, and rejection reasons. Article mode exposes a failure instead of a success-empty result. An earlier strict-schema request returned `BadRequest`; article extraction uses JSON Object Mode and local quote/schema checks. Groq documents JSON Object Mode for GPT-OSS 120B ([Groq structured outputs](https://console.groq.com/docs/structured-outputs)).

An earlier post-fix batch replay attempted all four articles and received rate limits. The later sequential run below supersedes its attempt order: it first passed a short format probe, then stopped at Chan Chan on a quota error. The pre-fix batch had 0/12 selected gold claims present; precision and fidelity were not measurable because it emitted no claims.

### Sequenced verification attempt

The short Apollo format probe passed with the current configuration: raw response 953 characters, `finish_reason="stop"`, 1,376 prompt tokens and 740 completion tokens, 4 parsed, 4 accepted and 0 rejected. Its output cap was 4,096 tokens. Chan Chan then used the same model, temperature, response format and output cap, but received HTTP 429 with `error.type="tokens"` and `error.code="rate_limit_exceeded"`. The complete sanitized body reports a daily token limit of 200,000, 197,892 used, and 2,902 requested; retry-after was 344 seconds. The runner stopped at Chan Chan; it made no request for heatwave, Lucy, or dengue.

The saved attempt1 record does not contain the numeric HTTP status, provider response body, or API `error.type` / `error.code`. It contains only the wrapper message and provider exception class `BadRequestError`; that class is not sufficient evidence to assign an HTTP status or cause. The actual SDK request arguments were reconstructed from the saved input and request-building code: Groq `openai/gpt-oss-120b`, temperature `0.0`, `response_format={"type":"json_object"}`, `max_completion_tokens=4096`, plus the exact system/user messages. The article was 5,234 characters (under the application’s 12,000-character limit); provider tokenization was not returned. The new short probe confirms the same request settings are accepted for short input. It does not identify why the earlier Chan attempt received `BadRequest`; input-specific validation versus another provider-side cause remains unknown because the body is missing. No parameter was changed based on that incomplete record.

Future provider failures now preserve a sanitized HTTP status, complete available error body, API error type/code, and request arguments, excluding credentials. The attempt1 provider body cannot be recovered retroactively.

The regression-article completion rate is **0/4**: Chan Chan had a request failure, and heatwave, Lucy, and dengue were not attempted. Since no regression article completed, reference recall, extra-item precision, and normalization fidelity are not measurable on completed articles. The short format probe is reported separately, not included in the 4-article completion denominator. Success/failure parameters and results are saved in `data/dev/article_mvp_regression/diagnostics/request_success_failure_comparison.json`; per-request records and raw probe output are in `data/dev/article_mvp_regression/after/sequenced_extractions.json` and `after/raw/reply-01.txt`. Failed requests have no finish reason, usage, raw model reply, parse result, or filter counts; these are recorded as unavailable rather than zero.

## Retrieval query repair and coverage

To isolate retrieval from extraction, the 12 manually selected claims were passed to the existing search-and-fetch pipeline. The verdict model was disabled, so this section reports retrieved passages only and makes no verdict-accuracy claim.

- Before the repair, number-heavy claims generated queries such as `38`, `4 16`, and `650 20, 2025`. The query builder now emits a compact form containing all non-stopword terms and the complete claim wording. This retains entities, predicates, quantities, measures, and dates without sentence-specific rules.
- Before: 15 queries (14 successful, 1 failed), 60 candidate pages, 36 selected and fetched pages, 5 passages across 4/12 claims. Two claims had a passage from a non-input page.
- After: 18 queries (9 successful, 9 failed), 45 candidate pages, 27 selected and fetched pages, 13 passages across 7/12 claims. Six claims had at least one passage from a non-input page. Three claims ended with `search_unavailable` after one retry.
- Numeric query examples now preserve context: Lucy’s `650 miles`, flyby subject, and `April 20, 2025`; dengue’s 4.6 million/2,400 (2023) and 13 million/8,200 (2024); the Chan Chan 38-person claim and site. Before and after queries, candidate URLs, and complete matched passage text are saved in `data/dev/article_mvp_regression/retrieval_comparison.json` and the corresponding run files.
- Passage coverage means the retrieval pipeline returned a fetched paragraph. With `judge="none"`, it does not mean the paragraph supports every part of a compound claim. Input-page versus non-input-page is a location distinction; external pages are not automatically independent corroboration.

Retrieval comparison has a network limitation: the after run had nine query errors and three unavailable claims, including DNS failures in the configured search path. The 7/12 paragraph coverage is the observed run result, not a clean estimate of the query fix’s full potential. The first failed run and one retry are both preserved.

## Interpretation and limits

The pre-fix article MVP did not pass the extraction gate: four articles produced no candidates, and one actual article-mode replay proved the empty result concealed a malformed response. The current JSON Object Mode and output cap produced a valid short format response, but Chan Chan was blocked by the daily token limit, so article extraction quality remains unmeasured. Retrieval paragraphs were found for 7/12 target claims after repair versus 4/12 before; provider search failures limit this comparison.

This pilot is small and intentionally targeted. It does not measure verdict quality, exhaustive article-level extraction recall, or whether external pages are independent. Keep these four articles as the regression set. Once Groq extraction succeeds again, run this set first; then use new, unseen articles for an independent evaluation rather than tuning on this set.

## Reproduction notes

- Article-mode inputs were extracted from each linked page's article body; full article text is not included in this report.
- Extraction used the existing free Groq route, with OpenRouter and paid fallback disabled. No permission switch was changed.
- Retrieval used the existing DuckDuckGo-backed search and page fetch path, with `judge="none"` and a temporary cache.
- The article sources and their manually selected claims were passed to retrieval so the original input page could be identified separately.
- Targeted regression command: `.venv/bin/python -m unittest tests.test_extract_claims tests.test_search_evidence tests.test_articles tests.test_llm_route` — 70 tests passed.
- No verdict-model calls were made.
