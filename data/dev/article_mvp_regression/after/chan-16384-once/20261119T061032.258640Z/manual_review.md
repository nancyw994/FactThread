# Manual review: Chan Chan full-text extraction at 16,384

Reviewed the saved Groq result against the same Chan Chan article and the prior manual-review checklist. No additional model, search, or judge request was made.

## Run

- Model: `openai/gpt-oss-120b` through Groq; one extraction request, HTTP 200.
- Prompt: revised version `sha256:e52cb13057888cc38bba9f73e2e6ac7fd229593d848a6b16829203269ade1f77`.
- Input: complete 5,234-character article; SHA-256 `785df4792b277555c254ff0a8f56f67db6b88c5d9b85f0274aaa65ea4d81034c`.
- Output cap: 16,384; `finish_reason=stop`; 27 parsed and accepted, 0 rejected.
- Usage: 8,528 completion tokens, 2,712 prompt tokens, 11,240 total. Reported reasoning usage is 5,640 tokens. The provider returned `text_tokens: null`, so visible-output token usage is unavailable; it is not inferred from the difference.

## Checklist

- **Unknown information: preserved.** Claim 22 states that the individuals’ ages, genders, and circumstances of death remain unknown.
- **Attribution in continuous quotations: partial.** Claim 24 retains Prieto, and claim 26 retains Meneses/El País. Claim 9 (“There was no scientific data…”) does not carry forward Meneses’s attribution from the preceding quote. Claim 27 (“Evidently, the citadel…”) does not carry forward Meneses’s attribution from claim 26.
- **2009 and “so far”: mixed.** Claim 11’s normalized claim omits 2009; its `missing_context` flags the omitted year. Claim 23 rewrites “so far” as “based on current evidence,” preserving the epistemic limitation in substance, though not verbatim.
- **Staircase relation: present, not separately extracted.** Claim 15 includes “A staircase leads down into the tomb’s central chamber,” but combines it with the chamber’s dimensions. It is not a separate claim. The normalized claim also drops the Andina attribution that appears in the source quote.
- **Ending and 7 percent: covered.** Claim 25 includes the researchers’ hope that the tomb will illuminate Chimú practices and social hierarchies; claim 26 includes the “no more than 7 percent” figure; claim 27 includes the citadel’s continuing potential to reveal more.

## Additional fidelity notes

- Claim 1 drops the appositive identifying Chan Chan as the pre-Inca Chimú capital.
- Claim 5 retains “this year” without adding a year to `missing_context`.
- No exact duplicate normalized claims were observed in this 27-claim output.

Overall, raising the cap produced a normally finished response with no quote-validation rejections and covered the article’s closing 7 percent statement. The manual checklist still finds attribution and normalization gaps; the staircase relation is present but not split into its own claim.
