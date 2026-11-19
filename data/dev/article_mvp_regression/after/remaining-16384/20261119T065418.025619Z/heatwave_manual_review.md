# Manual review: heatwave extraction

Reviewed the completed 16,384-token Groq extraction against the saved article and the three human-selected reference claims. No search or judge was run.

## Run state

- Full 6,885-character input, no truncation; model `openai/gpt-oss-120b`; revised prompt hash unchanged.
- HTTP 200, `finish_reason=stop`; 17 parsed/accepted, 0 rejected.
- Usage: 2,935 prompt tokens; 4,442 completion tokens, including 2,707 reasoning tokens; 7,377 total. Visible-text token count was not separately returned (`text_tokens=null`).

## Review scores

- **Reference coverage: 2/3.** The winter-duration and summer-intensity references are represented. The gold reference that the CS-LIM used 64 years of real-world sea-surface temperature and height data is absent from accepted claims.
- **Source-side information gaps: 1 claim has a real source gap.** Claim 14’s source says improved forecasts could give communities more preparation time and potentially reduce harm, but gives no measured harm reduction or evidence quantifying that causal effect. Keep this separate from extraction or normalization omissions.
- **Extraction omissions (source facts present but absent from accepted claims):** the 64-year observation period; thousands of synthetic events; the CS-LIM’s seasonal treatment versus stationary LIM’s constant background assumption; and the explanation that winter-starting events may be less intense but can last for years like The Blob. These are notable omissions; article footer biographies and tool links were not counted.
- **Normalization / quote-context losses: 3 claim groups.** Claim 1 removes the California-to-Alaska definition of the Northeast Pacific and the relation that this region experienced The Blob. Claim 11 omits the “Alternatively, in a rarer situation” qualifier from its quote/normalized form. Claim 14 drops “this and future studies” and “tools” when describing the forecast pathway.
- **Over-merge:** claim 1 combines the regional record-extreme statement with The Blob’s start and duration. No exact duplicate normalized claims were found.
- **Validator metadata:** claim 10 is faithful to its source quote; the generated `missing_context` warning about lost tense/polarity/certainty is a false positive and is not counted as a source-side gap or normalization loss.

The accepted output also preserves the article’s main ENSO pathways, winter/summer contrast, 2026/27 forecast, and uncertainty language such as “might,” “some,” “may,” and “could.”
