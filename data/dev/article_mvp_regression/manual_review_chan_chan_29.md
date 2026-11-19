# Manual review: saved 29-claim Chan Chan extraction

## Scope

Reviewed the 29 accepted claims in `baselines/chan_chan_groq_gpt-oss-120b_8192.json` against the saved full Chan Chan article in `articles.json`. This is a human review of the existing Groq output only. No model, search, or judge request was made. The saved full-article run is a usable baseline, not evidence that full-article extraction is generally better than chunking.

## Findings

- **Omitted article statement:** The article says much about the tomb remains unknown, including the individuals’ ages, genders, and circumstances of death. None of the 29 claims captures that knowledge-state statement.
- **Attribution lost or weakened:** Claims 7–9, 11–12, 23–25, and 28–29. Details are listed below. Claims 11 and 12 also omit the attribution from `source_quote`; the others generally retain some attribution in the quote but drop it or weaken it in `normalized_claim`.
- **Relationship omitted in normalization:** Claim 16 keeps the chamber dimensions but drops that a staircase leads down into the tomb’s central chamber.
- **Overlap:** Claim 25 repeats the “discovery of the decade” portion already present in claim 24’s source quote. Claims 1 and 22 both mention the total of at least 38 people, but claim 22 also adds the 17 skeletons found during platform excavation, so this is meaningful overlap rather than a full duplicate.
- **Scope qualifier drift:** Claim 23 drops “so far”; claim 29 drops “Evidently.”
- The remaining claims preserve their central asserted meaning on this pass. Claim 5 appropriately leaves “this year” unresolved in `missing_context`.

## Claim-level review notes

1. **No issue.** Keeps the undisturbed mausoleum, minimum count, and funerary goods.
2. **No issue.** Keeps the approximate age, people, burial place, and Chimú attribution.
3. **No issue.** Preserves the grave-good list.
4. **No issue.** Preserves the inference and its uncertainty (“perhaps”).
5. **No issue.** “This year” remains unresolved and is recorded in `missing_context`.
6. **No issue.** Keeps the tomb’s date relative to the Inca conquest.
7. **Attribution weakened.** The quote names lead archaeologist Jorge Meneses and BBC News, while normalization changes this to “Jorge Meneses said,” dropping his role and the outlet.
8. **Attribution lost in normalization.** The quote identifies Meneses telling El País; the normalized claim is un-attributed.
9. **Attribution lost.** This is the adjacent continuation of Meneses’s quoted remarks to El País; the normalized claim presents it as bare fact.
10. **No issue.** Preserves the Chimú date range and region.
11. **Attribution lost; quote context incomplete.** The article attributes the irrigation-canal statement to hydraulic engineer Charles Ortloff, speaking to Smithsonian magazine’s Bruce Hathaway in 2009. Neither the quote nor normalized claim preserves that attribution.
12. **Attribution lost; quote context incomplete.** The “first true engineering society” characterization is attributed to Ortloff in the article, but the quote and normalized claim do not preserve that source.
13. **No issue.** Preserves the relationship between accession and construction of a new royal compound.
14. **No issue.** Preserves where rulers and their retinues were laid to rest.
15. **No issue.** Preserves the mausoleum’s location.
16. **Relationship omitted in normalization.** The source quote includes the staircase leading down into the tomb’s central chamber; normalization retains only the chamber’s measurements and Andina attribution.
17. **No material issue.** Preserves the individual, skull, pigment uncertainty, and cinnabar description.
18. **No issue.** Preserves the missing bones as the reason researchers posit moving or reburial.
19. **No issue.** Preserves the Andean practice and the interaction with or worship of the dead.
20. **No issue.** The short quote is context-dependent, but the normalization resolves “one” to one of the rooms flanking the central chamber.
21. **No issue.** Resolves “the other” to the other flanking room and keeps the approximate count and artifacts.
22. **Overlap with claim 1, not a full duplicate.** Adds the 17 skeletons found during platform excavation and the resulting minimum total.
23. **Attribution and qualifier weakened.** The normalized claim keeps Henry Luis Gayoso’s name but drops his Ministry of Culture role, the New York Times attribution, and “so far.”
24. **Attribution lost.** The statement is Gabriel Prieto’s, an anthropologist at the University of Florida who was not involved in the research, speaking to the Times. Normalization presents it without that attribution.
25. **Overlaps claim 24 and loses attribution.** “Discovery of the decade” is repeated in claim 24’s source quote. If retained separately, it should remain attributed to Prieto.
26. **No issue.** Preserves the researchers’ hope and the future purpose of the undisturbed tomb.
27. **No issue.** Preserves the discovery’s implication that much of the capital remains unexplored.
28. **Attribution lost in normalization.** The quote identifies Meneses telling El País; the normalized claim is un-attributed.
29. **Attribution and evidential marker lost.** This is Meneses’s continuation after the El País attribution in claim 28’s sentence. Normalization drops both the speaker context and “Evidently.”

## Interpretation

The most concrete extraction defect in this saved set is not a missing number: it is the omission of the article’s explicit “much remains unknown” statement. Attribution handling needs attention across several quoted or attributed claims, and claim 16’s normalization loses a spatial relation. These observations apply to this saved Chan Chan output only; they do not compare Gemini chunking with Groq full-article extraction.

## Follow-up review: saved revised-prompt output

Also reviewed `after/prompt-revision/20261118T194855.244118Z/chan_chan.json` offline against the same article. It used the same Groq model, full input, and 8192 output cap as the baseline, but a different prompt version. This is a quality check of the saved result, not a controlled claim that one prompt or input strategy is superior.

- The revised output is truncated: HTTP 200, `finish_reason=length`, 8,192 completion tokens, 2,712 prompt tokens, 10,904 total tokens; recorded elapsed time was about 82.8 seconds.
- It parsed 23 candidates; 22 were accepted and one rejected. The rejected quote says “walled palates” where the article says “walled palaces,” so the fact about Chimú kings’ walled palaces and each ruler building a new compound on accession is missing from the accepted output.
- The explicit unknowns statement appears as claim 20, fixing the corresponding omission in the saved baseline.
- Attribution improved for Meneses, Ortloff, Gayoso, Prieto, and Andina in several claims. Claim 9 (“There was no scientific data…”) still loses Meneses’s attribution from its continuation context.
- Claim 11 preserves Ortloff’s attribution, but its normalized claim drops the 2009 date; the local diagnostic correctly calls out the missing year. Claim 5 retains “this year” but no longer records the unresolved year in `missing_context`.
- Claim 14 again drops the staircase-to-central-chamber relation. Claims 17–18 resolve the two rooms’ references from context.
- The response ends before the later claims: the researchers’ hope that the tomb would illuminate Chimú practices and hierarchies, the statement that Chan Chan remains largely unexplored, and the “no more than 7 percent” excavation figure are absent. The existing coverage comparison records 2 of 3 reference claims matched; the 7 percent claim is among the missing references.
- Claim 22 combines Prieto’s “discovery of the decade” characterization with the “first time” comparison. No exact duplicate normalized claims were detected, although the first claim and the “additional 17 / at least 38” claim overlap in their population count.

This follow-up reinforces that fixing one omission (the unknowns statement) did not make the revised output complete: it added an exact-quote rejection, lost some qualifier/context, and truncated before the article’s closing facts. Neither saved Groq result is evidence about Gemini chunking quality.
