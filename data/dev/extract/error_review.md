# Extraction review

Model: `nvidia/nemotron-3.5-super-vl:free`. Every quote in the four passages occurs once in its article, and none were rejected by the quote check. Notes below are only the differences from the human list. Later paragraphs that restate a claim are explanations, not extra claims.

## harbor

Fact type, verification scope, and status are separate. Status is `ready`, `needs_context`, `pending_document`, or `skipped`. A local fixture run then either attaches evidence or leaves a reason. It does not assign a verdict.

"The report says" stays on the 62% claim. The price clause does not inherit it. The price is `needs_context` because the article names no product or service. The 62% score and the 55% baseline resolve to the Harbor-Math report and match sentences in the fictional report. "That exam" resolves to Harbor-Math because it is the only exam name before that sentence.

"Headlines called the gain twice as good" asserts that the headlines used those words. It is `ready` against the fictional headline page. It does not require the comparison baseline. The separate claim that the report never names that system is a Harbor-Math report omission, so the complete fictional report is attached and an empty passage list does not prove the omission.

"The lab's note" and a following "It" resolve to the lab's note. The article never says that note is the Harbor-Math report, so those claims are `source_not_found` rather than checked against the report. "Short on purpose" is an intention claim, not an opinion. No fixture states that intention, so it stays undecided.

"I think" and the 2027 forecast are skipped. "The prediction is the writer's forecast, not a result in the report" is split: the forecast label is `text_analysis` and is not retrieved; "not a result in the report" gets the complete Harbor-Math report. "The line is a preference" is also `text_analysis`. "No named baseline in the document" is `needs_context` because both the report and the note are possible referents.

## acme_price

Nothing was missed. The $8 standard-tier price and the $3 batch-tier price were split, and the date and tier were kept. "Cheaper than before" is marked needs_context, and the missing item is the earlier price. The opinion and the 2026 prediction have the right categories.

Extra: the page does not print the old price, the $8 rate does not cover every Acme product, and the batch tier did not change. Those sentences are in the article. The human list does not count each one as its own fact.

## payments_team

Nothing was missed. The 40% and 12% figures were split, and both keep the payments team and Q2 2025. The 12% quote says "that team"; the standalone claim resolves it to the payments team. The opinion and the 90% prediction have the right categories.

Extra: both figures are limited to the payments team, the post gives no company-wide rate, and no figure is given for any team except payments. Those are later explanations, not new measurements.

## clinic_policy

Nothing was missed. The January 15, 2025 human-review policy, Lumen Guide, the opinion, and the 2026 ban are present. "Accuracy rose by 10 points" is needs_context. What is missing is the test name, the earlier score, and the unit.

Mild meaning change: the category stays opinion, but the standalone claim drops "I believe," so it reads like the judgment itself.

Extra: the sidebar does not name the test, the date, or the unit. That sentence explains why the previous claim lacks context.

## One evidence check

The Harbor claim "Model X scored 62% on Harbor-Math" was checked against a passage supplied by hand. Validation passed, and the verdict is supported. The quote matches the stored report text.
