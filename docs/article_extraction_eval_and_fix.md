# 文章声明提取：评估与修复轮次

**日期：** 2026-11-19
**基线：** commit `7eedc74`（本轮开始前已将用户既有改动全部提交并推送）。本轮修改未提交。
**状态：** 代码修复完成并通过离线验证；提示词 `article-claims-v2` 的提取质量效果**未验证**（没有新的模型生成）。

## 方法

- 全程离线，只使用已保存的原文、原始回复和结果：Chan Chan `after/chan-16384-once/20261119T061032.258640Z/`，Heatwave `after/remaining-16384/20261119T065418.025619Z/`。未调用提取、搜索、judge 或其他外部模型 API；未改动 provider、付费或 OpenRouter 设置；未读取密钥。
- Lucy（429，无输出）和 Dengue（未请求）不计作提取遗漏。
- 真实 subagent：每个 agent 是独立的 `codex exec` 进程，同时最多运行 2 个。Agent A（语义与覆盖）和 Agent B（代码与状态）各自在只读、断网的沙箱里独立审查；主 agent 回到原文和代码核实并裁决；单个实现 agent 在断网的 `workspace-write` 沙箱中写代码；未参与实现的 Agent C 做独立验证和一次针对性复核。
- 各 agent 的 brief、裁决和全部报告保存在仓库外的 `~/factthread-agent-run/20261119T021152/`；修改前的完整工作区快照在其中的 `snapshot/`。
- 仓库中没有 `AGENTS.md`；项目规则以 README.md 和 `docs/article_mvp_freeze_and_pilot_eval.md` 为准。

## 1. 确认的问题

| ID | 问题 | 证据 | 优先级 |
|---|---|---|---|
| F1 | 截断回复被记为成功提取 | `web/articles.py` 中 `provider_extract` 只把 `result["content"]` 交给解析器，忽略 `finish_reason`。B 用 stub 构造 `finish_reason="length"` 加可解析的 JSON，`extract_claims` 返回 `status: completed`。 | 1（状态） |
| F2 | 旧提取结果被当作当前结果展示 | `ArticleBook.start_extract` 只要原文哈希、URL 相同且状态为 `extracted`/`done` 就复用，不检查提示词、模型和输出上限。B 复现：替换 hook 后仍返回旧 hook 的结果（`calls ['old'] ... candidate_count 1`）。 | 1（状态） |
| F3 | 忠实声明被标为 needs_context | `_finalize` 在序数或情态检查失败、回退为原文措辞后，仍把对应缺口留在 `missing_context`。Heatwave 第 10 条：`source_quote == normalized_claim`，却有 `missing_context=["规范化声明未保留原文的时态、否定或确定程度"]`，状态为 `needs_context`；`heatwave_manual_review.md` 判为误报。 | 2（含义） |
| F4 | 覆盖与上下文丢失缺少检测 | 见下方证据。 | 3（覆盖） |

F4 证据（原文均取自 `articles.json`）：

- Heatwave H11：引文从句子中间开始（`that an Eastern Pacific El Niño develops, we might not see ...`），丢失了可能性限定 “Alternatively, in a rarer situation”。
- Heatwave H1：规范化声明丢掉 “the open ocean area from California to Alaska which most notably experienced”，失去区域定义及其与 The Blob 的关系，两件事改用 “and” 连接。
- Chan Chan C24：规范化声明丢掉了对说话人的限定 “who wasn’t involved in the research”。
- Chan Chan C9（`There was no scientific data on the individuals ...`）和 C27（`Evidently, the citadel has much more to tell.`）：都是紧接在已归属给 Meneses 的引语之后的连续引语，规范化声明里没有说话人。
- Heatwave 漏提参考声明 `In this case, the researchers’ CS-LIM used 64 years of real-world sea surface temperature and height data.`（参考覆盖 2/3），另漏约 7 条方法与机制事实；Chan Chan 漏提 `making its survival, untouched by colonial and contemporary looters alike, even more impressive.`

歧义项（本轮只靠提示词处理或不改）：

- Chan Chan C1 丢掉了 Chimú 首都的同位语，但该信息在文中别处出现。
- Heatwave H7 丢掉了 “staying power”，属于冗余解释。
- Heatwave H14 丢掉了条件 “With improved understanding ... from this and future studies”，属于语义问题。

不成立或检查后无问题：引文字符位置与 Unicode 高亮；解析、请求、空结果、过滤后为空各自独立的状态；未运行状态的展示；统计口径的措辞。B 报告的 19 个测试错误是只读沙箱无法创建临时目录所致；在沙箱外运行同样 4 个测试模块为 70/70 通过。

## 2. 修改与对应验证

| 修复 | 修改 | 验证 |
|---|---|---|
| F1 | `finish_reason` 传入提取流程；状态以实际提供声明的那次尝试为准。非 `stop` 记为 `incomplete`，界面把已有声明标为部分结果；没有候选时说明回复被截断、未产生可用候选。截断后重试并以 `stop` 结束的记为 `completed`，各次尝试的 finish_reason 都保留在诊断里。 | `test_finish_reason_length_is_incomplete_with_partial_claims_and_stop_is_complete`、`test_successful_stop_repair_is_complete_after_truncated_attempt`、`test_incomplete_provider_reply_is_exposed_as_partial_extraction`、`test_incomplete_zero_candidate_ui_explains_no_usable_candidates` |
| F2 | 新增 `extraction_fingerprint`，由提示词内容、`PROMPT_VERSION`、模型、输出上限、response_format 和 `EXTRACTION_CODE_VERSION="article-extraction-2"` 组成，完全一致才复用。没有 fingerprint 的旧文章不复用，但仍可查看。 | `test_extraction_fingerprint_reuses_only_an_exact_config_match`；C 读代码确认各字段都已覆盖。 |
| F3 | 序数或情态回退改记为规范化说明（“模型规范化改变了时态/确定程度，已回退为原文措辞”），不再写入 `missing_context`；数字丢失检查不回退，仍报缺口。 | `test_modality_fallback_is_a_note_not_a_stale_gap_and_number_loss_stays_a_gap` |
| F4a | 未覆盖句子诊断：列出与任何候选的字符范围都不重叠的原文句子。仅作审阅辅助，不是召回指标，也不会生成声明。 | `test_uncovered_sentence_review_diagnostic_and_full_coverage` |
| F4b | 引文从句中开始的诊断，不改动引文。引文前只有开引号、括号或换行时不触发。 | `test_mid_sentence_quote_gets_note_but_sentence_start_does_not` |
| F4c | 直接引语的规范化声明没有说话人时，加缺口“直接引语未在规范化声明中标注说话人”。覆盖 tells/told 等归属动词、引号落在匹配范围之内，以及同一段落相邻引语中明确的归属。只检测，不推断、不补写说话人。 | `test_quoted_claim_tells_and_delimiters_are_detected_but_named_speaker_is_not`、`test_adjacent_quote_in_same_paragraph_uses_explicit_attribution_only` |
| F4d | 提示词修订一次，版本 `article-claims-v2`（版本行写入提示词）：保留同位语定义、可能性限定、对说话人身份或独立性的限定；引文从句子或分句边界开始；只有原文明确时才延续连续引语的说话人；提取方法、数据和样本事实。版本号计入 fingerprint。 | 未验证，需要新的生成结果（见第 6 节）。 |

新测试全部使用通用的合成文本，不包含回归文章的句子。Agent C 把新测试复制到 `7eedc74` 的 `git archive` 副本里运行：5 个失败、3 个报错，说明它们能检出修复前的问题；在当前代码上全部通过。C 没有发现针对样例的硬编码、超出范围的改动或对既有改动的回退。

独立验证共两轮。第 1 轮 C 找出 2 个阻塞问题：一是截断后重试成功仍被标为 incomplete；二是 F4c 漏掉了 Chan Chan C9 和 C27（不认识 `tells`，且引号落在匹配范围之内）。实现 agent 修复后，C 的针对性复核结论是**没有剩余阻塞问题**。

## 3. 测试命令与结果

以下在你的 Mac 上、沙箱外运行：

```
.venv/bin/python -m unittest tests.test_extract_claims tests.test_search_evidence tests.test_articles tests.test_llm_route
Ran 80 tests ... OK            （修复前为 70）

.venv/bin/python -m unittest discover -s tests
Ran 162 tests ... FAILED (failures=2)
```

这 2 个失败是 `test_check_server.CheckPageTests.test_a_long_passage_stays_short_until_it_is_expanded` 和 `test_a_supported_result_shows_the_quote_while_search_is_running`，在基线 `7eedc74` 上同样失败（152 tests，failures=2），不是本轮引入的，留待后续处理。

`git diff --check` 通过。

离线重放（只用已保存的原文和原始回复）：

```
.venv/bin/python data/dev/article_mvp_regression/offline_replay/replay.py --baseline-code-root /tmp/ft-base --output data/dev/article_mvp_regression/offline_replay/comparison.json
```

## 4. 可以有效比较的指标（仅代码路径）

把同一批已保存的原始回复分别送入旧代码和新代码，衡量的只是解析、状态和诊断行为，**不代表新提示词的效果**：提示词虽然改了，但这些回复是旧提示词生成的。

| 文章 | 接受/拒收 | ready / skipped / needs_context | 未覆盖句子 | 从句中开始提示 | 无说话人引语 | 回退说明 |
|---|---|---|---|---|---|---|
| Chan Chan，旧代码 | 27/0 | 19 / 7 / 1 | 无此项 | 无此项 | 无此项 | 无此项 |
| Chan Chan，新代码 | 27/0 | 18 / 7 / 2 | 5 | 5 | 2（C9、C27） | 0 |
| Heatwave，旧代码 | 17/0 | 12 / 4 / 1 | 无此项 | 无此项 | 无此项 | 无此项 |
| Heatwave，新代码 | 17/0 | 13 / 4 / 0 | 31 | 6 | 0 | 1（H10） |

只有两条声明的状态发生变化：Chan Chan C9 从 ready 变为 needs_context（检出缺少说话人），Heatwave H10 从 needs_context 变为 ready（移除了残留警告）。C 认为两处变化都有依据。没有误拒收，接受数也没有变化。

C 逐条核对了全部 11 条“从句中开始”提示：5 条合理，6 条是误报，误报都来自被压平进正文的标题、题名或署名行。同样由于这种压平，未覆盖句子数只是近似值，可能偏少，不能当作召回指标。

参考声明覆盖（Heatwave 2/3，Chan Chan 的数字见其人工复核）和 A 的语义发现都基于**旧提示词**生成的回复，作为修复前的评分和评分依据保留。目前还没有修复后的评分。

## 5. 未验证的模型质量与剩余问题

- 提示词 `article-claims-v2` 能否提升召回和忠实度**尚未验证**，不能用旧输出的重放来证明。
- Chan Chan 和 Heatwave 已参与问题定位和修复，现在属于**回归集**；泛化效果需要用未参与修复的新文章评估。
- F4b 会因标题、署名行被压平而误报，F4a 的分句是近似的，两者都只作审阅辅助。
- 歧义项（C1、H7、H14）只靠提示词处理，效果未知。
- 已有的 2 个 `test_check_server` 失败未修复。
- Lucy 和 Dengue 仍没有成功的提取结果。
- `data/dev/url/cache/` 和各个 `last_*.json` 是未提交的本地运行产物，本轮没有改动。

## 6. 后续最小在线验证方案

运行前需要明确授权。保持相同配置：Groq `openai/gpt-oss-120b`、整篇输入、`max_completion_tokens=16384`、temperature 0、JSON Object Mode，关闭 OpenRouter 和付费访问。

1. **回归（2–4 次请求）。** 用 `article-claims-v2` 对 Chan Chan 和 Heatwave 各请求 1 次；只有引文校验失败才触发一次修复重试，所以最多 4 次。按现有参考声明和本报告中冻结的 A 发现清单评分（Heatwave 的 64 年声明、H1、H11、C24、C9/C27 的说话人等），同时报告 finish_reason 和新的诊断计数。
2. **可选补全（2–4 次请求）。** 先确认 Groq 当日 token 额度足够（上次为每日 200k），再对 Lucy 和 Dengue 各请求 1 次。
3. **泛化（2–3 次请求）。** 选 2–3 篇新主题短文，**运行前**为每篇冻结 3 条参考声明和一份忠实度检查点，然后每篇请求 1 次。这些文章不用于调参。

最少 2 次请求（只做回归），建议 4–5 次（回归加 2–3 篇新文章），连同修复重试的上限约 10 次。

## 附录：article-claims-v2 最小在线回归（2026-11-19）

- 配置：Groq `openai/gpt-oss-120b`，上限 16384，OpenRouter/付费关闭，每篇 1 次请求、无修复；prompt sha256 `433aaed6…0c3d`。检查点先冻结于 `frozen_checkpoints_prompt_v2.json`。
- 运行：`after/prompt-v2-16384/20261119T084848.429760Z/`。Chan Chan 31 条接受 / 0 拒绝；Heatwave 34 / 1；两篇 finish_reason=stop；共 25,328 token（Heatwave 第一次遇到每分钟限流 429，等待后 resume，未计费）。
- 冻结检查点（7 项）：忠实覆盖 3（HW-64Y、CC-C9、CC-C27），部分覆盖 2（HW-H11 丢 “rarer situation” 限定；CC-C24 丢 “wasn’t involved”），遗漏 2（HW-H1 区域定义；CC-LOOT）。参考 claim 6/6 覆盖。
- 新发现：(1) c19 因引文首字母大小写不同被精确校验拒绝，导致 “中太平洋 El Niño → 阿拉斯加湾热浪” 丢失；(2) Heatwave #8 出现无关的 needs_context 理由；(3) speaker 字段全为空，归属只写在 normalized_claim 里。
- 结论：单次运行，仅作回归信号，不代表提取质量已验证；v2 修正了 C9 归属，但限定词保留和整句覆盖仍未达标。

## 附录：held-out 泛化测试（2026-11-19）

- 3 篇未参与调参的新文章（`data/dev/article_generalization/`）：NIH PFAS 新闻稿、NPR 首次购房者数据报道、NPR/AP Rubio 断网新闻（多发言人）。21 个检查点在模型运行前冻结（04:25:25-05:00，SHA256SUMS 校验通过）。
- 配置同上（v2、gpt-oss-120b、16384、每篇 1 次）；共 31,948 token；全部 finish_reason=stop。接受/拒绝：PFAS 29/5，NAR 40/1，Rubio 26/0。
- 结果：忠实覆盖 14，部分覆盖 7，遗漏 0。部分覆盖的共性：规范化时丢掉限定或归属（"so-called"、"what the agency described as"、Kennedy 引语被写成无主事实、未带 "did not present any evidence"），以及句中切分导致主语或比较基准缺失。
- 被拒 7 条（含 dev 的 Heatwave c19）：3 条只差首字母大小写，2 条只差标点（直/弯引号、句尾多加句号），2 条是真实改写，拒绝是正确的。前 5 条属于可以安全修复的校验缺陷，前提是修复后引文必须换回原文的精确片段并留下记录。
- 结论：新文章上没有退步，覆盖率可以接受；主要短板是限定词和归属的保留。这是单次运行，样本小，只作信号。

## 附录：排版级引文匹配修复（2026-11-19，article-extraction-3）

- 改动：`pipeline/extract_claims.py`。先做精确匹配；失败时才启用回退，回退只容忍首字母大小写、直/弯引号和撇号、连字符/破折号变体，以及引文末尾标点的差异。回退加了词边界保护（"no." 不能匹配到 "nothing" 里），而且必须唯一命中。命中后 source_quote 和 source_ranges 一律换成原文的精确片段，并写入 extraction_notes「引文与原文仅大小写/标点不同，已替换为原文精确片段」。`web/articles.py` 的版本号升到 `article-extraction-3`，旧缓存不会被复用。
- 测试：`tests.test_extract_claims` 33 个全部通过（新增 4 个，含 2 个边界负例）；全量 166 个，只有 2 个是已知的 `test_check_server` 旧失败。
- 离线重放：用已保存的输出重放，只重新跑校验和 finalize，没有发新的模型请求。原先被拒的 7 条里，5 条恢复（Heatwave c19、PFAS 4 条）；2 条真实改写仍然被拒。所有已接受 claim 重新校验后都不变。
