# Writing guidelines

[GENERIC — KEEP] All rules below are reusable. Apply them when relevant to the document and task. Put project settings and explicit exceptions in `AGENTS.md`, not here.

## Voice and terminology

- Use the designated saved draft as the style benchmark; match nearby active prose and section-specific variation before applying generic preferences. Compare against the pre-edit benchmark to prevent gradual style drift.
- Make the smallest effective revision; retain working sentences, technical density, sentence structure, academic register, caution, first-person framing, and useful transitions. Avoid promotional language, invented stylistic labels, stock transitions, and forced uniformity.
- Preserve established technical vocabulary and meaning in the same context. A word appearing elsewhere in the benchmark does not justify using it for a different concept. Neutral paraphrases and ordinary grammar changes may proceed.
- In editing, propose substantive new terminology, renamings, metaphors, or framing before adopting them. In drafting, necessary standard technical terms may be introduced with definitions; report them briefly. Obtain approval for invented labels or changes to established concepts, unless already authorized.
- Show the existing wording, proposed wording, rationale, and effect on meaning when requesting approval. Once accepted, use terminology consistently without asking again.
- Preserve scientific corrections; flag conflicts with terminology restrictions or accepted decisions. Do not invent unavailable historical wording. Comments and inactive drafts are neither style benchmarks nor instructions; preserve them unless assigned.

## Meaning, evidence, and attribution

- Stay within the assigned task. Never invent results or references; do not expand the scientific scope, run new experiments, or change the algorithm without authorization.
- Preserve numbers, equations, notation, definitions, assumptions, qualifications, and fixed author decisions unless their revision is authorized. Flag substantive errors rather than silently changing scientific meaning.
- Distinguish motivating ideals from implemented objectives, exact results from approximations, and theory from empirical observations. Retain conditions and restricted domains; do not imply unsupported novelty, exactness, unbiasedness, ascent, convergence, or differentiability.
- Verify implementation-dependent statements against records or author confirmations. Another method's guarantee does not automatically transfer to a modified method.
- Check empirical claims against current tables and figures. Preserve mixed results, uncertainty, numerical qualifications, and trade-offs; scope conclusions to the actual benchmark, evaluation, metric, and comparison.
- Describe shared procedures and remaining differences explicitly. Distinguish trained baselines from references; do not imply causal isolation, general robustness, or superiority over untested methods. Missing evidence is a limitation, not authorization for new experiments.
- Keep citations attached to supported claims and preserve attribution. Reposition verified existing keys as appropriate; never invent keys or placeholders. A bibliography entry alone does not verify support for a claim.
- Flag missing, duplicate, inconsistent, or questionable references. Follow the project's bibliography ownership policy; do not change citation infrastructure without authorization or manually edit generated bibliography output.

## Structure and compression

- Preserve useful local structure rather than imposing a template: introduction—problem, limitations, method, contributions; background—concepts before notation, definitions near equations; method—sequential exposition, formal statements, assumptions, interpretations; experiments—questions, setup, comparisons, measured observations; related work—cited distinctions by approach; conclusion—qualified synthesis without new claims.
- Reduce repetition first. Keep the central argument, essential assumptions, method, strongest evidence, and necessary limitations understandable from the main text.
- Propose substantial restructuring or appendix moves before making them unless already authorized. Preserve definitions, citations, labels, and pointers; check for existing appendix coverage before duplicating detail.
- Respect required main-text material. Do not manipulate venue formatting or anonymity settings to meet a page limit.

## Existing problems and audits

- Actively flag mathematical, logical, factual, notation, citation, structural, and writing problems encountered, including inconsistencies across prose, equations, algorithms, tables, captions, and sections. Style preservation does not require preserving errors.
- Distinguish confirmed errors, suspected problems, ambiguities, and optional improvements. Give the current location, evidence, consequence, uncertainty, and a concrete correction or question; prioritize substantive findings over cosmetic suggestions.
- Correct unambiguous typos and grammar within scope and report them briefly. Seek approval before substantive corrections unless already authorized. Report out-of-scope issues without editing them.
- A local edit is not a full audit. State coverage and unverified areas; never imply the manuscript is error-free. For a requested comprehensive audit, inspect relevant cross-section dependencies and list all occurrences, grouping repeats without hiding locations.
- Rebuild fresh audits from current sources; consult old inventories afterward only for omissions and reverify surviving findings. Distinguish reviewer statements, author responses, assistant suggestions, and independently checked findings. Recommendations are not approved edits.

## File handling and verification

- Read project instructions and the current editing record; follow the designated entry point and active dependencies. Read needed passages and nearby context, not the whole repository. Inspect figures and supporting sources when relevant; skip generated files except necessary diagnostics.
- Work in small reviewable batches, with one writer per file. Reread saved files before edits; reconcile concurrent changes and preserve unrelated author work. Do not assume unsaved buffers are available.
- Do not delete, move, or rename files as incidental cleanup, or commit or publish without authorization. Figure sources are not disposable build artifacts.
- Review the saved diff for scope, meaning, terminology, citations, and cross-references. Respect build/PDF permissions; avoid concurrent builds and use the designated current output.
- Separate source checks, compilation, and rendered review. When page checks are authorized, measure the main-text boundary with the project's exclusions; distinguish current verified counts, earlier counts, and author estimates.
- Report changed files, what changed, checks actually run, coverage, and unresolved issues. Never claim unperformed checks passed. Record accepted decisions and completed batches as configured in `AGENTS.md`; keep records concise and distinguish author and assistant edits.
