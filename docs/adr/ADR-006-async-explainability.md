# ADR-006: Asynchronous SurvivalSHAP for DeepHit Explainability

## Status
Accepted

## Context
Module 7 pairs DeepHit with SurvivalSHAP to explain which features are driving a given
hazard estimate. SHAP-family methods require running hundreds of forward passes per
explanation (perturbed-input evaluations against the coalition/background sample set).
The real-time hazard stream, by contrast, must publish updated goal-probability estimates
at the cadence of the live tracking feed, on the order of the frame rate — a budget that
hundreds of forward passes per update cannot fit inside without either dropping frames
or making every consumer of the hazard stream wait on the slowest explanation pass.

Computing SHAP synchronously, inline with the hazard inference path, would couple the
latency of an interpretability feature to the latency of the core real-time prediction
path — an interpretability nice-to-have should never be able to degrade the primary
signal.

## Decision
SurvivalSHAP is moved to an asynchronous background worker, decoupled from the
real-time inference path:

- The real-time stream pushes raw hazard scores to clients as soon as they are computed,
  with no dependency on SHAP.
- The background worker independently computes SHAP values on a fixed cadence (every 5
  seconds) rather than per-frame.
- The worker pushes the resulting LLM-generated textual summary to a **secondary**
  WebSocket channel, kept separate from the primary hazard-score channel so that slow or
  failed explainability computation can never block or degrade the primary signal.

## Consequences
- Explanations are necessarily stale relative to the live hazard score they describe (up
  to a 5-second lag). This is an accepted trade-off — explainability is diagnostic and
  human-facing, not a control input to any downstream automated decision.
- The background worker needs its own failure isolation (retries, timeouts) so that a
  SHAP computation failure never propagates to the primary WebSocket channel or the
  hazard inference path.
- Consumers of the textual summary channel must be built to tolerate a
  slower-than-real-time cadence and must not assume 1:1 correspondence between a hazard
  update and an explanation update.

## Alternatives Considered
- **Synchronous inline SHAP**: rejected — couples explainability latency to real-time
  hazard latency, risking dropped frames or stream stalls.
- **Reduced-sample SHAP (fewer forward passes) computed synchronously**: rejected —
  even a reduced sample count does not reliably fit the real-time budget, and reducing
  sample count degrades explanation fidelity in a way that is hard to bound.
- **Precomputed/offline-only explanations**: rejected — would not reflect the live
  in-match feature state that SurvivalSHAP is meant to explain.

## Update: Real Gemini Flash-Lite Integration

Milestone 15 implemented this ADR's async design with a deterministic MOCK executor
(`explainer.generate_explanation`) — no real network call, no real language model,
just a template that echoes back numbers already computed by the model + Captum. This
was always a placeholder for "a real LLM integration (currently a mock/templated
executor everywhere)," named as open in every findings document since. This update
replaces the mock with a real Gemini Flash-Lite call for the two production paths that
generate explanation text (the WebSocket spike-alert pipeline, `api.py`), while leaving
the mock permanently in place as the deterministic fallback and default test executor.

**SDK/model verified against real, current documentation before writing any code, not
recalled from memory** — model names and SDK packages have changed multiple times
across this project's own history (the roboflow/AGPL investigation in ADR-014 is the
direct precedent for "verify, don't assume" applied to a third-party ML service).
Confirmed: `google-genai` (PyPI, v2.16.0) is Google's current, GA-recommended SDK — the
older `google-generativeai` package is deprecated and NOT used. The model lineup has
moved to a Gemini 3 generation since this project's own knowledge cutoff;
`gemini-3.5-flash-lite` — confirmed present on both Google's models page and pricing
page — is the current fastest/cheapest Flash-Lite variant, not the older
`gemini-2.5-flash-lite` a from-memory guess would have produced. Free-tier RPM/TPM/RPD
numbers are no longer published as a static table in the current docs (unlike some
older API generations); Google's own docs direct users to their account's live AI
Studio dashboard for exact figures — no specific quota number is hardcoded or assumed
anywhere in this integration, and this ADR's own spike-triggered (not per-frame)
calling pattern keeps real call volume inherently low regardless of the exact number.

**Design: additive, gated, never call-site-aware.** `explainer.py` gained two new
functions, `generate_explanation` (the original mock) left completely unmodified:
- `generate_explanation_real(prompt)` — the real Gemini call (`google-genai`'s native
  async client, `client.aio.models.generate_content`, not `asyncio.to_thread` — an
  awaitable version already exists, so Milestone 16's to-thread pattern, which exists
  for genuinely synchronous SDKs, does not apply here). Raises on any failure; never
  catches internally.
- `generate_tactical_explanation(prompt)` — the new public entrypoint. Checks
  `GEMINI_API_KEY` (loaded via `python-dotenv`, same convention as `ROBOFLOW_API_KEY`);
  absent key skips straight to the mock. Present key tries the real call; ANY failure
  (timeout, rate limit, invalid key, empty/malformed response, anything else) is caught,
  logged as a single WARNING via `logging` (never the key itself — see
  `_safe_error_text`'s defensive masking), and falls back to the mock. This function
  never raises for a Gemini-side failure — the live alert flow and any reporting-tool
  caller must never crash because an external API had a bad moment.

`api.py`'s spike-alert pipeline now calls `generate_tactical_explanation` instead of
`generate_explanation` directly (its only production call site) — a one-line change,
exactly the "call sites don't need to know which executor is active" design this update
set out to achieve. `zone_explainer.py` (Milestone 43's zone-level prompt builder)
needed **no change at all**: it only ever builds a prompt string and has never called an
executor directly anywhere in production code (only `test_zone_explainer.py` does, to
exercise the mock) — any future caller that wires it up gets real-vs-mock gating for
free, automatically, the moment it calls `generate_tactical_explanation` instead of the
raw mock.

**Honesty constraint re-verified against REAL output, not just reasoned about.** A real
LLM, unlike the deterministic mock, can plausibly hallucinate exactly the kind of
unsupported claim (a duration, a recovery direction) Milestone 43's zone-explainer
already proved this project's data cannot support. `generate_explanation_real` sends an
explicit system instruction (via the SDK's own `system_instruction` config channel, not
string-concatenated into the user prompt) forbidding timing/duration/speed claims and
player-movement/recovery claims, on top of `build_zone_explanation_prompt`'s own
existing inline version of the same constraint (defense in depth, not redundant).
**Result: real Gemini output passed the exact same regex-based honesty check
(`test_zone_explainer.py`'s `_UNSUPPORTED_CLAIM_PATTERN`) on the first attempt, for both
the scalar-feature prompt (Milestone 15) and the zone-level prompt (Milestone 43) — no
prompt iteration was needed.** Both real outputs are reproduced in
`docs/REPORTING_FINDINGS.md`'s own update for direct review.

**Testing: mock stays the default everywhere; the real integration is opt-in and
CI-safe.** `test_explainer.py`/`test_zone_explainer.py` are unchanged and still call the
mock directly, unconditionally — a real LLM's prose does not contain the mock's literal
`"Tactical Analysis:"` template string, so these tests would break their own intent
(unit-testing the deterministic parser) if pointed at real output instead. A new test,
`test_explainer_real_gemini_integration`, skips cleanly (not fails) whenever
`GEMINI_API_KEY` is absent — the same pattern the SoccerNet-gated test already
established — verified directly by temporarily relocating this project's own `.env`
file and re-running the full suite (157 passed, 3 skipped: SoccerNet, this new test, and
`test_tactical_map_renderer.py`'s pre-existing Roboflow-gated test, which also lost its
key when `.env` was moved) before restoring it. With the key present, this same test
exercises a real call, the honesty check, and a simulated-invalid-key graceful-fallback
check, all in one pass.

## Alternatives Considered (Update)

- **Fold the real call directly into `generate_explanation`**: rejected — that function
  is unit-tested directly, by name, for its own deterministic parsing logic; making it
  conditionally real would either break that test's intent or require the test itself to
  special-case away real behavior, both worse than a small, clearly-named second
  function.
- **Have `generate_explanation_real` catch its own errors and fall back internally**:
  rejected in favor of a two-function split (`generate_explanation_real` raises cleanly;
  `generate_tactical_explanation` catches and falls back) — this makes the fallback
  behavior directly, independently testable (simulate a bad key, assert the *real*
  function raises; assert the *dispatcher* still returns mock output) rather than
  entangling "did the call fail" with "did the fallback work" inside one function.
