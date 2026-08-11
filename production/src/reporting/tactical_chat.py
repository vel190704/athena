"""AI Tactical Chat (new reporting track, Part B): grounding-architecture
functions ONLY -- pure, synchronous, no LLM call, no session state. The
actual async Gemini/mock dispatch (`explainer.generate_chat_reply_with_source`)
and the in-memory per-session conversation history both live in `api.py`,
the same "sync prompt-builder in the reporting layer, async orchestration in
the serving layer" split every other LLM-touching reporting feature in this
project already uses (`zone_explainer.build_zone_explanation_prompt`,
`match_report.build_match_report_narrative_prompt`).

Step 0 dedup check (see the task's own report): `explainer.py`'s existing
`generate_tactical_explanation`/`generate_explanation_real` are SINGLE-SHOT
-- one prompt in, one text out, no conversation memory, no session concept,
called fresh for every spike alert or zone-report generation. Nothing in
this codebase maintains conversation history or re-grounds a running
context across multiple turns. That is the genuine, new capability this
module exists for.

STEP 1 DESIGN, STATED DIRECTLY (read before changing anything below): a
multi-turn chat is a much larger hallucination surface than a single-shot
explanation -- a user can ask anything, including questions this system has
no computed basis to answer ("what will happen in 10 minutes", "which
player should be subbed off"). This is NOT an open-ended chat wrapper
around Gemini. Every turn is grounded against an explicit, BOUNDED context
package (`build_context_package`/`format_context_package_text` below) --
the current match's compiled report (team reports, opposition analysis,
pass network summary, recent alerts), reusing `match_report.
generate_automatic_match_report` UNMODIFIED rather than recomputing a
second aggregation. `build_chat_prompt` REBUILDS this context fresh into
every single prompt sent to the model -- `api.py`'s endpoint re-fetches it
on every turn rather than caching it once per session, so a new alert
logged between turns is reflected on the very next reply, and the model is
never left reasoning from context that has gone stale mid-conversation.
"""

from production.src.reporting.match_report import generate_automatic_match_report

# Hand-picked, stated explicitly (this project's standing practice for
# every tunable constant): how many of a match's most recent alerts to
# include in the context package. Alerts are already the SPARSEST, most
# information-dense real signal available (each one already carries its
# own real, previously-generated explanation_text) -- 5 is enough for the
# model to reference specific recent moments without ballooning prompt
# size turn over turn (this project has real matches with 100+ alerts
# logged from repeated test runs against the same match_id; sending all of
# them every turn would grow the prompt unboundedly for no benefit, since
# only the CURRENT tactical picture is what a live chat user is asking
# about, not full alert history -- /alerts/history already exists as its
# own dedicated, filterable browsing surface for that).
MAX_RECENT_ALERTS_IN_CONTEXT = 5


def build_context_package(match_id: int) -> dict:
    """The bounded CONTEXT PACKAGE for one match -- reuses
    `generate_automatic_match_report` UNMODIFIED (Step 0: this is
    literally the same compiled document Part A's Automatic Match Report
    already builds; there is no reason for AI Tactical Chat to recompute
    a second, parallel aggregation of the same underlying data). Called
    fresh by `api.py`'s chat endpoint on EVERY turn -- see this module's
    own docstring for why that matters.
    """
    return generate_automatic_match_report(match_id)


def format_context_package_text(compiled_report: dict, max_recent_alerts: int = MAX_RECENT_ALERTS_IN_CONTEXT) -> str:
    """Turns `build_context_package`'s compiled dict into the readable
    text block embedded in every chat prompt. This is the ONLY
    information the model is allowed to reason from (per the
    grounding-instruction text `build_chat_prompt` wraps around it) -- if
    a fact is not derivable from this text, the model has been explicitly
    instructed to say so rather than guess.
    """
    match_id = compiled_report["match_id"]
    if compiled_report.get("no_data"):
        return (
            f"No computed data is available for match_id={match_id} "
            f"({compiled_report.get('reason', 'unknown reason')})."
        )

    teams = compiled_report["teams"]
    lines = [f"Match ID: {match_id}", f"Teams: {', '.join(teams)}"]

    for team in teams:
        report = compiled_report["team_reports"].get(team, {})
        opposition = compiled_report["opposition_analysis"].get(team, {})
        lines.append(f"\n{team}:")

        if report.get("matches_used", 0) > 0:
            threat_by_zone = report.get("threat_by_pitch_zone") or {}
            known = {k: v for k, v in threat_by_zone.items() if v is not None}
            if known:
                zone_text = ", ".join(f"{zone}: {value * 100:.1f}%" for zone, value in known.items())
                lines.append(f"  Threat by pitch zone (mean predicted shot probability): {zone_text}")
            weak_zones = report.get("weakest_control_zones") or []
            if weak_zones:
                wz = weak_zones[0]
                lines.append(
                    f"  Weakest pitch-control zone: grid cell (col={wz['col']}, row={wz['row']}), "
                    f"mean control={wz['mean_control']:.3f}"
                )
        else:
            lines.append("  No 360 freeze-frame coverage for this match -- no zone/control data computed.")

        long_pass_share = (opposition.get("build_up_tendency") or {}).get("long_pass_share")
        if long_pass_share is not None:
            lines.append(f"  Build-up long-pass share: {long_pass_share * 100:.1f}%")
        set_piece_share = (opposition.get("set_piece_reliance") or {}).get("set_piece_shot_share")
        if set_piece_share is not None:
            lines.append(f"  Set-piece shot share: {set_piece_share * 100:.1f}%")

    alerts = compiled_report.get("alerts", [])
    lines.append(f"\nTotal tactical alerts logged for this match: {compiled_report.get('alert_count', 0)}")
    if alerts:
        recent = alerts[:max_recent_alerts]
        lines.append(f"Most recent {len(recent)} alert(s) (most recent first):")
        for alert in recent:
            lines.append(
                f"  - [{alert.get('logged_at_utc')}] threat {alert.get('threat_before'):.3f} -> "
                f"{alert.get('threat_after'):.3f} (delta {alert.get('delta'):+.3f}): "
                f"{alert.get('explanation_text')}"
            )

    return "\n".join(lines)


# STEP 1.2: this is the STRONGER, more explicit version of
# `explainer._HONESTY_SYSTEM_INSTRUCTION`'s constraint the task calls for,
# specific to multi-turn drift risk. Deliberately embedded as PROMPT TEXT,
# not a change to the shared `_HONESTY_SYSTEM_INSTRUCTION` constant itself
# -- that constant is shared, already-tested infrastructure other call
# sites (`zone_explainer.py`, the spike-alert pipeline) depend on
# unmodified; this is the SAME defense-in-depth technique
# `zone_explainer.build_zone_explanation_prompt` already uses (its own
# module docstring: "already states a version of this constraint inline,
# as defense in depth, not as a substitute for this one").
_CHAT_GROUNDING_INSTRUCTIONS = (
    "You are a tactical assistant chat for a live football analytics system, answering "
    "follow-up questions across a multi-turn conversation. Follow these rules strictly, on "
    "EVERY turn, even the fourth or fifth in a row:\n"
    "1. Only answer using information explicitly present in the CONTEXT PACKAGE below. It is "
    "rebuilt fresh from real, already-computed data before every single reply you give -- "
    "always trust it over anything implied earlier in this conversation.\n"
    "2. If the question asks for something NOT present in the CONTEXT PACKAGE -- a future "
    "prediction beyond what is computed (e.g. 'what will happen next', 'what will the final "
    "score be'), a specific personnel/substitution recommendation, an injury status, or any "
    "other detail this system has not computed -- you MUST say plainly that you do not have "
    "that information, rather than generating a plausible-sounding but ungrounded answer.\n"
    "3. Never state or imply a timing/duration claim (e.g. 'in the next few minutes') or a "
    "specific player movement/positioning claim beyond what the CONTEXT PACKAGE explicitly "
    "states.\n"
    "4. Keep each reply concise (2-4 sentences)."
)


def build_chat_prompt(context_text: str, conversation_history: list[dict], user_message: str) -> str:
    """Assembles the full prompt sent to the model for one turn: the
    grounding instructions above, the FRESH context package text, the
    conversation so far (server-side history, `api.py`'s own in-memory
    session store -- passed in here as plain data, this function has no
    session concept of its own), and the new user message.

    `conversation_history`: a list of `{"role": "user"|"assistant",
    "text": str}` dicts, oldest first -- exactly the shape `api.py`'s
    session store keeps, so no conversion happens at the call site.
    """
    if conversation_history:
        history_text = "\n".join(
            f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['text']}"
            for turn in conversation_history
        )
    else:
        history_text = "(no prior turns -- this is the first message in this session)"

    return (
        f"{_CHAT_GROUNDING_INSTRUCTIONS}\n\n"
        "=== CONTEXT PACKAGE (real, computed match data, refreshed this turn) ===\n"
        f"{context_text}\n"
        "=== END CONTEXT PACKAGE ===\n\n"
        f"=== CONVERSATION SO FAR ===\n{history_text}\n=== END CONVERSATION ===\n\n"
        f"User's new question: {user_message}\n\n"
        "Respond as the assistant's next message in this conversation."
    )
