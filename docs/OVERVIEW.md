# Project Athena — Plain-Language Overview

## What this project does

Project Athena watches football (soccer) matches — either from structured
match data or from video — and estimates, moment by moment, how dangerous
the situation on the pitch is for the team in possession. It doesn't try
to be an all-knowing football brain; it's built to answer narrower,
testable questions like "how likely is a goal in the next 15-30 seconds?"
and "if this team pressed higher right now, would the danger go up or
down?" There are two parallel versions of the system: one that runs on
official match-event data (positions, passes, shots — already fully
tested against real matches), and one that's meant to eventually run
directly on broadcast video (spot players and the ball on screen, track
them, and feed that into the same prediction engine). The video version
is built and works in testing, but hasn't yet been validated against real
broadcast footage — see below.

## Two things worth knowing about how this was built

**1. A real bug was caught before it ever reached a live system.** While
building the video pipeline, one step converts a player's on-screen pixel
movement into real-world speed (meters per second), using the same math
technique used to map a flat camera image onto the pitch's true
dimensions. That technique is only valid when applied to a *position*, not
to a *movement* between two positions — but an early version of the code
applied it to movement directly anyway. The bug was subtle: it didn't
crash or throw an obviously wrong result. It silently produced a player
"moving" at **4,242 meters per second** faster than reality (for
reference, a sprinting player tops out around 9-10 m/s), in a direction
that wasn't even correct. This was only caught because every component in
this project is tested against a hand-checked, known-correct example
before being trusted — the test compared the code's output against the
right answer, worked out by hand, and the mismatch exposed the bug
immediately. It was fixed before this code path was ever connected to a
live match feed.

**2. A real "we tried it, and it didn't help" result — reported honestly,
not hidden.** One experiment gave the prediction model a memory: instead
of only looking at where a player is *right now*, it also looked at where
that player *usually* tends to be, based on their history in past
matches, to see if that historical context improved predictions. It
didn't — predictions with the added historical memory were measured to be
*very slightly worse*, not better. Rather than dropping the experiment
quietly or reframing it as a partial win, the result is written up plainly,
along with the most likely reason: out of roughly 55 matches available,
only 4 could actually be used to build this "usual tendency" memory. That
wasn't just bad luck — it's the direct cost of a deliberate fairness
safeguard, which excluded any match that could have quietly given the
model a sneak peek at answers it shouldn't have had. The safeguard did its
job, but it also left only 4 matches to learn a habit from, which is
probably why the memory had little useful signal to add. This kind of
"it didn't work, and here's probably why" reporting shows up repeatedly
across this project, not just here — negative results are treated as
equally worth recording as positive ones, so anyone picking this project
up later knows exactly what's been tried and ruled out, and why, instead
of re-testing it blindly.

## What's real vs. still in progress

- **The match-data track (no video) is fully built and validated** against
  real historical matches — thousands of real data points, honestly
  measured, not simulated.
- **The video track is fully built and works correctly in controlled
  testing** (synthetic examples, one real practice video clip), but has
  **not yet been tested against real, ground-truth-labeled broadcast
  footage** — that requires a data-license agreement that hasn't been
  finalized yet. Every claim about the video pipeline's accuracy is
  explicitly labeled as "works in testing," never overstated as
  "proven on live TV footage."
- **A reporting layer** (player and team profile summaries, built on top
  of the already-validated match-data track) is complete and tested
  against real players and teams, including deliberately awkward edge
  cases (a player with almost no recorded touches, a goalkeeper, a team
  with very little cached data) to make sure the numbers it reports never
  look more confident than the underlying data actually supports.
