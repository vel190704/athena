"""Milestone 16 (Module 3): live match feed simulator -- replays cached
StatsBomb events sequentially, as an async generator, to simulate a live
telemetry stream for the FastAPI/WebSocket server (per README.txt Module 3:
"StatsBomb JSONs replayed sequentially to simulate live feeds").
"""

import asyncio

from production.src.ingestion.statsbomb_io import (
    fetch_match_360,
    fetch_match_events,
    parse_360_frame,
)


async def live_match_stream(match_id: int, delay: float = 1.0):
    """Yields (event, parsed_360_frame) pairs in event order, one at a time,
    `await asyncio.sleep(delay)` apart -- simulating a live feed's pacing.

    Only events with an associated 360 freeze-frame are yielded. Events
    without one cannot be featurized by the existing extraction pipeline
    (it requires player positions/velocities from the freeze-frame), so
    they are skipped here rather than passed through to crash or silently
    produce garbage features downstream.
    """
    events = fetch_match_events(match_id)
    frames = fetch_match_360(match_id)
    frames_by_event_uuid = {f["event_uuid"]: f for f in frames}

    for event in events:
        if "location" not in event:
            continue
        frame_data = frames_by_event_uuid.get(event["id"])
        if frame_data is None:
            continue

        parsed_frame = parse_360_frame(event, frame_data)
        yield event, parsed_frame

        await asyncio.sleep(delay)
