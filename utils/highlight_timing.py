EVENT_WINDOW_BEFORE_AFTER = {
    "goal": (20.0, 8.0),
    "penalty": (16.0, 8.0),
    "foul lead to penalty": (14.0, 8.0),
    "red card": (10.0, 8.0),
    "second yellow card": (9.0, 7.0),
    "yellow card": (8.0, 6.0),
    "saved by goal-keeper": (12.0, 8.0),
    "shot off target": (12.0, 7.0),
    "corner": (10.0, 7.0),
    "free kick": (10.0, 7.0),
    "var": (12.0, 10.0),
    "injury": (10.0, 9.0),
    "substitution": (8.0, 8.0),
}

DEFAULT_WINDOW_BEFORE_AFTER = (9.0, 7.0)


def sanitize_event_name(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def get_window_bounds(peak_time: float, min_time: float, max_time: float, event_name: str) -> tuple[float, float]:
    before, after = EVENT_WINDOW_BEFORE_AFTER.get(event_name, DEFAULT_WINDOW_BEFORE_AFTER)
    start = max(min_time, peak_time - before)
    end = min(max_time, peak_time + after)
    if end <= start:
        end = min(max_time, start + 1.0)
    return start, end


def resolve_event_interval(event: dict, min_time: float, max_time: float) -> tuple[float, float]:
    """Resolve the cut interval for an event, using pre-computed bounds or re-computing from peak."""
    if "start_time_seconds" in event and "end_time_seconds" in event:
        start = float(event["start_time_seconds"])
        end = float(event["end_time_seconds"])
    else:
        # Recompute from peak time and event type if pre-computed bounds don't exist
        peak = float(event.get("peak_time_seconds", 0.0))
        event_name = event.get("event", "event")
        start, end = get_window_bounds(peak, min_time=min_time, max_time=max_time, event_name=event_name)

    # Clamp to valid range
    start = max(min_time, start)
    end = min(max_time, end)
    if end <= start:
        end = min(max_time, start + 1.0)
    return start, end
