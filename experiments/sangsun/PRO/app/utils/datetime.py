from datetime import datetime

def to_iso(dt) -> str | None:
    if not dt:
        return None
    if isinstance(dt, datetime):
        return dt.replace(microsecond=0).isoformat() + "Z"
    return str(dt)
