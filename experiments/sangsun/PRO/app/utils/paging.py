import base64
import json
from datetime import datetime

def encode_cursor(dt: datetime, pk: int) -> str:
    payload = {"ts": dt.isoformat(timespec="seconds"), "id": pk}
    raw = json.dumps(payload).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")

def decode_cursor(cursor: str) -> tuple[datetime, int]:
    raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
    obj = json.loads(raw.decode("utf-8"))
    ts = datetime.fromisoformat(obj["ts"])
    return ts, int(obj["id"])
