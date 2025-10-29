import sys
from datetime import datetime, timezone

def iso_to_us(iso: str) -> int:
    """'2025-06-06T06:03:07Z' -> 1749205387000000 (μs)"""
    if iso.endswith("Z"):
        dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(iso.replace("Z", "")).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000)

if __name__ == "__main__":
    iso = "2025-06-06T06:03:07Z"
    if iso.endswith("Z"):
        dt = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    else:
        dt = datetime.fromisoformat(iso.replace("Z", "")).replace(tzinfo=timezone.utc)
    res = dt.timestamp() * 1_000
    print(res)