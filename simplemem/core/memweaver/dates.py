"""Session date parsing for MemWeaver.

LoCoMo carries session-level dates only, written as ``"1:56 pm on 8 May, 2023"``.
The fabric stores validity intervals at day resolution (design doc section 2:
"天级分辨率 ... 结构决定，非参数"), so every session is reduced to an ISO date
plus, where available, an ISO datetime used as the entry timestamp.
"""

from datetime import datetime
import re
from typing import Optional, Tuple


_MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# "1:56 pm on 8 May, 2023" / "7:00 am on 24 December 2022"
_LOCOMO_PATTERN = re.compile(
    r"(?P<hour>\d{1,2})\s*:\s*(?P<minute>\d{2})\s*(?P<meridiem>am|pm)?\s*"
    r"on\s+(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)\.?,?\s+(?P<year>\d{4})",
    re.IGNORECASE,
)

# "8 May, 2023" (no clock time)
_DAY_MONTH_YEAR_PATTERN = re.compile(
    r"(?P<day>\d{1,2})\s+(?P<month>[A-Za-z]+)\.?,?\s+(?P<year>\d{4})",
    re.IGNORECASE,
)

_ISO_PATTERN = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2})(?:[T ](?P<time>\d{2}:\d{2}(?::\d{2})?))?")


def parse_session_datetime(raw: Optional[str]) -> Tuple[str, str]:
    """Parse a session timestamp into ``(iso_date, iso_datetime)``.

    Both components are ``""`` when nothing can be parsed; ``iso_datetime`` is
    ``""`` when only a date is available. Parsing is deterministic for the
    LoCoMo and ISO forms; ``dateparser`` is only a last resort.
    """
    if not raw:
        return "", ""

    text = str(raw).strip()

    iso_match = _ISO_PATTERN.search(text)
    if iso_match:
        date = iso_match.group("date")
        time = iso_match.group("time") or ""
        if time and len(time) == 5:
            time = f"{time}:00"
        return date, f"{date}T{time}" if time else ""

    match = _LOCOMO_PATTERN.search(text)
    if match:
        month = _MONTHS.get(match.group("month").lower())
        if month:
            hour = int(match.group("hour")) % 12
            meridiem = (match.group("meridiem") or "").lower()
            if meridiem == "pm":
                hour += 12
            elif not meridiem:
                hour = int(match.group("hour")) % 24
            try:
                stamp = datetime(
                    int(match.group("year")),
                    month,
                    int(match.group("day")),
                    hour,
                    int(match.group("minute")),
                )
            except ValueError:
                return "", ""
            return stamp.date().isoformat(), stamp.isoformat()

    match = _DAY_MONTH_YEAR_PATTERN.search(text)
    if match:
        month = _MONTHS.get(match.group("month").lower())
        if month:
            try:
                day = datetime(
                    int(match.group("year")), month, int(match.group("day"))
                ).date()
            except ValueError:
                return "", ""
            return day.isoformat(), ""

    return _parse_with_dateparser(text)


def to_day(value: Optional[str]) -> str:
    """Reduce any timestamp-ish string to a ``YYYY-MM-DD`` day, or ``""``."""
    if not value:
        return ""
    date, _ = parse_session_datetime(value)
    return date


def _parse_with_dateparser(text: str) -> Tuple[str, str]:
    try:
        import dateparser
    except ImportError:
        return "", ""

    try:
        parsed = dateparser.parse(text)
    except Exception:
        return "", ""

    if not parsed:
        return "", ""
    return parsed.date().isoformat(), parsed.isoformat()
