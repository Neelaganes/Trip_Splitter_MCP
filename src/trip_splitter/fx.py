# money exchange related data
import asyncio
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

import httpx2

API = "https://api.frankfurter.dev/v2"
TIMEOUT = httpx2.Timeout(5.0)
ATTEMPTS = 2
RETRY_DELAY = 0.5


class UnknownCurrencyError(ValueError):
    pass


class RatesUnavailableError(Exception):
    pass


@dataclass(frozen=True)
class Rate:
    base: str
    quote: str
    rate: Decimal
    date: date


async def get_rate(base: str, quote: str, on: date, transport: httpx2.AsyncBaseTransport | None = None) -> Rate:
    """What one unit of base was worth in quote on a day (or the closest earlier day with a rate)."""
    if base == quote:
        return Rate(base, quote, Decimal(1), on)
    response = await fetch(f"{API}/rate/{base}/{quote}", {"date": on.isoformat()}, transport)
    if response.status_code == 422:
        raise UnknownCurrencyError(f"Unknown currency {base} or {quote}. Use a 3-letter code like INR, EUR or IDR.")
    if response.status_code != 200:
        raise RatesUnavailableError(f"No {base} to {quote} rate for {on} (HTTP {response.status_code}).")
    try:
        data = json.loads(response.text, parse_float=Decimal)
        rate = Rate(base, quote, Decimal(data["rate"]), date.fromisoformat(data["date"]))
        if rate.rate <= 0:
            raise ValueError(f"rate {rate.rate}")
    except (ValueError, KeyError, TypeError, InvalidOperation) as error:
        raise RatesUnavailableError(f"Frankfurter sent an answer we couldn't read ({error}).") from error
    return rate


async def fetch(url: str, params: dict[str, str], transport: httpx2.AsyncBaseTransport | None) -> httpx2.Response:
    async with httpx2.AsyncClient(timeout=TIMEOUT, transport=transport) as client:
        for attempt in range(1, ATTEMPTS + 1):
            try:
                response = await client.get(url, params=params)
            except httpx2.TransportError as error:
                problem = type(error).__name__
            else:
                if response.status_code < 500 and response.status_code != 429:
                    return response
                problem = f"HTTP {response.status_code}"
            if attempt < ATTEMPTS:
                await asyncio.sleep(RETRY_DELAY)
    raise RatesUnavailableError(f"Exchange rates are unavailable right now ({problem}).")
