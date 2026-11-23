from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

class Category(StrEnum):
    FOOD = "food"
    STAY = "stay"
    TRAVEL = "travel"
    ACTIVITIES = "activities"
    SHOPPING = "shopping"
    OTHER = "other"

ZERO_DECIMAL_CURRENCIES = frozenset({"BIF", "CLP", "DJF", "GNF", "ISK", "JPY", "KMF", "KRW", "PYG", "RWF", "UGX", "VND", "VUV", "XAF", "XOF", "XPF"})
THREE_DECIMAL_CURRENCIES = frozenset({"BHD", "IQD", "JOD", "KWD", "LYD", "OMR", "TND"})

@dataclass(frozen=True)
class Trip:
    id: str
    name: str
    members: tuple[str, ...]
    base_currency: str = "INR"
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if any(not member.strip() for member in self.members):
            raise ValueError(f"Member names can't be empty, got {list(self.members)}.")
        if len({member.casefold() for member in self.members}) != len(self.members):
            raise ValueError(f"Member names must be unique (ignoring case), got {list(self.members)}.")



def _check_split(total_minor: int, names: list[str]) -> None:
    if total_minor <= 0:
        raise ValueError(f"Amount must be positive, got {total_minor} paise.")
    if not names:
        raise ValueError("Split between at least one member.")
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError(f"Each member can appear only once in a split, got {names}.")


def split_equal(total_minor: int, members: list[str], paid_by: str) -> dict[str, int]:
    """Split total_minor (paise) equally among members. Shares always sum to total_minor.

    Leftover paise go one at a time to the payer first (if they are in the split),
    then to the others in alphabetical order, ignoring case.
    """
    _check_split(total_minor, members)

    base, leftover = divmod(total_minor, len(members))
    order = sorted(members, key=lambda name: (name.casefold() != paid_by.casefold(), name.casefold()))
    return {name: base + (1 if i < leftover else 0) for i, name in enumerate(order)}

def allocate(total_minor: int, weights: dict[str, int], paid_by: str) -> dict[str, int]:
    """Share total_minor in proportion to weights, in whole units. Shares always sum to total_minor.

    Leftover units go to the largest remainders, then to the payer first, then alphabetically,
    ignoring case. With equal weights this is exactly split_equal.
    """
    whole = sum(weights.values())
    shares = {name: total_minor * weight // whole for name, weight in weights.items()}
    leftover = total_minor - sum(shares.values())
    order = sorted(
        weights,
        key=lambda name: (-(total_minor * weights[name] % whole), name.casefold() != paid_by.casefold(), name.casefold()),
    )
    for name in order[:leftover]:
        shares[name] += 1
    return shares


def split_exact(total_minor: int, amounts: dict[str, int]) -> dict[str, int]:
    """Use the given per-person amounts (paise) as the shares, after checking them."""
    _check_split(total_minor, list(amounts))
    negative = [name for name, amount in amounts.items() if amount < 0]
    if negative:
        raise ValueError(f"Amounts can't be negative, got {negative}.")
    if sum(amounts.values()) != total_minor:
        raise ValueError(
            f"Exact amounts add up to {sum(amounts.values())} paise "
            f"but the expense is {total_minor} paise."
        )
    return dict(amounts)
    

class RateSource(StrEnum):
    FRANKFURTER = "frankfurter"
    USER = "user"


@dataclass(frozen=True)
class Conversion:
    """How an expense paid in another currency became an amount in the trip's currency."""
    currency: str
    amount_minor: int
    rate: Decimal
    rate_date: date
    source: RateSource = RateSource.FRANKFURTER

    def __post_init__(self) -> None:
        if self.amount_minor <= 0 or self.rate <= 0:
            raise ValueError(f"A conversion needs a positive amount and rate, got {self.amount_minor} and {self.rate}.")


@dataclass(frozen=True)
class Expense:
    paid_by: str
    amount_minor: int
    shares: dict[str, int]
    description: str = ""
    category: Category = Category.OTHER
    note: str = ""
    id: str | None = None
    created_at: datetime | None = None
    spent_on: date | None = None
    conversion: Conversion | None = None

    def __post_init__(self) -> None:
        if sum(self.shares.values()) != self.amount_minor:
            raise ValueError(
                f"Shares add up to {sum(self.shares.values())} paise "
                f"but the expense is {self.amount_minor} paise."
            )
def balances(expenses: list[Expense]) -> dict[str, int]:
    """Net balance per member in paise: what they paid minus their share of everything.
    Positive means the group owes them; negative means they owe the group.
    The balances of any set of expenses sum to zero.
    """
    totals: defaultdict[str, int] = defaultdict(int)
    for expense in expenses:
        totals[expense.paid_by] += expense.amount_minor
        for member, share in expense.shares.items():
            totals[member] -= share
    return dict(totals)

@dataclass(frozen=True)
class Payment:
    from_member: str
    to_member: str
    amount_minor: int


def settle_up(net: dict[str, int]) -> list[Payment]:
    """Payments that bring every balance to zero: at most one fewer than the people involved.

    The person who owes the most pays the person owed the most, as much as they can, until
    everyone is at zero. Ties are broken alphabetically, ignoring case, so the result never varies.
    """
    if sum(net.values()) != 0:
        raise ValueError(f"Balances must sum to zero, got {sum(net.values())} paise.")

    remaining = {name: amount for name, amount in net.items() if amount != 0}
    payments: list[Payment] = []
    while remaining:
        debtor = min(remaining, key=lambda name: (remaining[name], name.casefold()))
        creditor = min(remaining, key=lambda name: (-remaining[name], name.casefold()))
        amount = min(-remaining[debtor], remaining[creditor])
        payments.append(Payment(debtor, creditor, amount))
        remaining[debtor] += amount
        remaining[creditor] -= amount
        remaining = {name: left for name, left in remaining.items() if left != 0}
    return payments


def decimals(currency: str) -> int:
    """How many digits a currency has after the point (ISO 4217): JPY 0, INR 2, KWD 3."""
    if currency in ZERO_DECIMAL_CURRENCIES:
        return 0
    if currency in THREE_DECIMAL_CURRENCIES:
        return 3
    return 2


def to_minor(amount: Decimal, currency: str) -> int:
    """An amount like 300.01 as whole minor units (30001 paise). Refuses more decimals than the currency has."""
    minor = amount.scaleb(decimals(currency))
    if minor != minor.to_integral_value():
        raise ValueError(f"{currency} amounts have at most {decimals(currency)} decimals, got {amount}.")
    return int(minor)


def to_major(amount_minor: int, currency: str) -> str:
    """Whole minor units as text in the currency's own format: 30001 INR is "300.01", 500 JPY is "500"."""
    return str(Decimal(amount_minor).scaleb(-decimals(currency)))


def convert(amount_minor: int, currency: str, rate: Decimal, to_currency: str) -> int:
    """Convert minor units of one currency into minor units of another. Rounded once, half up."""
    exact = Decimal(amount_minor).scaleb(-decimals(currency)) * rate
    converted = int(exact.scaleb(decimals(to_currency)).quantize(Decimal(1), rounding=ROUND_HALF_UP))
    if converted <= 0:
        raise ValueError(f"{to_major(amount_minor, currency)} {currency} is less than the smallest {to_currency} amount.")
    return converted


def implied_rate(amount_minor: int, currency: str, converted_minor: int, to_currency: str) -> Decimal:
    """The rate that turns amount_minor of currency into exactly converted_minor of to_currency."""
    return Decimal(to_major(converted_minor, to_currency)) / Decimal(to_major(amount_minor, currency))

