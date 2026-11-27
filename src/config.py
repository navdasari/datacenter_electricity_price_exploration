"""All modeling assumptions in one place.

`ModelConfig.baseline()` reproduces the published results. Each field carries metadata
used by the dashboard and the docs:

  label, help, source   what it is and where the value comes from
  ui                    (min, max, step) for a slider, or a list of options; None = not
                        adjustable in the app (shown read-only)
  reestimate            True if changing it changes the econometric estimates (the double
                        ML and fixed-effects regressions must rerun); False if it only
                        changes a mechanical calculation applied afterwards

`cfg.estimation_key()` hashes only the fields with reestimate=True, so cached estimates
are reused whenever only mechanical assumptions change.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace


def _f(default, label, help, source, ui=None, reestimate=True, **kw):
    meta = dict(label=label, help=help, source=source, ui=ui, reestimate=reestimate)
    if isinstance(default, (dict, list, set, tuple)) and not isinstance(default, (str, bytes)):
        return field(default_factory=lambda d=default: type(d)(d), metadata=meta, **kw)
    return field(default=default, metadata=meta, **kw)


# --------------------------------------------------------------------------- data center load
@dataclass(frozen=True)
class DataCenterLoadConfig:
    national_2025_twh: float = _f(
        230.0, "National data center load, 2025 (TWh)",
        "LBNL's published series ends in 2024 (192 TWh). 2025 is an extrapolation at about 20% growth.",
        "Assumption (LBNL 2025 Update trajectory)", ui=(195.0, 300.0, 5.0))
    national_2026_twh: float = _f(
        275.0, "National data center load, 2026 (TWh)",
        "Extrapolation at about 20% growth from 2025. Affects 2026 year-to-date only.",
        "Assumption (LBNL 2025 Update trajectory)", ui=(200.0, 380.0, 5.0))
    backcast_decline: float = _f(
        0.03, "Backcast before 2014 (decline per year)",
        "National data center load before 2014 is taken back from the 2014 value at this rate.",
        "Assumption", ui=(0.0, 0.10, 0.01))
    remainder_allocation: str = _f(
        "commercial", "Allocating load in states without an EPRI estimate",
        "National load not covered by EPRI's state estimates is spread across the other states "
        "in proportion to this sales measure.",
        "Assumption", ui=["commercial", "commercial_industrial", "total"])
    trend_start: int = _f(
        2012, "Pre-boom trend window: first year",
        "First year of the commercial + industrial trend used for the state-specific measure.",
        "Assumption", ui=(2010, 2016, 1))
    trend_end: int = _f(
        2019, "Pre-boom trend window: last year (and chaining base)",
        "Last year of the trend window. The state-specific path is chained from this year's level.",
        "Assumption", ui=None)
    floor_frac: float = _f(
        0.3, "Floor on the state-specific measure (share of national-scaling value)",
        "Keeps the state-specific load from falling below this fraction of the national-scaling value.",
        "Assumption", ui=(0.0, 0.8, 0.1))
    validation_tol: float = _f(
        0.35, "Validation tolerance vs EPRI 2023",
        "A state uses the state-specific measure only if its implied 2023 load is within this "
        "fraction of EPRI's 2023 estimate.",
        "Assumption", ui=(0.10, 0.80, 0.05))


# --------------------------------------------------------------------------- capacity markets
@dataclass(frozen=True)
class CapacityConfig:
    illinois_pjm_weight: float = _f(
        0.6, "Illinois share of load in PJM",
        "ComEd (northern Illinois) is in PJM; the rest of the state is in MISO.",
        "Assumption", ui=(0.0, 1.0, 0.1))
    use_dominion_zone: bool = _f(
        True, "Use Dominion zone price for Virginia",
        "Use the Dominion zone's capacity price where PJM published one (2025/26: $444/MW-day).",
        "PJM 2025/26 BRA report", ui=[True, False])
    pass_through_vert_int: float = _f(
        0.2, "Capacity pass-through, Virginia and West Virginia",
        "Share of the capacity price reaching residential rates where the utility owns most of "
        "its capacity. Restructured states get full pass-through.",
        "Judgment call", ui=(0.0, 1.0, 0.05))
    residential_peak_load_factor: float = _f(
        0.55, "Residential load factor at peak",
        "Converts capacity prices ($/MW-day) to ¢/kWh of residential load. Lower = residential "
        "customers drive more of the peak, so each kWh carries more capacity cost.",
        "Assumption", ui=(0.35, 0.80, 0.05))
    dc_share_2025_2027: float = _f(
        0.45, "Data center share of PJM capacity costs, 2025/26–2027/28",
        "Share of capacity charges the PJM market monitor attributes to data center load "
        "($21.3B of $47.2B over three auctions).",
        "PJM Independent Market Monitor", ui=(0.0, 0.8, 0.05), reestimate=False)
    dc_share_2028: float = _f(
        0.40, "Data center share of PJM capacity costs, 2028/29",
        "Market monitor attribution for the 2028/29 auction ($6.5B of $16.4B).",
        "PJM Independent Market Monitor", ui=(0.0, 0.8, 0.05), reestimate=False)
    dc_share_band: float = _f(
        0.05, "Uncertainty band on the data center capacity share (±)",
        "Width of the range applied to the share above when computing the 95% range.",
        "Assumption", ui=(0.0, 0.2, 0.01), reestimate=False)

    def dc_cap_share(self) -> dict:
        """Data center share of PJM capacity charges by delivery year (June-May)."""
        return {2025: self.dc_share_2025_2027, 2026: self.dc_share_2025_2027,
                2027: self.dc_share_2025_2027, 2028: self.dc_share_2028}

    def cap_to_ckwh(self) -> float:
        """$/MW-day -> cents/kWh of residential load."""
        return 365 / (8760 * self.residential_peak_load_factor) / 10


# --------------------------------------------------------------------------- spreading
@dataclass(frozen=True)
class SpreadingConfig:
    fixed_share: float = _f(
        0.5, "Fixed-cost share of the residential rate",
        "Share of the residential rate that is fixed cost and gets spread over added data center "
        "load. Only changes the split between the two data center bars, never their net.",
        "Assumption", ui=(0.0, 0.9, 0.05), reestimate=False)


# --------------------------------------------------------------------------- inflation
@dataclass(frozen=True)
class InflationConfig:
    pass_through: float = _f(
        1.0, "Inflation pass-through",
        "How much of CPI inflation passes into residential rates each year. Only splits the "
        "national bar into inflation and above-inflation; the national total is unchanged.",
        "Calibrated on 2012–2019", ui=(0.0, 1.5, 0.05), reestimate=False)
    current_year_weight: float = _f(
        0.5, "Weight on this year's inflation (vs last year's)",
        "Rate cases lag costs, so the inflation term averages this year's and last year's CPI "
        "inflation with this weight on the current year.",
        "Assumption", ui=(0.0, 1.0, 0.1), reestimate=False)


# --------------------------------------------------------------------------- fuel
@dataclass(frozen=True)
class FuelConfig:
    heat_rate: float = _f(
        7.5, "Gas heat rate (MMBtu/MWh)",
        "Converts gas prices to fuel cost per kWh for the gas-cost regressors. It only rescales those "
        "regressors, so it cannot change any result (the coefficients rescale to offset it); kept "
        "for documentation, not adjustable.",
        "Assumption (typical combined-cycle / fleet average)", ui=None)
    retail_choice: tuple = _f(
        ("CT", "DE", "DC", "IL", "ME", "MD", "MA", "NH", "NJ", "NY", "OH", "PA", "RI", "TX"),
        "Restructured retail states", "States where the marginal gas price flows to retail rates.",
        "EIA retail choice status", ui=None)


# --------------------------------------------------------------------------- estimation
@dataclass(frozen=True)
class EstimationConfig:
    exposure: str = _f(
        "state_specific", "Data center load measure",
        "state_specific: state's own commercial + industrial growth where it validates against "
        "EPRI; national_scaling: every state grows at the national rate.",
        "Model choice", ui=["state_specific", "national_scaling"])
    lag: bool = _f(
        True, "Include a one-year lag of data center share",
        "Rates respond to costs with a lag; the lag lets part of the effect arrive a year later.",
        "Model choice", ui=[True, False])
    est_start: int = _f(2010, "Estimation sample: first year", "", "Model choice", ui=None)
    est_end: int = _f(2025, "Estimation sample: last year", "", "Model choice", ui=None)
    n_folds: int = _f(5, "Cross-fitting folds", "", "Model choice", ui=None)
    n_rep: int = _f(5, "Cross-fitting repetitions", "", "Model choice", ui=None)
    national_test_exposure: str = _f(
        "national_scaling", "Load measure for the national data center test",
        "The national test uses the national-scaling series (the published baseline).",
        "Model choice", ui=None)

    def exposure_col(self, which=None) -> str:
        return {"state_specific": "dc_mwh", "national_scaling": "dc_mwh_v1"}[which or self.exposure]


# --------------------------------------------------------------------------- uncertainty
@dataclass(frozen=True)
class UncertaintyConfig:
    z: float = _f(1.96, "Critical value for the range", "1.96 = 95% under normality.",
                  "Convention", ui=None, reestimate=False)
    combine: str = _f(
        "add", "Combining the two uncertainty sources",
        "add: the capacity-share band is added to the sampling range (conservative). "
        "quadrature: combined as independent errors (narrower).",
        "Convention", ui=["add", "quadrature"], reestimate=False)


# --------------------------------------------------------------------------- robustness
@dataclass(frozen=True)
class RobustnessConfig:
    synth_pre: tuple = _f((2010, 2019), "Synthetic control pre-period", "", "Model choice", ui=None)
    synth_post: tuple = _f((2020, 2025), "Synthetic control post-period", "", "Model choice", ui=None)
    donor_max_dc_share: float = _f(3.0, "Donor states: max data center share (%)", "", "Model choice", ui=None)
    donor_exclude: tuple = _f(("CA",), "Donor states excluded", "", "Model choice", ui=None)


GROUP_TITLES = {
    "dc_load": "Data center load: geographic allocation",
    "capacity": "Capacity market",
    "spreading": "Fixed-cost spreading",
    "inflation": "Inflation pass-through",
    "fuel": "Fuel cost",
    "estimation": "Estimation",
    "uncertainty": "Uncertainty ranges",
    "robustness": "Robustness checks",
}


@dataclass(frozen=True)
class ModelConfig:
    dc_load: DataCenterLoadConfig = field(default_factory=DataCenterLoadConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    spreading: SpreadingConfig = field(default_factory=SpreadingConfig)
    inflation: InflationConfig = field(default_factory=InflationConfig)
    fuel: FuelConfig = field(default_factory=FuelConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    @classmethod
    def baseline(cls) -> "ModelConfig":
        return cls()

    # ---- serialization
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        kw = {}
        for g in fields(cls):
            sub = g.default_factory()
            vals = {}
            for f in fields(sub):
                if f.name in d.get(g.name, {}):
                    v = d[g.name][f.name]
                    vals[f.name] = tuple(v) if isinstance(getattr(sub, f.name), tuple) else v
            kw[g.name] = replace(sub, **vals)
        return cls(**kw)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ModelConfig":
        return cls.from_dict(json.loads(s))

    # ---- updates and introspection
    def with_values(self, **dotted) -> "ModelConfig":
        """cfg.with_values(**{"capacity.pass_through_vert_int": 0.5})"""
        groups = {g.name: getattr(self, g.name) for g in fields(self)}
        for k, v in dotted.items():
            g, f = k.split(".")
            groups[g] = replace(groups[g], **{f: v})
        return ModelConfig(**groups)

    def items(self):
        """Yield (group, field_name, value, metadata) for every assumption."""
        for g in fields(self):
            sub = getattr(self, g.name)
            for f in fields(sub):
                yield g.name, f.name, getattr(sub, f.name), dict(f.metadata)

    def estimation_dict(self) -> dict:
        out = {}
        for g, name, val, meta in self.items():
            if meta.get("reestimate") and g != "robustness":
                if isinstance(val, tuple):
                    val = list(val)
                elif isinstance(val, float):
                    val = round(val, 9)  # slider values like 0.49999999999 match 0.5
                out[f"{g}.{name}"] = val
        return out

    def estimation_key(self) -> str:
        """Stable hash of the assumptions that change the econometric estimates."""
        s = json.dumps(self.estimation_dict(), sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]

    def diff(self, other: "ModelConfig") -> dict:
        """Dotted names -> (self value, other value) for fields that differ."""
        a = {f"{g}.{n}": v for g, n, v, _ in self.items()}
        b = {f"{g}.{n}": v for g, n, v, _ in other.items()}
        return {k: (a[k], b[k]) for k in a if a[k] != b[k]}

    def needs_reestimation(self, other: "ModelConfig") -> bool:
        return self.estimation_key() != other.estimation_key()
