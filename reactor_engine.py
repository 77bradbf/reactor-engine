#!/usr/bin/env python3
"""
Pure Python reactor engineering engine.

Solves common reactor design cases from kinetics and design equations:
  - CSTR volume for a target conversion
  - CSTR conversion for a specified volume
  - Isothermal PFR conversion profile
  - Non-isothermal PFR conversion and temperature profiles

Default kinetics:
    k = A exp(-Ea / RT)
    A = 5.7e5
    Ea = 33400 J/mol
    R = 8.314 J/(mol K)

Default rate law:
    -r_A = k C_A C_B

All units must be self-consistent. The examples below use:
    mol, L, s, J, K
"""

from __future__ import annotations

import argparse
import csv
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar


R_GAS = 8.314
DEFAULT_A = 5.7e5
DEFAULT_EA = 33_400.0
EPS = 1.0e-12


@dataclass(frozen=True)
class Kinetics:
    """Arrhenius kinetic parameters."""

    pre_exponential: float = DEFAULT_A
    activation_energy: float = DEFAULT_EA
    gas_constant: float = R_GAS

    def k(self, temperature: float) -> float:
        """Return rate constant at temperature in K."""
        if temperature <= 0:
            raise ValueError("Temperature must be greater than 0 K.")
        return float(
            self.pre_exponential
            * np.exp(-self.activation_energy / (self.gas_constant * temperature))
        )


@dataclass(frozen=True)
class Feed:
    """Inlet molar flows and volumetric flow."""

    fa0: float = 10.0
    fb0: float = 10.0
    fi0: float = 0.0
    v0: float = 100.0
    t0: float = 350.0

    @property
    def ca0(self) -> float:
        return self.fa0 / self.v0


@dataclass(frozen=True)
class Stoichiometry:
    """Stoichiometry and gas expansion data for A + bB -> pP."""

    b_per_a: float = 1.0
    product_per_a: float = 1.0
    gas_phase: bool = False
    y_a0: float = 1.0
    delta: float = 0.0

    @property
    def epsilon(self) -> float:
        return self.y_a0 * self.delta if self.gas_phase else 0.0


@dataclass(frozen=True)
class HeatTransfer:
    """Non-isothermal PFR heat-balance parameters."""

    delta_h_rxn: float = -50_000.0
    ua: float = 0.0
    t_a: float = 350.0
    cp_a: float = 100.0
    cp_b: float = 100.0
    cp_p: float = 100.0
    cp_i: float = 100.0


@dataclass(frozen=True)
class ReactorState:
    """Molar flows, concentrations, and volumetric flow at a reactor point."""

    fa: float
    fb: float
    fp: float
    fi: float
    ca: float
    cb: float
    cp: float
    ci: float
    volumetric_flow: float


@dataclass(frozen=True)
class ReactorResult:
    """Computed reactor result and profile arrays."""

    mode: str
    volume: float
    conversion: float
    exit_temperature: float
    selectivity: float
    volumes: np.ndarray
    conversions: np.ndarray
    temperatures: np.ndarray | None = None


@dataclass(frozen=True)
class SeriesReactionResult:
    """Computed result for first-order series reactions A -> B -> C."""

    mode: str
    independent_variable_name: str
    optimum_value: float
    ca_at_optimum: float
    cb_at_optimum: float
    cc_at_optimum: float
    selectivity_b_to_c: float
    yield_b: float
    independent_values: np.ndarray
    ca: np.ndarray
    cb: np.ndarray
    cc: np.ndarray


@dataclass(frozen=True)
class PowerLawFitResult:
    """Power-law kinetic fit result for r = k x1^a x2^b."""

    mode: str
    reactant_1_name: str
    reactant_2_name: str
    product_name: str
    alpha: float
    beta: float
    k: float
    r_squared: float
    rate_units: str
    independent_units: str
    observed_rates: np.ndarray
    predicted_rates: np.ndarray


@dataclass(frozen=True)
class SpeciesPropertyResult:
    """Pure-species property result from an advanced property backend."""

    mode: str
    species: str
    backend: str
    temperature: float
    pressure: float
    properties: dict[str, str]


@dataclass(frozen=True)
class ParallelCstrResult:
    """CSTR optimization result for parallel desired/undesired reactions."""

    mode: str
    ca_optimum: float
    cb_exit: float
    rate_d: float
    rate_u1: float
    rate_u2: float
    total_rate: float
    selectivity: float
    residence_time: float
    volumetric_flow_rate: float
    previous_profit_per_liter: float
    optimized_profit_per_liter: float
    extra_profit_per_liter: float
    extra_profit_per_time: float
    ca_values: np.ndarray
    selectivity_values: np.ndarray


def limiting_conversion(feed: Feed, stoich: Stoichiometry) -> float:
    """Maximum conversion before A or B is exhausted."""
    limits = [1.0]
    if stoich.b_per_a > 0.0:
        limits.append(feed.fb0 / (stoich.b_per_a * feed.fa0))
    return max(0.0, min(limits))


def molar_flows(conversion: float, feed: Feed, stoich: Stoichiometry) -> tuple[float, float, float, float]:
    """Return FA, FB, FP, FI at conversion X."""
    x = float(np.clip(conversion, 0.0, limiting_conversion(feed, stoich)))
    fa = feed.fa0 * (1.0 - x)
    fb = feed.fb0 - stoich.b_per_a * feed.fa0 * x
    fp = stoich.product_per_a * feed.fa0 * x
    fi = feed.fi0
    return fa, fb, fp, fi


def volumetric_flow(
    conversion: float,
    feed: Feed,
    stoich: Stoichiometry,
    temperature: float | None = None,
    ideal_gas_temperature_correction: bool = False,
) -> float:
    """
    Return volumetric flow.

    Required gas expansion model:
        v = v0 (1 + epsilon X)

    The optional temperature correction is useful for ideal-gas non-isothermal
    work, but defaults off to match the stated design rule exactly.
    """
    expansion_factor = 1.0 + stoich.epsilon * conversion
    if ideal_gas_temperature_correction:
        if temperature is None:
            raise ValueError("Temperature is required for ideal-gas correction.")
        expansion_factor *= temperature / feed.t0
    return max(EPS, feed.v0 * expansion_factor)


def state_at(
    conversion: float,
    temperature: float,
    feed: Feed,
    stoich: Stoichiometry,
    ideal_gas_temperature_correction: bool = False,
) -> ReactorState:
    """Return reactor state at a conversion and temperature."""
    fa, fb, fp, fi = molar_flows(conversion, feed, stoich)
    v = volumetric_flow(
        conversion,
        feed,
        stoich,
        temperature=temperature,
        ideal_gas_temperature_correction=ideal_gas_temperature_correction,
    )
    return ReactorState(
        fa=fa,
        fb=fb,
        fp=fp,
        fi=fi,
        ca=max(0.0, fa) / v,
        cb=max(0.0, fb) / v,
        cp=max(0.0, fp) / v,
        ci=max(0.0, fi) / v,
        volumetric_flow=v,
    )


def reaction_rate(
    conversion: float,
    temperature: float,
    kinetics: Kinetics,
    feed: Feed,
    stoich: Stoichiometry,
    ideal_gas_temperature_correction: bool = False,
) -> float:
    """Return positive disappearance rate -r_A."""
    state = state_at(
        conversion,
        temperature,
        feed,
        stoich,
        ideal_gas_temperature_correction=ideal_gas_temperature_correction,
    )
    return kinetics.k(temperature) * state.ca * state.cb


def heat_capacity_flow(
    conversion: float,
    feed: Feed,
    stoich: Stoichiometry,
    heat: HeatTransfer,
) -> float:
    """Return sum(F_i Cp_i), units of energy / time / K."""
    fa, fb, fp, fi = molar_flows(conversion, feed, stoich)
    cp_flow = fa * heat.cp_a + fb * heat.cp_b + fp * heat.cp_p + fi * heat.cp_i
    return max(EPS, cp_flow)


def single_reaction_selectivity(conversion: float, feed: Feed, stoich: Stoichiometry) -> float:
    """
    Return P/U selectivity.

    This engine models one desired reaction by default, so the undesired-product
    flow is zero and P/U selectivity is infinite for any positive conversion.
    """
    desired_product = stoich.product_per_a * feed.fa0 * max(0.0, conversion)
    if desired_product <= EPS:
        return 0.0
    return float("inf")


def cstr_volume_for_conversion(
    target_conversion: float,
    temperature: float,
    kinetics: Kinetics,
    feed: Feed,
    stoich: Stoichiometry,
) -> float:
    """Solve V = FA0 X / (-r_A) for CSTR volume."""
    max_x = limiting_conversion(feed, stoich)
    if not 0.0 <= target_conversion < max_x:
        raise ValueError(
            f"Target conversion must be in [0, {max_x:.6g}) for this feed."
        )
    rate = reaction_rate(target_conversion, temperature, kinetics, feed, stoich)
    if rate <= EPS:
        return float("inf")
    return feed.fa0 * target_conversion / rate


def cstr_conversion_for_volume(
    volume: float,
    temperature: float,
    kinetics: Kinetics,
    feed: Feed,
    stoich: Stoichiometry,
) -> float:
    """Solve the CSTR design equation for conversion at a specified volume."""
    if volume < 0.0:
        raise ValueError("Volume must be nonnegative.")
    if volume == 0.0:
        return 0.0

    max_x = limiting_conversion(feed, stoich)
    upper = max(EPS, max_x * (1.0 - 1.0e-9))

    def residual(x: float) -> float:
        return cstr_volume_for_conversion(x, temperature, kinetics, feed, stoich) - volume

    return brentq(residual, EPS, upper, xtol=1.0e-10, rtol=1.0e-10, maxiter=200)


def solve_pfr(
    volume: float,
    temperature: float,
    kinetics: Kinetics,
    feed: Feed,
    stoich: Stoichiometry,
    points: int = 250,
) -> ReactorResult:
    """Integrate the isothermal PFR design equation."""
    if volume <= 0.0:
        raise ValueError("PFR volume must be greater than zero.")

    max_x = limiting_conversion(feed, stoich)
    terminal_x = max_x * (1.0 - 1.0e-8)

    def ode(v_reactor: float, y: Iterable[float]) -> list[float]:
        del v_reactor
        x = float(y[0])
        rate = reaction_rate(x, temperature, kinetics, feed, stoich)
        return [rate / feed.fa0]

    def exhausted_event(v_reactor: float, y: Iterable[float]) -> float:
        del v_reactor
        return terminal_x - float(y[0])

    exhausted_event.terminal = True  # type: ignore[attr-defined]
    exhausted_event.direction = -1  # type: ignore[attr-defined]

    t_eval = np.linspace(0.0, volume, points)
    solution = solve_ivp(
        ode,
        (0.0, volume),
        [0.0],
        t_eval=t_eval,
        events=exhausted_event,
        max_step=max(volume / points, EPS),
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    volumes = solution.t
    conversions = np.clip(solution.y[0], 0.0, max_x)
    final_x = float(conversions[-1])
    return ReactorResult(
        mode="pfr",
        volume=float(volumes[-1]),
        conversion=final_x,
        exit_temperature=temperature,
        selectivity=single_reaction_selectivity(final_x, feed, stoich),
        volumes=volumes,
        conversions=conversions,
        temperatures=None,
    )


def solve_nonisothermal_pfr(
    volume: float,
    kinetics: Kinetics,
    feed: Feed,
    stoich: Stoichiometry,
    heat: HeatTransfer,
    points: int = 250,
    ideal_gas_temperature_correction: bool = False,
) -> ReactorResult:
    """Integrate coupled PFR material and energy balances."""
    if volume <= 0.0:
        raise ValueError("PFR volume must be greater than zero.")

    max_x = limiting_conversion(feed, stoich)
    terminal_x = max_x * (1.0 - 1.0e-8)

    def ode(v_reactor: float, y: Iterable[float]) -> list[float]:
        del v_reactor
        x = float(np.clip(y[0], 0.0, terminal_x))
        temperature = max(EPS, float(y[1]))
        rate = reaction_rate(
            x,
            temperature,
            kinetics,
            feed,
            stoich,
            ideal_gas_temperature_correction=ideal_gas_temperature_correction,
        )
        dx_dv = rate / feed.fa0
        heat_generation = rate * (-heat.delta_h_rxn)
        heat_exchange = heat.ua * (heat.t_a - temperature)
        dt_dv = (heat_exchange + heat_generation) / heat_capacity_flow(
            x, feed, stoich, heat
        )
        return [dx_dv, dt_dv]

    def exhausted_event(v_reactor: float, y: Iterable[float]) -> float:
        del v_reactor
        return terminal_x - float(y[0])

    exhausted_event.terminal = True  # type: ignore[attr-defined]
    exhausted_event.direction = -1  # type: ignore[attr-defined]

    t_eval = np.linspace(0.0, volume, points)
    solution = solve_ivp(
        ode,
        (0.0, volume),
        [0.0, feed.t0],
        t_eval=t_eval,
        events=exhausted_event,
        max_step=max(volume / points, EPS),
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    volumes = solution.t
    conversions = np.clip(solution.y[0], 0.0, max_x)
    temperatures = solution.y[1]
    final_x = float(conversions[-1])
    final_t = float(temperatures[-1])
    return ReactorResult(
        mode="nonisothermal-pfr",
        volume=float(volumes[-1]),
        conversion=final_x,
        exit_temperature=final_t,
        selectivity=single_reaction_selectivity(final_x, feed, stoich),
        volumes=volumes,
        conversions=conversions,
        temperatures=temperatures,
    )


def cstr_result(
    volume: float | None,
    target_conversion: float | None,
    temperature: float,
    kinetics: Kinetics,
    feed: Feed,
    stoich: Stoichiometry,
    points: int = 250,
) -> ReactorResult:
    """Solve a CSTR for either volume or conversion and build a plotting profile."""
    max_x = limiting_conversion(feed, stoich)
    x_grid = np.linspace(EPS, max_x * (1.0 - 1.0e-6), points)
    v_grid_raw = np.array(
        [
            cstr_volume_for_conversion(x, temperature, kinetics, feed, stoich)
            for x in x_grid
        ]
    )
    finite_mask = np.isfinite(v_grid_raw)
    x_grid = x_grid[finite_mask]
    v_grid = v_grid_raw[finite_mask]

    if target_conversion is not None:
        solved_x = target_conversion
        solved_v = cstr_volume_for_conversion(
            target_conversion, temperature, kinetics, feed, stoich
        )
    elif volume is not None:
        solved_v = volume
        solved_x = cstr_conversion_for_volume(volume, temperature, kinetics, feed, stoich)
    else:
        raise ValueError("Specify either volume or target conversion for a CSTR.")

    return ReactorResult(
        mode="cstr",
        volume=float(solved_v),
        conversion=float(solved_x),
        exit_temperature=temperature,
        selectivity=single_reaction_selectivity(float(solved_x), feed, stoich),
        volumes=v_grid,
        conversions=x_grid,
        temperatures=None,
    )


def validate_series_inputs(k1: float, k2: float, ca0: float) -> None:
    """Validate first-order series reaction parameters."""
    if k1 <= 0.0:
        raise ValueError("series-k1 must be greater than zero.")
    if k2 <= 0.0:
        raise ValueError("series-k2 must be greater than zero.")
    if ca0 <= 0.0:
        raise ValueError("ca0 must be greater than zero.")


def series_batch_concentrations(
    time_h: np.ndarray | float,
    k1: float,
    k2: float,
    ca0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytical constant-volume batch solution for A -> B -> C."""
    validate_series_inputs(k1, k2, ca0)
    t = np.asarray(time_h, dtype=float)
    ca = ca0 * np.exp(-k1 * t)
    if np.isclose(k1, k2):
        cb = ca0 * k1 * t * np.exp(-k1 * t)
    else:
        cb = ca0 * k1 / (k2 - k1) * (np.exp(-k1 * t) - np.exp(-k2 * t))
    cc = ca0 - ca - cb
    return ca, cb, cc


def series_cstr_concentrations(
    tau_h: np.ndarray | float,
    k1: float,
    k2: float,
    ca0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steady-state CSTR solution for A -> B -> C as a function of tau."""
    validate_series_inputs(k1, k2, ca0)
    tau = np.asarray(tau_h, dtype=float)
    ca = ca0 / (1.0 + k1 * tau)
    cb = k1 * tau * ca / (1.0 + k2 * tau)
    cc = ca0 - ca - cb
    return ca, cb, cc


def series_batch_time_for_max_b(k1: float, k2: float) -> float:
    """Time where B is maximum for first-order batch A -> B -> C."""
    if np.isclose(k1, k2):
        return 1.0 / k1
    return float(np.log(k1 / k2) / (k1 - k2))


def series_cstr_tau_for_max_b(k1: float, k2: float) -> float:
    """Residence time where B is maximum for first-order CSTR A -> B -> C."""
    return float(1.0 / np.sqrt(k1 * k2))


def series_selectivity_b_to_c(cb: float, cc: float) -> float:
    """Overall selectivity of desired intermediate B relative to undesired C."""
    if cc <= EPS:
        return float("inf")
    return cb / cc


def series_yield_b(ca0: float, ca: float, cb: float) -> float:
    """Overall yield of B based on A consumed."""
    a_consumed = ca0 - ca
    if a_consumed <= EPS:
        return 0.0
    return cb / a_consumed


def solve_series_batch(
    k1: float,
    k2: float,
    ca0: float,
    time_start: float,
    time_end: float,
    points: int,
) -> SeriesReactionResult:
    """Solve and summarize first-order batch series reactions."""
    if time_end <= time_start:
        raise ValueError("time-end must be greater than time-start.")
    times = np.linspace(time_start, time_end, points)
    ca, cb, cc = series_batch_concentrations(times, k1, k2, ca0)
    optimum_time = series_batch_time_for_max_b(k1, k2)
    ca_opt, cb_opt, cc_opt = [
        float(value) for value in series_batch_concentrations(optimum_time, k1, k2, ca0)
    ]
    return SeriesReactionResult(
        mode="series-batch",
        independent_variable_name="time",
        optimum_value=optimum_time,
        ca_at_optimum=ca_opt,
        cb_at_optimum=cb_opt,
        cc_at_optimum=cc_opt,
        selectivity_b_to_c=series_selectivity_b_to_c(cb_opt, cc_opt),
        yield_b=series_yield_b(ca0, ca_opt, cb_opt),
        independent_values=times,
        ca=ca,
        cb=cb,
        cc=cc,
    )


def solve_series_cstr(
    k1: float,
    k2: float,
    ca0: float,
    tau_start: float,
    tau_end: float,
    points: int,
) -> SeriesReactionResult:
    """Solve and summarize first-order CSTR series reactions."""
    if tau_end <= tau_start:
        raise ValueError("time-end must be greater than time-start.")
    tau = np.linspace(tau_start, tau_end, points)
    ca, cb, cc = series_cstr_concentrations(tau, k1, k2, ca0)
    optimum_tau = series_cstr_tau_for_max_b(k1, k2)
    ca_opt, cb_opt, cc_opt = [
        float(value) for value in series_cstr_concentrations(optimum_tau, k1, k2, ca0)
    ]
    return SeriesReactionResult(
        mode="series-cstr",
        independent_variable_name="residence time",
        optimum_value=optimum_tau,
        ca_at_optimum=ca_opt,
        cb_at_optimum=cb_opt,
        cc_at_optimum=cc_opt,
        selectivity_b_to_c=series_selectivity_b_to_c(cb_opt, cc_opt),
        yield_b=series_yield_b(ca0, ca_opt, cb_opt),
        independent_values=tau,
        ca=ca,
        cb=cb,
        cc=cc,
    )


def column_letters_to_index(cell_ref: str) -> int:
    """Convert Excel cell letters to zero-based column index."""
    letters = "".join(char for char in cell_ref if char.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return index - 1


def read_simple_xlsx_rows(xlsx_path: Path) -> list[dict[str, str]]:
    """
    Read the first worksheet of a simple .xlsx table using only stdlib.

    Expected shape:
        first row = headers
        following rows = data
    """
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(xlsx_path) as workbook_zip:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in workbook_zip.namelist():
            shared_root = ET.fromstring(workbook_zip.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("main:si", ns):
                text_parts = [
                    text_node.text or "" for text_node in item.findall(".//main:t", ns)
                ]
                shared_strings.append("".join(text_parts))

        sheet_name = "xl/worksheets/sheet1.xml"
        if sheet_name not in workbook_zip.namelist():
            worksheet_names = [
                name
                for name in workbook_zip.namelist()
                if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
            ]
            if not worksheet_names:
                raise ValueError(f"No worksheet found in {xlsx_path}.")
            sheet_name = sorted(worksheet_names)[0]

        worksheet_root = ET.fromstring(workbook_zip.read(sheet_name))

    table: list[list[str]] = []
    for row in worksheet_root.findall(".//main:sheetData/main:row", ns):
        values: list[str] = []
        for cell in row.findall("main:c", ns):
            cell_ref = cell.attrib.get("r", "")
            column_index = column_letters_to_index(cell_ref)
            while len(values) <= column_index:
                values.append("")

            cell_type = cell.attrib.get("t", "")
            value_node = cell.find("main:v", ns)
            inline_text_node = cell.find("main:is/main:t", ns)
            if cell_type == "s" and value_node is not None:
                shared_index = int(value_node.text or 0)
                values[column_index] = shared_strings[shared_index]
            elif cell_type == "inlineStr" and inline_text_node is not None:
                values[column_index] = inline_text_node.text or ""
            elif value_node is not None:
                values[column_index] = value_node.text or ""
            else:
                values[column_index] = ""
        if any(value.strip() for value in values):
            table.append(values)

    if len(table) < 2:
        return []

    headers = [header.strip() for header in table[0]]
    rows: list[dict[str, str]] = []
    for values in table[1:]:
        row_dict = {
            header: values[index].strip() if index < len(values) else ""
            for index, header in enumerate(headers)
            if header
        }
        if any(value for value in row_dict.values()):
            rows.append(row_dict)
    return rows


def read_table_rows(data_path: Path) -> list[dict[str, str]]:
    """Read tabular fit data from CSV or XLSX.

    If pandas/openpyxl are installed, use them for more robust real spreadsheet
    handling. Otherwise, fall back to the lightweight readers built into this
    file so the basic homework modes still work with only NumPy/SciPy.
    """
    suffix = data_path.suffix.lower()
    try:
        import pandas as pd
    except ImportError:
        pd = None

    if pd is not None and suffix in {".csv", ".xlsx"}:
        if suffix == ".csv":
            dataframe = pd.read_csv(data_path)
        else:
            dataframe = pd.read_excel(data_path)
        dataframe = dataframe.dropna(how="all")
        dataframe.columns = [str(column).strip() for column in dataframe.columns]
        return [
            {
                str(key).strip(): "" if value is None else str(value).strip()
                for key, value in row.items()
            }
            for row in dataframe.to_dict(orient="records")
        ]

    if suffix == ".csv":
        with data_path.open(newline="") as file:
            return list(csv.DictReader(file))
    if suffix == ".xlsx":
        return read_simple_xlsx_rows(data_path)
    raise ValueError(
        f"Unsupported data file type '{data_path.suffix}'. Use .csv or .xlsx."
    )


def require_numeric_column(rows: list[dict[str, str]], column: str) -> np.ndarray:
    """Return a numeric column with clear errors for blank or invalid cells."""
    values: list[float] = []
    for row_index, row in enumerate(rows, start=2):
        raw_value = (row.get(column) or "").strip()
        if raw_value == "":
            raise ValueError(
                f"Blank value in column '{column}' on spreadsheet row {row_index}. "
                "Check that every data row has x1, x2, and rate OR total_flow, x1, x2, y_product."
            )
        try:
            values.append(float(raw_value))
        except ValueError as exc:
            raise ValueError(
                f"Could not convert column '{column}' on spreadsheet row {row_index} "
                f"to a number: {raw_value!r}."
            ) from exc
    return np.array(values)


def load_power_law_fit_data(
    csv_path: Path | None,
    catalyst_weight_g: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str, str, str]:
    """
    Load data for power-law kinetic fitting.

    If no CSV is supplied, the default data set is the ethane hydrogenolysis
    problem from the photo:
        H2 + C2H6 -> 2 CH4
        r'_CH4 = k P_C2H6^alpha P_H2^beta

    CSV options for future problems:
      1. Direct rate data with columns: x1,x2,rate
      2. Product-flow CSTR data with columns: total_flow,x1,x2,y_product
    """
    if csv_path is None:
        total_flow_mol_h = np.array([0.61, 0.60, 0.30, 0.72, 0.52])
        p_c2h6_atm = np.array([0.50, 1.00, 0.40, 0.60, 0.60])
        p_h2_atm = np.array([0.51, 0.54, 0.60, 0.60, 0.40])
        y_ch4 = np.array([0.07, 0.16, 0.16, 0.10, 0.06])
        rate = total_flow_mol_h * y_ch4 / catalyst_weight_g
        return (
            p_c2h6_atm,
            p_h2_atm,
            rate,
            "C2H6",
            "H2",
            "CH4",
            "mol CH4/(g_cat h)",
            "atm",
        )

    rows = read_table_rows(csv_path)
    if not rows:
        raise ValueError(
            f"No data rows found in {csv_path}. "
            "Make sure the file is saved and contains a header plus at least 3 data rows."
        )

    required_direct = {"x1", "x2", "rate"}
    required_product_flow = {"total_flow", "x1", "x2", "y_product"}
    columns = set(rows[0])
    x1 = require_numeric_column(rows, "x1")
    x2 = require_numeric_column(rows, "x2")

    if required_direct.issubset(columns):
        rate = require_numeric_column(rows, "rate")
    elif required_product_flow.issubset(columns):
        total_flow = require_numeric_column(rows, "total_flow")
        y_product = require_numeric_column(rows, "y_product")
        rate = total_flow * y_product / catalyst_weight_g
    else:
        raise ValueError(
            "CSV must contain either x1,x2,rate or total_flow,x1,x2,y_product columns."
        )

    return x1, x2, rate, "reactant 1", "reactant 2", "product", "rate units", "x units"


def fit_power_law_rate(
    x1: np.ndarray,
    x2: np.ndarray,
    rate: np.ndarray,
    reactant_1_name: str,
    reactant_2_name: str,
    product_name: str,
    rate_units: str,
    independent_units: str,
) -> PowerLawFitResult:
    """Fit r = k x1^alpha x2^beta by linear regression of log(rate)."""
    if np.any(x1 <= 0.0) or np.any(x2 <= 0.0) or np.any(rate <= 0.0):
        raise ValueError("Power-law fitting requires positive x1, x2, and rate values.")

    design = np.column_stack([np.ones_like(x1), np.log(x1), np.log(x2)])
    coefficients, *_ = np.linalg.lstsq(design, np.log(rate), rcond=None)
    log_k, alpha, beta = coefficients
    predicted_log_rate = design @ coefficients
    predicted_rate = np.exp(predicted_log_rate)

    residual_sum = float(np.sum((np.log(rate) - predicted_log_rate) ** 2))
    total_sum = float(np.sum((np.log(rate) - np.mean(np.log(rate))) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum > EPS else 1.0

    return PowerLawFitResult(
        mode="kinetic-fit-power-law",
        reactant_1_name=reactant_1_name,
        reactant_2_name=reactant_2_name,
        product_name=product_name,
        alpha=float(alpha),
        beta=float(beta),
        k=float(np.exp(log_k)),
        r_squared=r_squared,
        rate_units=rate_units,
        independent_units=independent_units,
        observed_rates=rate,
        predicted_rates=predicted_rate,
    )


def solve_kinetic_fit_power_law(
    csv_path: Path | None,
    catalyst_weight_g: float,
) -> PowerLawFitResult:
    """Solve a two-reactant power-law kinetic fitting problem."""
    data = load_power_law_fit_data(csv_path, catalyst_weight_g)
    return fit_power_law_rate(*data)


def parallel_rate(
    k: float,
    ca: np.ndarray | float,
    cb: np.ndarray | float,
    exponent_a: float,
    exponent_b: float,
) -> np.ndarray:
    """Return a power-law liquid-phase rate k CA^a CB^b."""
    return k * np.power(np.maximum(ca, 0.0), exponent_a) * np.power(
        np.maximum(cb, 0.0), exponent_b
    )


def parallel_selectivity(
    ca: np.ndarray | float,
    cb: np.ndarray | float,
    k_d: float,
    k_u1: float,
    k_u2: float,
    d_exp_a: float,
    d_exp_b: float,
    u1_exp_a: float,
    u1_exp_b: float,
    u2_exp_a: float,
    u2_exp_b: float,
) -> np.ndarray:
    """Instantaneous selectivity of desired D relative to U1 + U2."""
    rate_d = parallel_rate(k_d, ca, cb, d_exp_a, d_exp_b)
    rate_u1 = parallel_rate(k_u1, ca, cb, u1_exp_a, u1_exp_b)
    rate_u2 = parallel_rate(k_u2, ca, cb, u2_exp_a, u2_exp_b)
    return rate_d / np.maximum(rate_u1 + rate_u2, EPS)


def net_profit_per_liter(
    d_concentration: float,
    u1_concentration: float,
    u2_concentration: float,
    desired_price: float,
    undesired_cost: float,
) -> float:
    """Return net product value per liter of exit stream."""
    return (
        desired_price * d_concentration
        - undesired_cost * u1_concentration
        - undesired_cost * u2_concentration
    )


def solve_parallel_cstr(
    ca0: float,
    cb0: float,
    cb_exit: float,
    volume: float,
    k_d: float,
    k_u1: float,
    k_u2: float,
    d_exp_a: float,
    d_exp_b: float,
    u1_exp_a: float,
    u1_exp_b: float,
    u2_exp_a: float,
    u2_exp_b: float,
    previous_d: float,
    previous_u1: float,
    previous_u2: float,
    optimized_d: float,
    optimized_u1: float,
    optimized_u2: float,
    desired_price: float,
    undesired_cost: float,
    points: int,
) -> ParallelCstrResult:
    """Optimize liquid-phase parallel reaction selectivity in a CSTR."""
    if ca0 <= 0.0 or cb0 <= 0.0:
        raise ValueError("parallel-ca0 and parallel-cb0 must be greater than zero.")
    if not 0.0 < cb_exit < cb0:
        raise ValueError("parallel-exit-cb must be between 0 and parallel-cb0.")
    if volume <= 0.0:
        raise ValueError("parallel-volume must be greater than zero.")

    upper_ca = ca0 * (1.0 - 1.0e-9)
    lower_ca = max(EPS, ca0 * 1.0e-9)

    def negative_selectivity(ca: float) -> float:
        return -float(
            parallel_selectivity(
                ca,
                cb_exit,
                k_d,
                k_u1,
                k_u2,
                d_exp_a,
                d_exp_b,
                u1_exp_a,
                u1_exp_b,
                u2_exp_a,
                u2_exp_b,
            )
        )

    optimum = minimize_scalar(
        negative_selectivity,
        bounds=(lower_ca, upper_ca),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    ca_opt = float(optimum.x)
    rate_d = float(parallel_rate(k_d, ca_opt, cb_exit, d_exp_a, d_exp_b))
    rate_u1 = float(parallel_rate(k_u1, ca_opt, cb_exit, u1_exp_a, u1_exp_b))
    rate_u2 = float(parallel_rate(k_u2, ca_opt, cb_exit, u2_exp_a, u2_exp_b))
    total_rate = rate_d + rate_u1 + rate_u2
    residence_time = (ca0 - ca_opt) / max(total_rate, EPS)
    volumetric_flow_rate = volume / max(residence_time, EPS)

    previous_profit = net_profit_per_liter(
        previous_d, previous_u1, previous_u2, desired_price, undesired_cost
    )
    optimized_profit = net_profit_per_liter(
        optimized_d, optimized_u1, optimized_u2, desired_price, undesired_cost
    )
    extra_profit_per_liter = optimized_profit - previous_profit
    extra_profit_per_time = extra_profit_per_liter * volumetric_flow_rate

    ca_values = np.linspace(lower_ca, ca0, points)
    selectivity_values = parallel_selectivity(
        ca_values,
        cb_exit,
        k_d,
        k_u1,
        k_u2,
        d_exp_a,
        d_exp_b,
        u1_exp_a,
        u1_exp_b,
        u2_exp_a,
        u2_exp_b,
    )

    return ParallelCstrResult(
        mode="parallel-cstr",
        ca_optimum=ca_opt,
        cb_exit=cb_exit,
        rate_d=rate_d,
        rate_u1=rate_u1,
        rate_u2=rate_u2,
        total_rate=total_rate,
        selectivity=rate_d / max(rate_u1 + rate_u2, EPS),
        residence_time=residence_time,
        volumetric_flow_rate=volumetric_flow_rate,
        previous_profit_per_liter=previous_profit,
        optimized_profit_per_liter=optimized_profit,
        extra_profit_per_liter=extra_profit_per_liter,
        extra_profit_per_time=extra_profit_per_time,
        ca_values=ca_values,
        selectivity_values=selectivity_values,
    )


def species_molar_cp(
    species: str,
    temperature: float,
    pressure: float,
    backend: str,
) -> float:
    """Return molar heat capacity in J/(mol K) from thermo or CoolProp."""
    backend_normalized = backend.lower()
    errors: list[str] = []

    if backend_normalized in {"auto", "thermo"}:
        try:
            from thermo import Chemical

            chemical = Chemical(species, T=temperature, P=pressure)
            candidates = [
                getattr(chemical, "Cp", None),
                getattr(chemical, "Cpg", None),
                getattr(chemical, "Cpl", None),
            ]
            for value in candidates:
                if value is None:
                    continue
                cp = float(value)
                if np.isfinite(cp) and cp > 0.0:
                    return cp
            errors.append(f"thermo returned no usable Cp for {species}")
        except Exception as exc:
            errors.append(f"thermo: {exc}")

    if backend_normalized in {"auto", "coolprop"}:
        try:
            from CoolProp.CoolProp import PropsSI

            cp = float(PropsSI("Cpmolar", "T", temperature, "P", pressure, species))
            if np.isfinite(cp) and cp > 0.0:
                return cp
            errors.append(f"CoolProp returned no usable Cp for {species}")
        except Exception as exc:
            errors.append(f"CoolProp: {exc}")

    if backend_normalized not in {"auto", "thermo", "coolprop"}:
        raise ValueError("property-backend must be auto, thermo, or coolprop.")

    raise RuntimeError(
        f"Could not calculate molar Cp for species '{species}' at "
        f"T={temperature:.6g} K and P={pressure:.6g} Pa. "
        "Install advanced dependencies with: pip install -r requirements-advanced.txt. "
        f"Backend errors: {' | '.join(errors)}"
    )


def species_heat_capacity_flow(
    conversion: float,
    temperature: float,
    pressure: float,
    feed: Feed,
    stoich: Stoichiometry,
    species_a: str,
    species_b: str,
    species_p: str,
    species_i: str | None,
    backend: str,
) -> float:
    """Return sum(F_i Cp_i) using real species Cp(T,P), J/(time K)."""
    fa, fb, fp, fi = molar_flows(conversion, feed, stoich)
    cp_a = species_molar_cp(species_a, temperature, pressure, backend)
    cp_b = species_molar_cp(species_b, temperature, pressure, backend)
    cp_p = species_molar_cp(species_p, temperature, pressure, backend)
    cp_flow = fa * cp_a + fb * cp_b + fp * cp_p
    if fi > EPS:
        if not species_i:
            raise ValueError("species-i is required when fi0 is greater than zero.")
        cp_i = species_molar_cp(species_i, temperature, pressure, backend)
        cp_flow += fi * cp_i
    return max(EPS, cp_flow)


def solve_species_nonisothermal_pfr(
    volume: float,
    kinetics: Kinetics,
    feed: Feed,
    stoich: Stoichiometry,
    delta_h_rxn: float,
    ua: float,
    t_a: float,
    pressure: float,
    species_a: str,
    species_b: str,
    species_p: str,
    species_i: str | None,
    backend: str,
    points: int = 250,
    ideal_gas_temperature_correction: bool = False,
) -> ReactorResult:
    """Non-isothermal PFR with Cp(T,P) from thermo/CoolProp species data."""
    if volume <= 0.0:
        raise ValueError("PFR volume must be greater than zero.")
    if pressure <= 0.0:
        raise ValueError("Pressure must be greater than zero Pa.")

    # Check species names early so input mistakes fail before ODE integration.
    for name, species in (("species-a", species_a), ("species-b", species_b), ("species-p", species_p)):
        if not species:
            raise ValueError(f"{name} is required for species-nonisothermal-pfr mode.")
        species_molar_cp(species, feed.t0, pressure, backend)
    if feed.fi0 > EPS and species_i:
        species_molar_cp(species_i, feed.t0, pressure, backend)

    max_x = limiting_conversion(feed, stoich)
    terminal_x = max_x * (1.0 - 1.0e-8)

    def ode(v_reactor: float, y: Iterable[float]) -> list[float]:
        del v_reactor
        x = float(np.clip(y[0], 0.0, terminal_x))
        temperature = max(EPS, float(y[1]))
        rate = reaction_rate(
            x,
            temperature,
            kinetics,
            feed,
            stoich,
            ideal_gas_temperature_correction=ideal_gas_temperature_correction,
        )
        dx_dv = rate / feed.fa0
        heat_generation = rate * (-delta_h_rxn)
        heat_exchange = ua * (t_a - temperature)
        cp_flow = species_heat_capacity_flow(
            x,
            temperature,
            pressure,
            feed,
            stoich,
            species_a,
            species_b,
            species_p,
            species_i,
            backend,
        )
        dt_dv = (heat_exchange + heat_generation) / cp_flow
        return [dx_dv, dt_dv]

    def exhausted_event(v_reactor: float, y: Iterable[float]) -> float:
        del v_reactor
        return terminal_x - float(y[0])

    exhausted_event.terminal = True  # type: ignore[attr-defined]
    exhausted_event.direction = -1  # type: ignore[attr-defined]

    t_eval = np.linspace(0.0, volume, points)
    solution = solve_ivp(
        ode,
        (0.0, volume),
        [0.0, feed.t0],
        t_eval=t_eval,
        events=exhausted_event,
        max_step=max(volume / points, EPS),
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    volumes = solution.t
    conversions = np.clip(solution.y[0], 0.0, max_x)
    temperatures = solution.y[1]
    final_x = float(conversions[-1])
    final_t = float(temperatures[-1])
    return ReactorResult(
        mode="species-nonisothermal-pfr",
        volume=float(volumes[-1]),
        conversion=final_x,
        exit_temperature=final_t,
        selectivity=single_reaction_selectivity(final_x, feed, stoich),
        volumes=volumes,
        conversions=conversions,
        temperatures=temperatures,
    )


def format_property_value(value: object, units: str = "") -> str | None:
    """Format a property value, returning None for unavailable values."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return None
    if units:
        return f"{number:.6g} {units}"
    return f"{number:.6g}"


def require_advanced_backend(package_name: str, install_name: str) -> None:
    """Raise a helpful message when an optional advanced backend is unavailable."""
    raise RuntimeError(
        f"The optional package '{package_name}' is required for this backend. "
        f"Install advanced dependencies with: pip install -r requirements-advanced.txt "
        f"or install it directly with: pip install {install_name}"
    )


def thermo_species_properties(species: str, temperature: float, pressure: float) -> SpeciesPropertyResult:
    """Return pure-species properties using the thermo library."""
    try:
        from thermo import Chemical
    except ImportError:
        require_advanced_backend("thermo", "thermo")

    chemical = Chemical(species, T=temperature, P=pressure)
    properties: dict[str, str] = {}

    candidates = [
        ("Formula", getattr(chemical, "formula", None), ""),
        ("Molecular weight", getattr(chemical, "MW", None), "g/mol"),
        ("Phase", getattr(chemical, "phase", None), ""),
        ("Density", getattr(chemical, "rho", None), "kg/m^3"),
        ("Gas heat capacity", getattr(chemical, "Cpg", None), "J/(mol K)"),
        ("Liquid heat capacity", getattr(chemical, "Cpl", None), "J/(mol K)"),
        ("Enthalpy", getattr(chemical, "H", None), "J/mol"),
        ("Entropy", getattr(chemical, "S", None), "J/(mol K)"),
        ("Vapor pressure", getattr(chemical, "Psat", None), "Pa"),
        ("Heat of vaporization", getattr(chemical, "Hvap", None), "J/mol"),
        ("Melting point", getattr(chemical, "Tm", None), "K"),
        ("Boiling point", getattr(chemical, "Tb", None), "K"),
        ("Critical temperature", getattr(chemical, "Tc", None), "K"),
        ("Critical pressure", getattr(chemical, "Pc", None), "Pa"),
        ("Acentric factor", getattr(chemical, "omega", None), ""),
        ("Liquid viscosity", getattr(chemical, "mul", None), "Pa s"),
        ("Gas viscosity", getattr(chemical, "mug", None), "Pa s"),
        ("Liquid thermal conductivity", getattr(chemical, "kl", None), "W/(m K)"),
        ("Gas thermal conductivity", getattr(chemical, "kg", None), "W/(m K)"),
    ]
    for label, value, units in candidates:
        formatted = format_property_value(value, units)
        if formatted is not None:
            properties[label] = formatted

    return SpeciesPropertyResult(
        mode="species-properties",
        species=species,
        backend="thermo",
        temperature=temperature,
        pressure=pressure,
        properties=properties,
    )


def coolprop_species_properties(species: str, temperature: float, pressure: float) -> SpeciesPropertyResult:
    """Return pure-species properties using CoolProp."""
    try:
        from CoolProp.CoolProp import PhaseSI, PropsSI
    except ImportError:
        require_advanced_backend("CoolProp", "CoolProp")

    def props(output: str, units: str = "") -> str | None:
        try:
            value = PropsSI(output, "T", temperature, "P", pressure, species)
        except Exception:
            return None
        return format_property_value(value, units)

    def constant_props(output: str, units: str = "") -> str | None:
        try:
            value = PropsSI(output, species)
        except Exception:
            return None
        return format_property_value(value, units)

    properties: dict[str, str] = {}
    try:
        properties["Phase"] = PhaseSI("T", temperature, "P", pressure, species)
    except Exception:
        pass

    candidates = [
        ("Molecular weight", constant_props("MOLARMASS", "kg/mol")),
        ("Density", props("Dmass", "kg/m^3")),
        ("Molar density", props("Dmolar", "mol/m^3")),
        ("Mass heat capacity Cp", props("Cpmass", "J/(kg K)")),
        ("Molar heat capacity Cp", props("Cpmolar", "J/(mol K)")),
        ("Mass heat capacity Cv", props("Cvmass", "J/(kg K)")),
        ("Specific enthalpy", props("Hmass", "J/kg")),
        ("Molar enthalpy", props("Hmolar", "J/mol")),
        ("Specific entropy", props("Smass", "J/(kg K)")),
        ("Molar entropy", props("Smolar", "J/(mol K)")),
        ("Viscosity", props("VISCOSITY", "Pa s")),
        ("Thermal conductivity", props("CONDUCTIVITY", "W/(m K)")),
        ("Compressibility factor", props("Z", "")),
        ("Critical temperature", constant_props("Tcrit", "K")),
        ("Critical pressure", constant_props("Pcrit", "Pa")),
        ("Triple point temperature", constant_props("Ttriple", "K")),
    ]
    for label, formatted in candidates:
        if formatted is not None:
            properties[label] = formatted

    return SpeciesPropertyResult(
        mode="species-properties",
        species=species,
        backend="CoolProp",
        temperature=temperature,
        pressure=pressure,
        properties=properties,
    )


def solve_species_properties(
    species: str,
    temperature: float,
    pressure: float,
    backend: str,
) -> SpeciesPropertyResult:
    """Solve a pure-species property lookup with thermo or CoolProp."""
    if temperature <= 0.0:
        raise ValueError("Temperature must be greater than 0 K.")
    if pressure <= 0.0:
        raise ValueError("Pressure must be greater than 0 Pa.")

    backend_normalized = backend.lower()
    if backend_normalized == "thermo":
        return thermo_species_properties(species, temperature, pressure)
    if backend_normalized == "coolprop":
        return coolprop_species_properties(species, temperature, pressure)
    if backend_normalized != "auto":
        raise ValueError("property-backend must be auto, thermo, or coolprop.")

    errors: list[str] = []
    for solver in (thermo_species_properties, coolprop_species_properties):
        try:
            return solver(species, temperature, pressure)
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError(
        "No advanced property backend could evaluate the species. "
        "Install advanced dependencies with: pip install -r requirements-advanced.txt. "
        f"Backend errors: {' | '.join(errors)}"
    )


def format_selectivity(value: float) -> str:
    if np.isinf(value):
        return "infinite (single desired reaction; no undesired product modeled)"
    return f"{value:.6g}"


def print_result(result: ReactorResult) -> None:
    """Print final reactor values to the console."""
    print("\n=== Reactor Design Result ===")
    print(f"Mode: {result.mode}")
    print(f"Final reactor volume: {result.volume:.6g}")
    print(f"Final conversion X: {result.conversion:.6f}")
    print(f"Exit temperature: {result.exit_temperature:.3f} K")
    print(f"Selectivity P/U: {format_selectivity(result.selectivity)}")


def print_series_result(result: SeriesReactionResult) -> None:
    """Print final values for a first-order series reaction problem."""
    print("\n=== Series Reaction Result ===")
    print(f"Mode: {result.mode}")
    print("Reaction: A -> B -> C")
    if result.mode == "series-batch":
        print("CA(t) = CA0 exp(-k1 t)")
        print("CB(t) = CA0 k1/(k2-k1) [exp(-k1 t) - exp(-k2 t)]")
        print("CC(t) = CA0 - CA(t) - CB(t)")
    elif result.mode == "series-cstr":
        print("CA(tau) = CA0/(1 + k1 tau)")
        print("CB(tau) = k1 tau CA(tau)/(1 + k2 tau)")
        print("CC(tau) = CA0 - CA(tau) - CB(tau)")
    print(f"Optimum {result.independent_variable_name} for maximum B: {result.optimum_value:.6g} h")
    print(f"CA at optimum: {result.ca_at_optimum:.6g} mol/L")
    print(f"CB,max: {result.cb_at_optimum:.6g} mol/L")
    print(f"CC at optimum: {result.cc_at_optimum:.6g} mol/L")
    print(f"Overall selectivity B/C: {format_selectivity(result.selectivity_b_to_c)}")
    print(f"Overall yield of B: {result.yield_b:.6g}")


def print_power_law_fit_result(result: PowerLawFitResult) -> None:
    """Print a detailed two-reactant power-law fit."""
    print("\n=== Power-Law Kinetic Fit ===")
    print(f"Mode: {result.mode}")
    print(
        f"Model: r'_{result.product_name} = k "
        f"{result.reactant_1_name}^alpha {result.reactant_2_name}^beta"
    )
    print(f"alpha = {result.alpha:.6g}")
    print(f"beta = {result.beta:.6g}")
    print(
        f"k = {result.k:.6g} {result.rate_units}/"
        f"({result.independent_units}^(alpha+beta))"
    )
    print(f"R^2 on ln(rate) = {result.r_squared:.6f}")
    print("\nObserved vs predicted rates:")
    for index, (observed, predicted) in enumerate(
        zip(result.observed_rates, result.predicted_rates),
        start=1,
    ):
        print(f"  run {index}: observed={observed:.6g}, predicted={predicted:.6g}")


def print_species_property_result(result: SpeciesPropertyResult) -> None:
    """Print pure-species properties from an advanced backend."""
    print("\n=== Species Property Lookup ===")
    print(f"Mode: {result.mode}")
    print(f"Backend: {result.backend}")
    print(f"Species: {result.species}")
    print(f"Temperature: {result.temperature:.6g} K")
    print(f"Pressure: {result.pressure:.6g} Pa")
    print("\nProperties:")
    if not result.properties:
        print("  No properties returned by backend for this state/species.")
        return
    for label, value in result.properties.items():
        print(f"  {label}: {value}")


def print_parallel_cstr_result(result: ParallelCstrResult) -> None:
    """Print detailed parallel-reaction CSTR optimization results."""
    print("\n=== Parallel-Reaction CSTR Optimization ===")
    print("Reactions: A + B -> D, A + B -> U1, A + B -> U2")
    print("Instantaneous selectivity:")
    print("S_D/(U1+U2) = rD/(rU1 + rU2)")
    print("For the default photo problem:")
    print("S = kD CA CB / (kU1 CA^0.5 CB + kU2 CA^2 CB)")
    print(f"Exit CB used for rate evaluation: {result.cb_exit:.6g} mol/L")
    print(f"CA that maximizes instantaneous selectivity: {result.ca_optimum:.6g} mol/L")
    print(f"rD at optimum: {result.rate_d:.6g} mol/(L min)")
    print(f"rU1 at optimum: {result.rate_u1:.6g} mol/(L min)")
    print(f"rU2 at optimum: {result.rate_u2:.6g} mol/(L min)")
    print(f"Selectivity at optimum: {result.selectivity:.6g}")
    print(f"CSTR residence time from A balance: {result.residence_time:.6g} min")
    print(f"Recommended volumetric flow rate: {result.volumetric_flow_rate:.6g} L/min")
    print(f"Previous net value: ${result.previous_profit_per_liter:.6g}/L")
    print(f"Optimized net value: ${result.optimized_profit_per_liter:.6g}/L")
    print(f"Increase: ${result.extra_profit_per_liter:.6g}/L")
    print(
        "Increase at optimized flow, compared at the same flow rate: "
        f"${result.extra_profit_per_time:.6g}/min"
    )


def plot_profiles(
    result: ReactorResult,
    show: bool = True,
    save_dir: Path | None = None,
) -> None:
    """Plot conversion and, when available, temperature profiles."""
    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = Path(__file__).with_name(".matplotlib_cache")
        mpl_cache.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    if result.temperatures is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(result.volumes, result.conversions, color="#005f73", linewidth=2.4)
        ax.scatter([result.volume], [result.conversion], color="#ca6702", zorder=3)
        ax.set_title("Conversion vs. Reactor Volume")
        ax.set_xlabel("Reactor volume")
        ax.set_ylabel("Conversion, X")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        figures = [("conversion_vs_volume.png", fig)]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        axes[0].plot(
            result.volumes,
            result.conversions,
            color="#005f73",
            linewidth=2.4,
        )
        axes[0].set_title("Conversion vs. Reactor Volume")
        axes[0].set_ylabel("Conversion, X")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            result.volumes,
            result.temperatures,
            color="#ae2012",
            linewidth=2.4,
        )
        axes[1].set_title("Temperature Profile")
        axes[1].set_xlabel("Reactor volume")
        axes[1].set_ylabel("Temperature, K")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        figures = [("nonisothermal_profiles.png", fig)]

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for filename, fig in figures:
            output_path = save_dir / filename
            fig.savefig(output_path, dpi=180)
            print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close("all")


def plot_series_profiles(
    result: SeriesReactionResult,
    show: bool = True,
    save_dir: Path | None = None,
) -> None:
    """Plot CA, CB, and CC profiles for series reactions."""
    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = Path(__file__).with_name(".matplotlib_cache")
        mpl_cache.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.independent_values, result.ca, label="CA", linewidth=2.4)
    ax.plot(result.independent_values, result.cb, label="CB", linewidth=2.4)
    ax.plot(result.independent_values, result.cc, label="CC", linewidth=2.4)
    ax.scatter(
        [result.optimum_value],
        [result.cb_at_optimum],
        color="#ca6702",
        zorder=3,
        label="max CB",
    )
    ax.set_title(f"{result.mode}: A -> B -> C")
    ax.set_xlabel(f"{result.independent_variable_name.title()} (h)")
    ax.set_ylabel("Concentration (mol/L)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / f"{result.mode}_profiles.png"
        fig.savefig(output_path, dpi=180)
        print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close("all")


def plot_parallel_cstr_profiles(
    result: ParallelCstrResult,
    show: bool = True,
    save_dir: Path | None = None,
) -> None:
    """Plot instantaneous selectivity versus exit CA."""
    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = Path(__file__).with_name(".matplotlib_cache")
        mpl_cache.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.ca_values, result.selectivity_values, linewidth=2.4)
    ax.scatter(
        [result.ca_optimum],
        [result.selectivity],
        color="#ca6702",
        zorder=3,
        label="max selectivity",
    )
    ax.set_title("Parallel CSTR Selectivity")
    ax.set_xlabel("Exit CA (mol/L)")
    ax.set_ylabel("Instantaneous selectivity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / "parallel_cstr_selectivity.png"
        fig.savefig(output_path, dpi=180)
        print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close("all")


def plot_power_law_fit(
    result: PowerLawFitResult,
    show: bool = True,
    save_dir: Path | None = None,
) -> None:
    """Plot observed versus predicted rates for a power-law fit."""
    if "MPLCONFIGDIR" not in os.environ:
        mpl_cache = Path(__file__).with_name(".matplotlib_cache")
        mpl_cache.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_cache)

    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(result.observed_rates, result.predicted_rates, color="#005f73", s=64)
    low = min(float(np.min(result.observed_rates)), float(np.min(result.predicted_rates)))
    high = max(float(np.max(result.observed_rates)), float(np.max(result.predicted_rates)))
    ax.plot([low, high], [low, high], color="#ca6702", linewidth=2.0, label="perfect fit")
    ax.set_title("Power-Law Fit: Observed vs Predicted")
    ax.set_xlabel(f"Observed rate ({result.rate_units})")
    ax.set_ylabel(f"Predicted rate ({result.rate_units})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / "power_law_fit_parity.png"
        fig.savefig(output_path, dpi=180)
        print(f"Saved plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close("all")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve CSTR and PFR reactor design problems."
    )
    parser.add_argument(
        "--mode",
        choices=(
            "cstr",
            "pfr",
            "nonisothermal-pfr",
            "series-batch",
            "series-cstr",
            "kinetic-fit-power-law",
            "parallel-cstr",
            "species-properties",
            "species-nonisothermal-pfr",
        ),
        default="nonisothermal-pfr",
        help="Reactor model to solve.",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=100.0,
        help="Reactor volume for PFR, or known CSTR volume.",
    )
    parser.add_argument(
        "--target-x",
        type=float,
        default=None,
        help="For CSTR mode, solve required volume for this target conversion.",
    )
    parser.add_argument("--temperature", type=float, default=350.0, help="Isothermal reactor temperature, K.")
    parser.add_argument("--fa0", type=float, default=10.0, help="Inlet molar flow of A.")
    parser.add_argument("--fb0", type=float, default=10.0, help="Inlet molar flow of B.")
    parser.add_argument("--fi0", type=float, default=0.0, help="Inlet molar flow of inert I.")
    parser.add_argument("--v0", type=float, default=100.0, help="Inlet volumetric flow.")
    parser.add_argument("--t0", type=float, default=350.0, help="Inlet temperature, K.")
    parser.add_argument("--arrhenius-a", type=float, default=DEFAULT_A, help="Arrhenius pre-exponential factor.")
    parser.add_argument("--ea", type=float, default=DEFAULT_EA, help="Activation energy, J/mol.")
    parser.add_argument("--b-per-a", type=float, default=1.0, help="Stoichiometric B consumed per A.")
    parser.add_argument("--product-per-a", type=float, default=1.0, help="Desired product formed per A.")
    parser.add_argument("--gas-phase", action="store_true", help="Enable gas-phase expansion.")
    parser.add_argument("--y-a0", type=float, default=1.0, help="Inlet mole fraction of A for epsilon.")
    parser.add_argument("--delta", type=float, default=0.0, help="Gas expansion delta; epsilon = y_A0 * delta.")
    parser.add_argument(
        "--ideal-gas-temperature-correction",
        action="store_true",
        help="For non-isothermal gas work, multiply v by T/T0 in addition to 1 + epsilon X.",
    )
    parser.add_argument("--delta-h", type=float, default=-20_000.0, help="Heat of reaction, J/mol A.")
    parser.add_argument("--ua", type=float, default=250.0, help="Ua heat-transfer term per reactor volume.")
    parser.add_argument("--ta", type=float, default=350.0, help="Coolant/ambient temperature, K.")
    parser.add_argument("--cp-a", type=float, default=100.0, help="Heat capacity of A.")
    parser.add_argument("--cp-b", type=float, default=100.0, help="Heat capacity of B.")
    parser.add_argument("--cp-p", type=float, default=100.0, help="Heat capacity of product P.")
    parser.add_argument("--cp-i", type=float, default=100.0, help="Heat capacity of inert I.")
    parser.add_argument("--series-k1", type=float, default=0.5, help="First rate constant for A -> B, 1/h.")
    parser.add_argument("--series-k2", type=float, default=0.2, help="Second rate constant for B -> C, 1/h.")
    parser.add_argument("--ca0", type=float, default=2.0, help="Initial/feed concentration of A for series reactions, mol/L.")
    parser.add_argument("--time-start", type=float, default=0.0, help="Start time/residence time for series-reaction profiles, h.")
    parser.add_argument("--time-end", type=float, default=12.0, help="End time/residence time for series-reaction profiles, h.")
    parser.add_argument(
        "--fit-data-csv",
        type=Path,
        default=None,
        help="CSV or XLSX for power-law fitting. Use x1,x2,rate or total_flow,x1,x2,y_product columns.",
    )
    parser.add_argument(
        "--catalyst-weight",
        type=float,
        default=40.0,
        help="Catalyst mass for product-flow kinetic fits, g.",
    )
    parser.add_argument(
        "--species",
        default="water",
        help="Pure species name for thermo/CoolProp property lookup, e.g. water, methane, ethanol.",
    )
    parser.add_argument("--species-a", default="ethanol", help="Species name for reactant A in species-nonisothermal-pfr mode.")
    parser.add_argument("--species-b", default="oxygen", help="Species name for reactant B in species-nonisothermal-pfr mode.")
    parser.add_argument("--species-p", default="acetaldehyde", help="Species name for product P in species-nonisothermal-pfr mode.")
    parser.add_argument("--species-i", default=None, help="Optional inert species name when fi0 is nonzero.")
    parser.add_argument(
        "--pressure",
        type=float,
        default=101_325.0,
        help="Pressure for species property lookup, Pa.",
    )
    parser.add_argument(
        "--property-backend",
        choices=("auto", "thermo", "coolprop"),
        default="auto",
        help="Advanced property backend for species-properties mode.",
    )
    parser.add_argument("--parallel-ca0", type=float, default=2.0, help="Feed concentration of A for parallel CSTR, mol/L.")
    parser.add_argument("--parallel-cb0", type=float, default=1.5, help="Feed concentration of B for parallel CSTR, mol/L.")
    parser.add_argument("--parallel-exit-cb", type=float, default=0.5, help="Exit concentration of B for parallel CSTR, mol/L.")
    parser.add_argument("--parallel-volume", type=float, default=250.0, help="Parallel CSTR volume, L.")
    parser.add_argument("--parallel-kd", type=float, default=0.3, help="Rate constant for desired product D.")
    parser.add_argument("--parallel-ku1", type=float, default=0.1, help="Rate constant for undesired product U1.")
    parser.add_argument("--parallel-ku2", type=float, default=0.2, help="Rate constant for undesired product U2.")
    parser.add_argument("--d-exp-a", type=float, default=1.0, help="A exponent in rD.")
    parser.add_argument("--d-exp-b", type=float, default=1.0, help="B exponent in rD.")
    parser.add_argument("--u1-exp-a", type=float, default=0.5, help="A exponent in rU1.")
    parser.add_argument("--u1-exp-b", type=float, default=1.0, help="B exponent in rU1.")
    parser.add_argument("--u2-exp-a", type=float, default=2.0, help="A exponent in rU2.")
    parser.add_argument("--u2-exp-b", type=float, default=1.0, help="B exponent in rU2.")
    parser.add_argument("--previous-d", type=float, default=0.98, help="Previous outlet D concentration, mol/L.")
    parser.add_argument("--previous-u1", type=float, default=0.5, help="Previous outlet U1 concentration, mol/L.")
    parser.add_argument("--previous-u2", type=float, default=0.75, help="Previous outlet U2 concentration, mol/L.")
    parser.add_argument("--optimized-d", type=float, default=1.2, help="Optimized outlet D concentration, mol/L.")
    parser.add_argument("--optimized-u1", type=float, default=0.275, help="Optimized outlet U1 concentration, mol/L.")
    parser.add_argument("--optimized-u2", type=float, default=0.6, help="Optimized outlet U2 concentration, mol/L.")
    parser.add_argument("--desired-price", type=float, default=50.0, help="Sale price for D, dollars/mol.")
    parser.add_argument("--undesired-cost", type=float, default=10.0, help="Disposal cost for U1 and U2, dollars/mol.")
    parser.add_argument("--points", type=int, default=250, help="Number of plotting/integration points.")
    parser.add_argument("--save-plots", type=Path, default=None, help="Directory for plot image output.")
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive Matplotlib window.")
    return parser


def run_from_args(
    args: argparse.Namespace,
) -> ReactorResult | SeriesReactionResult | PowerLawFitResult | ParallelCstrResult | SpeciesPropertyResult:
    if args.mode == "series-batch":
        return solve_series_batch(
            k1=args.series_k1,
            k2=args.series_k2,
            ca0=args.ca0,
            time_start=args.time_start,
            time_end=args.time_end,
            points=args.points,
        )
    if args.mode == "series-cstr":
        return solve_series_cstr(
            k1=args.series_k1,
            k2=args.series_k2,
            ca0=args.ca0,
            tau_start=args.time_start,
            tau_end=args.time_end,
            points=args.points,
        )
    if args.mode == "kinetic-fit-power-law":
        return solve_kinetic_fit_power_law(
            csv_path=args.fit_data_csv,
            catalyst_weight_g=args.catalyst_weight,
        )
    if args.mode == "species-properties":
        return solve_species_properties(
            species=args.species,
            temperature=args.temperature,
            pressure=args.pressure,
            backend=args.property_backend,
        )
    if args.mode == "species-nonisothermal-pfr":
        kinetics = Kinetics(
            pre_exponential=args.arrhenius_a,
            activation_energy=args.ea,
        )
        feed = Feed(
            fa0=args.fa0,
            fb0=args.fb0,
            fi0=args.fi0,
            v0=args.v0,
            t0=args.t0,
        )
        stoich = Stoichiometry(
            b_per_a=args.b_per_a,
            product_per_a=args.product_per_a,
            gas_phase=args.gas_phase,
            y_a0=args.y_a0,
            delta=args.delta,
        )
        return solve_species_nonisothermal_pfr(
            volume=args.volume,
            kinetics=kinetics,
            feed=feed,
            stoich=stoich,
            delta_h_rxn=args.delta_h,
            ua=args.ua,
            t_a=args.ta,
            pressure=args.pressure,
            species_a=args.species_a,
            species_b=args.species_b,
            species_p=args.species_p,
            species_i=args.species_i,
            backend=args.property_backend,
            points=args.points,
            ideal_gas_temperature_correction=args.ideal_gas_temperature_correction,
        )
    if args.mode == "parallel-cstr":
        return solve_parallel_cstr(
            ca0=args.parallel_ca0,
            cb0=args.parallel_cb0,
            cb_exit=args.parallel_exit_cb,
            volume=args.parallel_volume,
            k_d=args.parallel_kd,
            k_u1=args.parallel_ku1,
            k_u2=args.parallel_ku2,
            d_exp_a=args.d_exp_a,
            d_exp_b=args.d_exp_b,
            u1_exp_a=args.u1_exp_a,
            u1_exp_b=args.u1_exp_b,
            u2_exp_a=args.u2_exp_a,
            u2_exp_b=args.u2_exp_b,
            previous_d=args.previous_d,
            previous_u1=args.previous_u1,
            previous_u2=args.previous_u2,
            optimized_d=args.optimized_d,
            optimized_u1=args.optimized_u1,
            optimized_u2=args.optimized_u2,
            desired_price=args.desired_price,
            undesired_cost=args.undesired_cost,
            points=args.points,
        )

    kinetics = Kinetics(
        pre_exponential=args.arrhenius_a,
        activation_energy=args.ea,
    )
    feed = Feed(
        fa0=args.fa0,
        fb0=args.fb0,
        fi0=args.fi0,
        v0=args.v0,
        t0=args.t0,
    )
    stoich = Stoichiometry(
        b_per_a=args.b_per_a,
        product_per_a=args.product_per_a,
        gas_phase=args.gas_phase,
        y_a0=args.y_a0,
        delta=args.delta,
    )
    heat = HeatTransfer(
        delta_h_rxn=args.delta_h,
        ua=args.ua,
        t_a=args.ta,
        cp_a=args.cp_a,
        cp_b=args.cp_b,
        cp_p=args.cp_p,
        cp_i=args.cp_i,
    )

    if args.mode == "cstr":
        return cstr_result(
            volume=args.volume if args.target_x is None else None,
            target_conversion=args.target_x,
            temperature=args.temperature,
            kinetics=kinetics,
            feed=feed,
            stoich=stoich,
            points=args.points,
        )
    if args.mode == "pfr":
        return solve_pfr(
            volume=args.volume,
            temperature=args.temperature,
            kinetics=kinetics,
            feed=feed,
            stoich=stoich,
            points=args.points,
        )
    return solve_nonisothermal_pfr(
        volume=args.volume,
        kinetics=kinetics,
        feed=feed,
        stoich=stoich,
        heat=heat,
        points=args.points,
        ideal_gas_temperature_correction=args.ideal_gas_temperature_correction,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_from_args(args)
    if isinstance(result, SeriesReactionResult):
        print_series_result(result)
    elif isinstance(result, PowerLawFitResult):
        print_power_law_fit_result(result)
    elif isinstance(result, ParallelCstrResult):
        print_parallel_cstr_result(result)
    elif isinstance(result, SpeciesPropertyResult):
        print_species_property_result(result)
    else:
        print_result(result)
    if args.no_show and args.save_plots is None:
        return
    if isinstance(result, SeriesReactionResult):
        plot_series_profiles(result, show=not args.no_show, save_dir=args.save_plots)
    elif isinstance(result, ParallelCstrResult):
        plot_parallel_cstr_profiles(
            result,
            show=not args.no_show,
            save_dir=args.save_plots,
        )
    elif isinstance(result, PowerLawFitResult):
        plot_power_law_fit(result, show=not args.no_show, save_dir=args.save_plots)
    elif isinstance(result, SpeciesPropertyResult):
        return
    else:
        plot_profiles(result, show=not args.no_show, save_dir=args.save_plots)


if __name__ == "__main__":
    main()
