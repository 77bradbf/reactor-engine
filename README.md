# Reactor Engineering Engine

Python calculator for chemical reaction engineering homework and study problems.

It can solve:

- single-reaction CSTR design
- single-reaction PFR design
- non-isothermal PFR material and energy balances
- first-order series reactions `A -> B -> C` in batch reactors
- first-order series reactions `A -> B -> C` in CSTRs
- power-law kinetic fitting from CSV or Excel data
- parallel-reaction CSTR selectivity and profit optimization
- pure-species property lookup using `thermo` or `CoolProp`

## Requirements

Install the base Python packages:

```bash
pip install -r requirements.txt
```

For more accurate/advanced work with real spreadsheet files, units, property
libraries, uncertainty reports, and detailed reactor networks, install:

```bash
pip install -r requirements-advanced.txt
```

The advanced packages add:

- `pandas` and `openpyxl` for stronger CSV/Excel reading
- `pint` for future unit-aware calculations
- `lmfit` for future nonlinear fitting with parameter uncertainty
- `thermo` and `CoolProp` for real chemical/thermophysical properties
- `cantera` for detailed thermodynamics, kinetics, transport, and reactor networks

Then run commands from the project folder. For example, after cloning:

```bash
git clone https://github.com/77bradbf/reactor-engine.git
cd reactor-engine
```

## Quick Start

Show all available options:

```bash
python3 reactor_engine.py --help
```

Run the default non-isothermal PFR example:

```bash
python3 reactor_engine.py
```

Run without showing plots:

```bash
python3 reactor_engine.py --no-show
```

Save plots instead of opening them:

```bash
python3 reactor_engine.py --save-plots plots --no-show
```

## Modes

### 1. CSTR

Find required CSTR volume for a target conversion:

```bash
python3 reactor_engine.py --mode cstr --target-x 0.5
```

Find conversion for a known CSTR volume:

```bash
python3 reactor_engine.py --mode cstr --volume 100
```

### 2. PFR

Find conversion for an isothermal PFR:

```bash
python3 reactor_engine.py --mode pfr --volume 100 --temperature 350
```

### 3. Non-Isothermal PFR

Solve conversion and temperature profiles:

```bash
python3 reactor_engine.py --mode nonisothermal-pfr --volume 100 --delta-h -20000 --ua 250 --ta 350
```

### 4. Series Batch Reactor

For `A -> B -> C`:

```bash
python3 reactor_engine.py --mode series-batch --series-k1 0.5 --series-k2 0.2 --ca0 2 --time-start 0 --time-end 12
```

The code prints:

- expressions for `CA(t)`, `CB(t)`, and `CC(t)`
- time for maximum `CB`
- `CA`, `CB`, and `CC` at that time
- overall selectivity `B/C`
- overall yield of `B`

### 5. Series CSTR

For `A -> B -> C` in a CSTR:

```bash
python3 reactor_engine.py --mode series-cstr --series-k1 0.5 --series-k2 0.2 --ca0 2 --time-start 0 --time-end 12
```

The code prints:

- expressions for `CA(tau)`, `CB(tau)`, and `CC(tau)`
- residence time that maximizes `CB`
- `CA`, `CB`, and `CC` at that residence time
- overall selectivity and yield

### 6. Power-Law Kinetic Fit

Use this when a problem asks you to find:

- reaction order in reactant 1, `alpha`
- reaction order in reactant 2, `beta`
- rate constant, `k`

Model:

```text
rate = k x1^alpha x2^beta
```

Run the built-in example:

```bash
python3 reactor_engine.py --mode kinetic-fit-power-law
```

Run with your own CSV or Excel file:

```bash
python3 reactor_engine.py --mode kinetic-fit-power-law --fit-data-csv examples/my_new_data.xlsx
```

If your table gives total flow and product mole fraction, include catalyst weight:

```bash
python3 reactor_engine.py --mode kinetic-fit-power-law --fit-data-csv examples/my_new_data.xlsx --catalyst-weight 40
```

#### Data Format A: Rate Already Given

CSV or Excel headers:

```text
x1 | x2 | rate
```

Example:

```text
x1      x2      rate
0.20    0.30    0.0048
0.40    0.30    0.0096
0.20    0.60    0.0068
0.50    0.50    0.0141
0.80    0.40    0.0202
```

#### Data Format B: Product Mole Fraction Given

CSV or Excel headers:

```text
total_flow | x1 | x2 | y_product
```

Example:

```text
total_flow   x1     x2     y_product
0.61         0.50   0.51   0.07
0.60         1.00   0.54   0.16
0.30         0.40   0.60   0.16
0.72         0.60   0.60   0.10
0.52         0.60   0.40   0.06
```

The code calculates:

```text
rate = total_flow * y_product / catalyst_weight
```

### 7. Parallel CSTR Selectivity Optimization

For parallel reactions:

```text
A + B -> D
A + B -> U1
A + B -> U2
```

Run the built-in example:

```bash
python3 reactor_engine.py --mode parallel-cstr
```

Run with explicit values:

```bash
python3 reactor_engine.py --mode parallel-cstr \
  --parallel-ca0 2 \
  --parallel-cb0 1.5 \
  --parallel-exit-cb 0.5 \
  --parallel-volume 250 \
  --parallel-kd 0.3 \
  --parallel-ku1 0.1 \
  --parallel-ku2 0.2 \
  --d-exp-a 1 --d-exp-b 1 \
  --u1-exp-a 0.5 --u1-exp-b 1 \
  --u2-exp-a 2 --u2-exp-b 1
```

The code prints:

- instantaneous selectivity equation
- `CA` that maximizes selectivity
- rates of `D`, `U1`, and `U2`
- CSTR residence time
- recommended volumetric flow rate
- profit comparison

### 8. Species Properties with `thermo` or `CoolProp`

Use this when you want real pure-species property data instead of manually typing values like heat capacity or density.

Install advanced dependencies first:

```bash
pip install -r requirements-advanced.txt
```

Look up water properties at 350 K and 1 atm using whichever backend is available:

```bash
python3 reactor_engine.py --mode species-properties --species water --temperature 350 --pressure 101325 --property-backend auto --no-show
```

Force the `thermo` backend:

```bash
python3 reactor_engine.py --mode species-properties --species ethanol --temperature 330 --pressure 101325 --property-backend thermo --no-show
```

Force the `CoolProp` backend:

```bash
python3 reactor_engine.py --mode species-properties --species Water --temperature 350 --pressure 101325 --property-backend coolprop --no-show
```

Depending on the backend/species, the code can print properties such as:

- phase
- molecular weight
- density
- heat capacity
- enthalpy
- entropy
- viscosity
- thermal conductivity
- vapor pressure
- critical temperature and pressure

### 9. Species-Coupled Non-Isothermal PFR

Use this when you want the PFR energy balance to use real species heat capacities from `thermo` or `CoolProp` instead of manually entering `--cp-a`, `--cp-b`, and `--cp-p`.

Install advanced dependencies first:

```bash
pip install -r requirements-advanced.txt
```

Example problem:

```text
A liquid-phase oxidation-like reaction is modeled as A + B -> P.
Use ethanol as A, oxygen as B, and acetaldehyde as product P.
The PFR volume is 50 L, feed is FA0 = 2 mol/min, FB0 = 3 mol/min,
v0 = 50 L/min, T0 = 330 K, pressure = 1 atm, heat of reaction = -50000 J/mol,
Ua = 120, and coolant temperature is 310 K.
Calculate conversion and exit temperature using real Cp(T,P) from thermo.
```

Run:

```bash
python3 reactor_engine.py --mode species-nonisothermal-pfr --volume 50 --fa0 2 --fb0 3 --v0 50 --t0 330 --species-a ethanol --species-b oxygen --species-p acetaldehyde --delta-h -50000 --ua 120 --ta 310 --pressure 101325 --property-backend thermo --no-show
```

Same idea with an inert species:

```bash
python3 reactor_engine.py --mode species-nonisothermal-pfr --volume 50 --fa0 2 --fb0 3 --fi0 5 --v0 100 --t0 330 --species-a ethanol --species-b oxygen --species-p acetaldehyde --species-i nitrogen --delta-h -50000 --ua 120 --ta 310 --pressure 101325 --property-backend thermo --no-show
```

This mode still uses the built-in Arrhenius rate law, but it replaces constant manually typed heat capacities with backend-calculated species heat capacities along the reactor.

## Important Notes

- Use consistent units.
- For kinetic fitting, use at least 3 data rows because the code solves for `k`, `alpha`, and `beta`.
- More data rows usually give a better fit.
- Excel files should use the first sheet and have headers in the first row.
- For plots to show on screen, do not use `--no-show`.

## Project Files

- `reactor_engine.py`: main calculator
- `REACTOR_ENGINE_USAGE.md`: extra usage notes
- `examples/`: example input data files
- `plots/`: saved example plots
