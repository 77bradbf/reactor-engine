# Reactor Engine Usage

## 1. Gas-Phase Catalytic Kinetic Fit

Photo problem:

```text
H2 + C2H6 -> 2 CH4
r'_CH4 = k P_C2H6^alpha P_H2^beta
```

The table columns go into the code like this:

```text
total_flow  = total molar flow rate, mol/h
x1          = P_C2H6, atm
x2          = P_H2, atm
y_product   = mole fraction CH4 in exit stream
```

The catalyst weight is:

```text
4 baskets * 10 g/basket = 40 g catalyst
```

Run the built-in photo data:

```bash
python3 reactor_engine.py --mode kinetic-fit-power-law
```

Run the same problem from the example CSV:

```bash
python3 reactor_engine.py --mode kinetic-fit-power-law --fit-data-csv examples/kinetic_fit_photo.csv --catalyst-weight 40
```

For a future problem, make a CSV with either:

```text
total_flow,x1,x2,y_product
```

or, if the rates are already calculated:

```text
x1,x2,rate
```

Then run:

```bash
python3 reactor_engine.py --mode kinetic-fit-power-law --fit-data-csv your_file.csv --catalyst-weight 40
```

## 2. Parallel-Reaction CSTR Selectivity Optimization

Photo problem:

```text
A + B -> D
A + B -> U1
A + B -> U2
```

Default rate-law inputs in the code:

```text
rD  = 0.3 CA^1   CB^1
rU1 = 0.1 CA^0.5 CB^1
rU2 = 0.2 CA^2   CB^1
```

Default feed and reactor inputs:

```text
CA0 = 2.0 mol/L
CB0 = 1.5 mol/L
exit CB = 0.5 mol/L
V = 250 L
```

Default profit inputs:

```text
old outlet: D = 0.98, U1 = 0.5, U2 = 0.75 mol/L
new outlet: D = 1.2, U1 = 0.275, U2 = 0.6 mol/L
D sale price = $50/mol
U1 and U2 disposal cost = $10/mol
```

Run the built-in photo problem:

```bash
python3 reactor_engine.py --mode parallel-cstr
```

For a future problem, change the values in the command:

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

The most important values to identify from a problem statement are:

```text
rate constants: kD, kU1, kU2
reaction orders: exponents on CA and CB
feed concentrations: CA0, CB0
specified exit concentration, usually CB
reactor volume, if asking for flow rate
old/new outlet concentrations and prices, if asking for profit
```
