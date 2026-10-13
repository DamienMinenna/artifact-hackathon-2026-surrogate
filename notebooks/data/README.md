# Data — Surrogate Models (PALLAS)

## Description

Particle-in-cell simulations of the PALLAS laser-plasma accelerator, run with Smilei. Every row
is one simulation: the machine settings that were used, and the electron bunch that came out.
**Simulation only — there is no measured data here, and therefore no diagnostic uncertainty and
no shot-to-shot jitter.**

## Getting the pack

The pack is about 200 MB zipped and is **not stored in this repository**.

> **Download:** _TBD — link to be added by the organisers._

**If `pallas_hackathon_data/` is already sitting in this `data/` directory, it is in place and
there is nothing for you to download** — the line above is for the downloadable copy. A
`.gitignore` beside this file keeps those ~200 MB out of any repository this folder is copied
into.

Unpack it so that the folder `pallas_hackathon_data/` sits inside this `data/` directory:

```
challenges/surrogate-models/
├── kickoff_notebook.ipynb
├── kickoff_notebook_full.ipynb
├── pallas_score.py
└── data/
    └── pallas_hackathon_data/     <- unpack here
        ├── campaign_A.parquet
        ├── ...
        └── test_trajectory_ood.parquet
```

The kickoff notebook finds it there by itself. If you keep the pack somewhere else, set
`PALLAS_PACK` to its path before starting Jupyter.

### The sample in `data/sample/`

Until the full pack is unpacked, both notebooks fall back on `data/sample/`: a 10 % random
sample (22 MB) with the same tables and columns. Campaign A keeps 1,307 of 13,073 simulations
(the same share of each scan); Campaign B keeps 529 of 5,286 simulations (286 of them produced
a bunch), with their trajectories and moments. The six `test_*` input files and the energy grid
are copied whole (the test inputs carry no targets), and `pack_reference.json` is included. The sample is
for checking that the code runs. Its baselines and scores differ from the full pack's, and the
row counts in the table below are the full pack's.

## Files

| file | rows | what it is |
|---|---|---|
| `campaign_A.parquet` | 13,073 | Campaign A — four scalar knobs, both scans in one table; `scan` is `random` (9,285 simulations) or `uniform` (3,788 on a grid) |
| `campaign_B.parquet` | 5,286 | Campaign B — laser fixed, the gas density profile varied and shipped as a 2000-point curve |
| `campaign_B_trajectories.parquet` | 322,615 | the bunch along the accelerator: 2,855 configurations × 113 positions |
| `campaign_B_moments.parquet` | 211,360 | the bunch's full 6×6 covariance at 20 planes, twice: `group` = `all` (every electron; what the scorer compares against) and `cohort` (a sub-population) |
| `common_energy_grid_MeV.npy` | 200 bins | the shared spectrum axis, 25.8–510.3 MeV |
| `test_direct.parquet` | 357 | held-out inputs: settings → bunch |
| `test_inverse.parquet` | 1,915 | held-out inputs: bunch + focal position `x_of` → `p_1`, `a_0`, `c_N2` (Campaign A) |
| `test_inverse_B.parquet` | 357 | held-out inputs: bunch + focal position `x_of` → the four density-profile settings (Campaign B) |
| `test_trajectory.parquet` | 40,341 | held-out inputs: settings + position → bunch |
| `test_trajectory_ood.parquet` | 15,368 | the 136 configurations of `test_trajectory` above the density cut, on their own |
| `test_moments_planes.parquet` | 3,213 | held-out inputs: settings + plane → the bunch's moments |
| `pack_reference.json` | — | **one file for everything you need to be comparable**: three id lists in its `sections`, and every reference number in its `scores` |
| `DATA_CARD.md` | — | the pack's own detailed card |

Read `pack_reference.json` once and index it by name:

```python
ref = json.load(open(f"{PACK}/pack_reference.json"))
SPLITS, SCORES = ref["sections"], ref["scores"]
SPLITS["ood_split"]["train"]                # the 2,277 configs below the density cut
SCORES["direct_reference_emulator"]         # what our internal forward model reaches
```

Its `sections` are `reference_split`, `ood_split` and `plane_holdout`; each carries its own `what` line saying what it is
for. Every id list is a list of `config` values, except `plane_holdout`, which lists planes in mm.

The two campaigns **do not share a knob set**, which is why they are two files. Campaign A
varies four scalars (`p_1`, `a_0`, `c_N2`, `x_of`) and sampled them twice — once at random and
once on a grid — which is what the `scan` column distinguishes; Campaign B fixes the laser and
varies the shape of the gas density profile (`P_max`, `cN2_max`, `L_inj`, `dip_frac`, `x_of`).
The hidden inverse test set is drawn from the random scan only.

**No target column appears in any shipped test file.** This is checked on every build of the
pack. The hidden true values are held by the organisers.

## Level 5 — particle clouds (a separate download, `level5/`)

Not in the pack itself, because of its size: Campaign B's bunch as individual electrons, for the
Hard objective "Predict the bunch as particles". Up to 2,000 tracked electrons per simulation (a
few configurations with a small bunch have fewer), the same electrons followed along the accelerator, for the 5,282 Campaign B configurations that are in no
test file. One `Config_<id>.npz` per configuration, the same `config` as everywhere else.

| part | what | size |
|---|---|---|
| `planes20/` | the twenty planes of `campaign_B_moments.parquet`; start here | 5.3 GB |
| full set | every snapshot from the bunch's first (1.5–2.1 mm) to 7.04 mm, every 22.5 µm — 222 to 247 per configuration; same keys | 54.2 GB |

The keys that matter:

- `state` (snapshots × electrons × 6), float32, in the order `zeta_um, y_um, z_um, ux, uy, uz`:
  `zeta_um` is the position along the beam (µm), measured from a point moving at the speed of
  light, NOT from the laser's front as in the moment table: the two differ by a few tens of µm
  (about 35 µm at 2 mm, growing slowly along the accelerator), the same orientation; positions
  within the bunch are unaffected. To compare with the moment table, compare widths and shapes, or
  shift each plane by the difference of the two `zeta` means. `y_um` and
  `z_um` are transverse (µm — as in the moment table, `z` here is a TRANSVERSE coordinate), and
  `ux, uy, uz` are momenta in units of m_e c, `ux` along the beam.
- `present` (snapshots × electrons), bool: the electron exists at that snapshot. Electrons are lost
  along the way, and a late-injecting bunch has none at its first planes. Where `present` is
  false, `state` is NaN.
- `weight` (electrons): each electron's share of the bunch's charge. **The clouds are
  charge-weighted, not equal-weight** — use `weight` in every mean, width and loss, or you model
  the simulation's sampling rather than the beam.
- `plane_mm` (`planes20/` only) and `z_win_mm`: the position along the accelerator of each
  snapshot, in mm. The snapshot taken for a plane sits within 10 µm of it.
- `theta` with `theta_keys`: the settings, as a check against `campaign_B.parquet`.

The moments computed from these clouds are close to, but not identical to,
`campaign_B_moments.parquet`: the moment table was computed from every simulated electron, and
the clouds are a sample of up to 2,000 of them.

## Level 6 — the laser and plasma fields (a separate download, `level6/`)

Not in the pack itself, because of its size: the fields on the laser axis, for the optional
objective "Predict the wake itself". `level6/README_level6.md` and `level6/load_level6.py`
sit beside the maps. For each of the 5,286 configurations of
`campaign_B.parquet` (none of them a test configuration), three maps of 157 snapshots × 320
points, 1.6 GB in all, float16.

| file | what | unit |
|---|---|---|
| `field_ex.npy` | longitudinal electric field on axis | GV/m |
| `field_rho_e.npy` | plasma electron density on axis (electron charge, so ≤ 0) | critical density n_c |
| `field_laser.npy` | laser envelope amplitude | a₀ (normalised vector potential) |
| `field_axes.npz` | the axes below | |

Each `.npy` is configurations × snapshots × points, rows in the order of `config`; open it with
`np.load(path, mmap_mode="r")` to avoid reading 530 MB at once. `field_axes.npz` holds:

- `config`: the configuration of each row, the same id as everywhere else.
- `x_window_um` (320): the position inside the window, 0 to 63.9 µm, every 0.2 µm, counted from
  the window's back edge — the laser sits in the front half, the wake behind it.
- `x_moved_um` (configurations × 157): the absolute position of the window's back edge along the
  accelerator at each snapshot, 0 to 7,020 µm, every 45 µm. It is the same for every
  configuration.
- `x_a0_um` (configurations × 157): the absolute position of the laser's envelope peak.
- `ts` (configurations × 157): the simulation step of each snapshot.

The rounding to float16 changes no value by more than about 0.025 % of its field's peak.

## Units

Units are the thing people get wrong here, so each column carries one.

- **Energies are MeV everywhere.** Campaign B was stored natively in Lorentz gamma and is
  converted in the pack; the native columns are not shipped, so there is nothing to pick up by
  mistake and no unit flag to check.
- `scan` is `random` or `uniform` in `campaign_A.parquet`, and is absent from `campaign_B.parquet`.
- `dE_mad` and `dE_std` are **relative** spreads, dimensionless.
- `q_end` is in **coulombs** — a 3 pC cut is `3e-12`. `q_pC` beside it is in picocoulombs.
- `p_1` is in mbar; `P_max` is in **pascals**; `x_of` in µm; `L_inj` and `plane_mm` and `z_mm` in mm.
- In `campaign_B_moments.parquet` only, **`z` is a transverse coordinate**; the position along the
  accelerator is `plane_mm`.

## What was done to the raw output

1. **Energy units harmonised** — MeV everywhere, native columns dropped.
2. **Spectra resampled** onto one shared 200-bin grid. Each row's native axis is kept as
   `ener_axis_MeV`, and `spec` on it is charge per MeV (it integrates to `q_pC`). `spec_common`
   is the charge in pC in each shared bin, and it sums to `q_pC` on every injected row.
3. **One injection flag** — `injected` is `q_pC >= 3.0`, computed identically for both
   campaigns. The source files' own labels are not shipped; they were produced under different
   rules and were not comparable.
4. **273 unphysical runs deleted**, not flagged — runs whose laser envelope oscillates (seven or
   more turning points at 5 % depth or more). They are simply absent; there is no column to filter.
5. **Envelope flag** — Campaign A carries `envelope_validated`, True when `p_1` ≤ 73.888 mbar.
   Campaign A was simulated with a laser envelope solver. Full-wave reruns of five of its runs
   reproduce the beam up to that pressure and not above it, so the runs above it are kept but
   flagged: their end-state energy is the least trustworthy number in the row. Filter on it,
   weight by it, or ignore it, and say which you did.

## Two id conventions

- **`config` is one simulation within its campaign.** It is unique in both Campaign A scans and
  in Campaign B, but the same number can name one simulation in A and another in B, so never join
  across campaigns. It is the column every test file joins on, and what every submission is keyed
  by. It is the only id in the pack. Because one row is one simulation, an ordinary row-wise
  split is already a split on simulations — there is nothing to group.
- **The moments table spells the id differently.** It names the simulation in `sim_id` as the
  string `"Config_1001"`; everywhere else in the pack, `test_moments_planes.parquet` included, the
  same id is the integer `1001`. Strip the prefix before joining:
  `mom["config"] = mom.sim_id.str.removeprefix("Config_").astype(int)`.
- **The moments table spells three centroids differently too.** It stores them as `mean_zeta_um`,
  `mean_y_um` and `mean_z_um`; the scorer and the hidden true values call them `mean_zeta`, `mean_y` and
  `mean_z`, in the same µm. Rename before you submit, or the scorer stops on a missing column.
- **A run that produced no bunch has no beam numbers.** On every row where `injected` is False,
  `q_pC`, the energies, the spread and `spec_common` are NaN (2,404 such rows in Campaign B).
- **`group` in the moments table:** `all` is every stored macro-particle; `cohort` is a selected
  sub-population. Use `all`: the hidden answers are built from it.
- **Seven moments columns describe how each plane was reconstructed**, not the bunch:
  `ref_plane_mm`, `ts_lo`, `ts_hi`, `frac`, `interp`, `ess`, `charge_closure`. You can ignore
  them; `data/ontologies/pack_schema.json` says what each one is.
- **Some moments rows are empty.** 2,637 of the 105,720 `group == "all"` rows have `q_tot_pC = 0`
  and NaN in every moment: at that plane the bunch has not formed yet (most are at 2.0–2.5 mm),
  or the simulation never formed one (4 configurations, every plane). Drop them before training.
- **Four centroids are close to zero by symmetry.** `mean_y`, `mean_z`, `mean_uy` and `mean_uz`
  are transverse offsets of a bunch that sits on axis, so they are mostly simulation noise and an
  R² near 0 on them is expected; `mean_ux` is the momentum along the beam (median about 217).

## Splits, and the three axes beyond accuracy

In-distribution R² with all the data is one axis, and on Campaign A's settings → endpoint task it
is saturated. These are the axes the challenge's three questions actually live on. Each ships as
an **id list** inside `pack_reference.json`, not as a pre-cut training file: the whole corpus is
here, and the section says which configurations a comparable model may train on. A team that
trains on everything simply stops being comparable.

- **Your own validation set** — no fold ships. Every public configuration is yours to train on;
  hold some out yourself to check your model before you submit. Comparable numbers across teams
  come from the organisers, who score every submission on the hidden test.
- **Data budget** — train at 100, 300 and 1,000 configurations and on all 2,855 public
  configurations with trajectories, and submit one `test_trajectory` prediction per size; the
  organisers score all four on the same hidden configurations. **No subsets ship: you choose which
  configurations make up each size**, and choosing them well is part of the task. Our reference
  curve is the mean and spread over ten random draws of each size, so compare your choice with
  random draws too.
- **Out of distribution, density** (`ood_split`) — train on its `train` list, the 2,277 public
  configurations below 6,528.6 Pa of `P_max`, then submit **one** model's `test_trajectory`
  predictions; the organisers score it on two hidden sets, 221 configurations inside that range
  and 136 above it. Neither is trained on, so the gap between the two numbers is extrapolation
  and nothing else.
- **Out of distribution, position** (`plane_holdout`) — train on 11 planes and predict the bunch at
  9 it has never seen (2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 5.0 mm). **Read 3.5 and 5.0 apart
  from the rest:** the other seven are interpolated across a 0.2 mm gap, those two across 0.6 mm
  and 2.0 mm.

The `reference_split` section is the fold the internal reference models were trained on; its test
configurations are the hidden test and are not in the public tables, so **the internal reference
numbers are a target to aim at, not a number you can reproduce** — the organisers' score of your
submission is what goes beside them.

## Scoring

`pallas_score.py`, in the challenge folder above this one, is the scorer the organisers run.

**Submit physical values.** The scale each target is scored on is applied for you — energy
linearly, charge as log₁₀ of pC floored at 0.1, energy spread as log₁₀ of percent.

**The position-dependent tasks are scored against the per-position mean.** R² divides by the
spread across configurations *at each position*, summed over positions — not by the spread over
every cell at once. Most of the variance along a trajectory is the z-trend itself, and a global
denominator would reward reproducing the average curve shape, which is trivial. This baseline
asks whether configurations can be told apart at each position, and it is the baseline every
reference number here uses.

**Every trajectory score is ranked inside the plasma only**; the scorer reports the drift after it
for information (free-space transport, not the physics the simulation is run for). The split is
at each configuration's own plasma end, `L_inj + 3.2` mm — the shipped density profile's own
support, not a flat plane. Because that boundary moves with `L_inj`, a position is not wholly one
zone or the other: 44 positions are scored in-plasma and 77 in the drift, and they overlap. Most
of the stored range is drift, so **a single pooled number would mostly score the beam after the
physics is over** — the scorer's default is therefore the in-plasma zone. A position carrying fewer than 30 configurations of a zone is dropped
from that zone's sum, and the scorer reports how many it scored and dropped.

Two of the six quantities, `emit_um` and `div_mrad`, are scored **in-plasma only**. Past the
plasma end `div_mrad` is a waist proxy that falls as β grows, and `emit_um` falls as halo
particles leave the diagnostic; both measure a definition there rather than the beam.

## Ontologies / schema

`ontologies/pack_schema.json` lists every file in the pack, every column, its dtype, its unit and
one line on what it means — generated from the pack itself, not written by hand.
