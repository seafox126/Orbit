# Solar-System Orbit Journey Simulator

A multi-stage heliocentric mission simulator for planning trajectories among all
major planets with configurable thrust timing, duration, and spacecraft mass.

## Highlights

- Supports **all included planets** (`mercury` through `neptune`) as origins,
  stage destinations, and third-body gravity contributors.
- Supports **multi-stage missions**, so you can model sequences such as:
  Earth -> Mars orbit -> Earth orbit.
- Spacecraft mass model with wet/dry mass and propellant depletion.
- Burn modeling with force vectors and either `isp_s` or `mass_flow_kg_s`.
- Optional third-body gravity (`include_planet_gravity`).
- CSV output including stage metadata and mass history.
- Optional trajectory plot (`--plot`) if `matplotlib` is installed.

## Run

```bash
python3 orbit_sim.py mission_config.example.json --csv mission.csv --plot mission.png
```

## Config schema (summary)

- `origin`: mission start planet.
- `launch_day`: absolute launch offset from model epoch.
- `dt_seconds`: integration step.
- `include_planet_gravity`: include gravity from every planet.
- `spacecraft`:
  - `initial_mass_kg`
  - `dry_mass_kg`
- `stages`: list of mission stages.
  - `name`: stage label.
  - `destination`: target planet for that stage.
  - `sim_days`: stage duration.
  - `snap_to_destination_orbit`: if true, stage ends by matching target orbit.
  - `burns`: stage-local burn schedule.
    - `start_days`, `duration_hours`
    - `thrust_x_n`, `thrust_y_n` (or legacy `ax`, `ay`)
    - optional `isp_s` and/or `mass_flow_kg_s`
    - optional `repeat_every_days`, `repeat_count`

## Professional-use note

This is a mission-design and trade-study tool, not certified flight software.
It uses simplified circular coplanar ephemerides and does not yet include full
high-fidelity perturbation suites (J2, SRP, maneuver uncertainty, etc.).
