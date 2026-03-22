#!/usr/bin/env python3
"""Solar-system trajectory simulator with multi-stage mission planning.

Capabilities:
- Spacecraft mass and propellant depletion.
- Burn schedules with repeats.
- Optional third-body gravity from all planets.
- Multi-stage missions (e.g., Earth->Mars orbit, then Mars->Earth orbit).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Physical constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
SUN_MASS = 1.98847e30  # kg
AU = 1.495978707e11  # m
DAY = 86400.0  # s
G0 = 9.80665  # m/s^2

# Simplified circular-orbit ephemerides and masses.
PLANETS = {
    "mercury": {"a": 0.387098 * AU, "period": 87.969 * DAY, "phase": 0.0, "mass": 3.3011e23},
    "venus": {"a": 0.723332 * AU, "period": 224.701 * DAY, "phase": 0.2, "mass": 4.8675e24},
    "earth": {"a": 1.0 * AU, "period": 365.256 * DAY, "phase": 0.4, "mass": 5.97237e24},
    "mars": {"a": 1.523679 * AU, "period": 686.980 * DAY, "phase": 0.9, "mass": 6.4171e23},
    "jupiter": {"a": 5.2044 * AU, "period": 4332.589 * DAY, "phase": 1.2, "mass": 1.8982e27},
    "saturn": {"a": 9.5826 * AU, "period": 10759.22 * DAY, "phase": 2.1, "mass": 5.6834e26},
    "uranus": {"a": 19.2184 * AU, "period": 30685.4 * DAY, "phase": 3.4, "mass": 8.6810e25},
    "neptune": {"a": 30.11 * AU, "period": 60189.0 * DAY, "phase": 5.0, "mass": 1.02413e26},
}


@dataclass(frozen=True)
class Spacecraft:
    initial_mass_kg: float
    dry_mass_kg: float


@dataclass(frozen=True)
class Burn:
    start: float  # s from stage start
    duration: float  # s
    thrust: tuple[float, float]  # N inertial x/y
    repeat_interval: float | None = None  # s
    repeat_count: int = 1
    isp_s: float | None = None  # s
    mass_flow_kg_s: float | None = None  # kg/s

    def active(self, t_stage: float) -> bool:
        if self.repeat_interval is None:
            return self.start <= t_stage < self.start + self.duration
        if t_stage < self.start:
            return False
        elapsed = t_stage - self.start
        cycle = int(elapsed // self.repeat_interval)
        if cycle < 0 or cycle >= self.repeat_count:
            return False
        in_cycle = elapsed - cycle * self.repeat_interval
        return 0 <= in_cycle < self.duration

    def mdot(self) -> float:
        if self.mass_flow_kg_s is not None:
            return self.mass_flow_kg_s
        if self.isp_s is not None:
            magnitude = math.hypot(*self.thrust)
            return magnitude / (self.isp_s * G0)
        return 0.0


@dataclass(frozen=True)
class MissionStage:
    name: str
    destination: str
    sim_days: float
    burns: list[Burn]
    snap_to_destination_orbit: bool = True


@dataclass(frozen=True)
class MissionConfig:
    origin: str
    launch_day: float
    dt_seconds: float
    include_planet_gravity: bool
    spacecraft: Spacecraft
    stages: list[MissionStage]


def planet_state(name: str, t_abs: float) -> tuple[float, float, float, float]:
    data = PLANETS[name.lower()]
    omega = 2.0 * math.pi / data["period"]
    theta = data["phase"] + omega * t_abs
    r = data["a"]
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    vx = -r * omega * math.sin(theta)
    vy = r * omega * math.cos(theta)
    return x, y, vx, vy


def point_mass_accel(x: float, y: float, bx: float, by: float, body_mass: float) -> tuple[float, float]:
    dx = x - bx
    dy = y - by
    r2 = dx * dx + dy * dy
    # Singularity softening for co-located states at initialization/capture.
    softened_r2 = max(r2, 1.0e8)
    r = math.sqrt(softened_r2)
    factor = -G * body_mass / (softened_r2 * r)
    return factor * dx, factor * dy


def gravity_accel(x: float, y: float, t_abs: float, include_planets: bool) -> tuple[float, float]:
    ax, ay = point_mass_accel(x, y, 0.0, 0.0, SUN_MASS)
    if include_planets:
        for name, pdata in PLANETS.items():
            px, py, _, _ = planet_state(name, t_abs)
            pax, pay = point_mass_accel(x, y, px, py, pdata["mass"])
            ax += pax
            ay += pay
    return ax, ay


def burn_force_and_mdot(burns: Iterable[Burn], t_stage: float) -> tuple[float, float, float]:
    fx, fy, mdot = 0.0, 0.0, 0.0
    for burn in burns:
        if burn.active(t_stage):
            fx += burn.thrust[0]
            fy += burn.thrust[1]
            mdot += burn.mdot()
    return fx, fy, mdot


def rk4_step(
    x: float,
    y: float,
    vx: float,
    vy: float,
    mass_kg: float,
    dry_mass_kg: float,
    t_abs: float,
    t_stage: float,
    dt: float,
    burns: Iterable[Burn],
    include_planets: bool,
) -> tuple[float, float, float, float, float]:
    def deriv(state: tuple[float, float, float, float, float], abs_time: float, stage_time: float) -> tuple[float, float, float, float, float]:
        sx, sy, svx, svy, sm = state
        gx, gy = gravity_accel(sx, sy, abs_time, include_planets)
        fx, fy, mdot = burn_force_and_mdot(burns, stage_time)

        effective_mdot = mdot if sm > dry_mass_kg else 0.0
        inv_mass = 1.0 / max(sm, dry_mass_kg)
        ax = gx + fx * inv_mass
        ay = gy + fy * inv_mass
        return svx, svy, ax, ay, -effective_mdot

    s0 = (x, y, vx, vy, mass_kg)
    k1 = deriv(s0, t_abs, t_stage)

    s1 = tuple(s0[i] + 0.5 * dt * k1[i] for i in range(5))
    k2 = deriv(s1, t_abs + 0.5 * dt, t_stage + 0.5 * dt)

    s2 = tuple(s0[i] + 0.5 * dt * k2[i] for i in range(5))
    k3 = deriv(s2, t_abs + 0.5 * dt, t_stage + 0.5 * dt)

    s3 = tuple(s0[i] + dt * k3[i] for i in range(5))
    k4 = deriv(s3, t_abs + dt, t_stage + dt)

    next_state = tuple(
        s0[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        for i in range(5)
    )

    nx, ny, nvx, nvy, nm = next_state
    return nx, ny, nvx, nvy, max(nm, dry_mass_kg)


def burn_from_config(entry: dict, initial_mass_kg: float) -> Burn:
    if "thrust_x_n" in entry or "thrust_y_n" in entry:
        thrust = (float(entry.get("thrust_x_n", 0.0)), float(entry.get("thrust_y_n", 0.0)))
    else:
        # Backward compatible format (ax/ay) converted to force at initial mass.
        ax = float(entry.get("ax", 0.0))
        ay = float(entry.get("ay", 0.0))
        thrust = (ax * initial_mass_kg, ay * initial_mass_kg)

    return Burn(
        start=float(entry["start_days"]) * DAY,
        duration=float(entry["duration_hours"]) * 3600.0,
        thrust=thrust,
        repeat_interval=(None if "repeat_every_days" not in entry else float(entry["repeat_every_days"]) * DAY),
        repeat_count=int(entry.get("repeat_count", 1)),
        isp_s=(None if "isp_s" not in entry else float(entry["isp_s"])),
        mass_flow_kg_s=(None if "mass_flow_kg_s" not in entry else float(entry["mass_flow_kg_s"])),
    )


def stage_from_config(entry: dict, initial_mass_kg: float) -> MissionStage:
    destination = entry["destination"].lower()
    if destination not in PLANETS:
        raise ValueError(f"stage destination must be one of {', '.join(PLANETS)}")

    burns = [burn_from_config(b, initial_mass_kg) for b in entry.get("burns", [])]
    return MissionStage(
        name=str(entry.get("name", f"transfer_to_{destination}")),
        destination=destination,
        sim_days=float(entry.get("sim_days", 120.0)),
        burns=burns,
        snap_to_destination_orbit=bool(entry.get("snap_to_destination_orbit", True)),
    )


def load_config(path: Path) -> MissionConfig:
    raw = json.loads(path.read_text())

    origin = raw["origin"].lower()
    if origin not in PLANETS:
        raise ValueError(f"origin must be one of {', '.join(PLANETS)}")

    sc_raw = raw.get("spacecraft", {})
    spacecraft = Spacecraft(
        initial_mass_kg=float(sc_raw.get("initial_mass_kg", 1500.0)),
        dry_mass_kg=float(sc_raw.get("dry_mass_kg", 900.0)),
    )
    if spacecraft.initial_mass_kg <= 0:
        raise ValueError("spacecraft.initial_mass_kg must be > 0")
    if spacecraft.dry_mass_kg <= 0 or spacecraft.dry_mass_kg > spacecraft.initial_mass_kg:
        raise ValueError("spacecraft.dry_mass_kg must be > 0 and <= initial_mass_kg")

    # Preferred format: explicit mission stages.
    if "stages" in raw:
        stages = [stage_from_config(s, spacecraft.initial_mass_kg) for s in raw["stages"]]
    else:
        # Backward compatibility for one-stage missions.
        legacy_stage = {
            "name": "legacy_stage",
            "destination": raw["destination"],
            "sim_days": raw.get("sim_days", 365.0),
            "burns": raw.get("burns", []),
        }
        stages = [stage_from_config(legacy_stage, spacecraft.initial_mass_kg)]

    if not stages:
        raise ValueError("config must include at least one stage")

    return MissionConfig(
        origin=origin,
        launch_day=float(raw.get("launch_day", 0.0)),
        dt_seconds=float(raw.get("dt_seconds", 600.0)),
        include_planet_gravity=bool(raw.get("include_planet_gravity", True)),
        spacecraft=spacecraft,
        stages=stages,
    )


def run_stage(
    stage_idx: int,
    stage: MissionStage,
    t_abs_start: float,
    x: float,
    y: float,
    vx: float,
    vy: float,
    mass_kg: float,
    cfg: MissionConfig,
) -> tuple[list[dict[str, float]], float, float, float, float, float, str]:
    dt = cfg.dt_seconds
    t_abs = t_abs_start
    t_stage = 0.0
    stage_end = stage.sim_days * DAY

    rows: list[dict[str, float]] = []
    while t_stage <= stage_end:
        tx, ty, tvx, tvy = planet_state(stage.destination, t_abs)
        distance = math.hypot(x - tx, y - ty)
        rows.append(
            {
                "stage_index": float(stage_idx),
                "stage_name": stage.name,
                "target": stage.destination,
                "t_abs_days": t_abs / DAY,
                "t_stage_days": t_stage / DAY,
                "ship_x_au": x / AU,
                "ship_y_au": y / AU,
                "target_x_au": tx / AU,
                "target_y_au": ty / AU,
                "distance_au": distance / AU,
                "speed_kms": math.hypot(vx, vy) / 1000.0,
                "mass_kg": mass_kg,
            }
        )

        if t_stage >= stage_end:
            break

        x, y, vx, vy, mass_kg = rk4_step(
            x,
            y,
            vx,
            vy,
            mass_kg,
            cfg.spacecraft.dry_mass_kg,
            t_abs,
            t_stage,
            dt,
            stage.burns,
            cfg.include_planet_gravity,
        )
        t_abs += dt
        t_stage += dt

    # Stage capture option: approximate successful orbit insertion around destination.
    if stage.snap_to_destination_orbit:
        x, y, vx, vy = planet_state(stage.destination, t_abs)

    return rows, t_abs, x, y, vx, vy, stage.destination


def simulate(cfg: MissionConfig) -> list[dict[str, float]]:
    t_abs = cfg.launch_day * DAY
    x, y, vx, vy = planet_state(cfg.origin, t_abs)
    mass_kg = cfg.spacecraft.initial_mass_kg

    history: list[dict[str, float]] = []
    for idx, stage in enumerate(cfg.stages, start=1):
        stage_rows, t_abs, x, y, vx, vy, _ = run_stage(
            stage_idx=idx,
            stage=stage,
            t_abs_start=t_abs,
            x=x,
            y=y,
            vx=vx,
            vy=vy,
            mass_kg=mass_kg,
            cfg=cfg,
        )
        mass_kg = stage_rows[-1]["mass_kg"]
        history.extend(stage_rows)

    return history


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_plot(path: Path, rows: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for --plot") from exc

    ship_x = [r["ship_x_au"] for r in rows]
    ship_y = [r["ship_y_au"] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ship_x, ship_y, label="Spacecraft", color="tab:blue", linewidth=1.0)
    ax.scatter([0], [0], label="Sun", s=120, c="gold", edgecolors="black")

    for name in PLANETS:
        # Draw each planetary orbit to support "all planets" analysis.
        orbit_x = []
        orbit_y = []
        for i in range(361):
            theta_t = PLANETS[name]["period"] * i / 360.0
            px, py, _, _ = planet_state(name, theta_t)
            orbit_x.append(px / AU)
            orbit_y.append(py / AU)
        ax.plot(orbit_x, orbit_y, alpha=0.25, linewidth=0.8)

    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_title("Solar-system multi-stage trajectory")
    ax.axis("equal")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)


def best_approach(rows: list[dict[str, float]]) -> dict[str, float]:
    return min(rows, key=lambda r: r["distance_au"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Mission JSON configuration file")
    parser.add_argument("--csv", type=Path, default=Path("journey.csv"), help="Output CSV path")
    parser.add_argument("--plot", type=Path, help="Optional PNG plot output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rows = simulate(cfg)
    write_csv(args.csv, rows)

    if args.plot:
        render_plot(args.plot, rows)

    closest = best_approach(rows)
    final_mass = rows[-1]["mass_kg"]
    prop_used = cfg.spacecraft.initial_mass_kg - final_mass

    print("Simulation complete")
    print(f"  stages: {len(cfg.stages)}")
    print(f"  samples: {len(rows)}")
    print(f"  closest approach: {closest['distance_au']:.4f} AU")
    print(f"  during stage: {int(closest['stage_index'])} ({closest['stage_name']})")
    print(f"  final mass: {final_mass:.2f} kg")
    print(f"  propellant used: {prop_used:.2f} kg")
    print(f"  output csv: {args.csv}")
    if args.plot:
        print(f"  output plot: {args.plot}")


if __name__ == "__main__":
    main()
