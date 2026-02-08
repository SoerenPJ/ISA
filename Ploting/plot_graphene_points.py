import sys
from pathlib import Path
import math

import numpy as np
import matplotlib.pyplot as plt


def resolve_config_path(arg: str) -> Path:
    p = Path(arg)
    if p.exists() and p.is_file():
        return p
    if not p.suffix:
        p2 = Path(str(p) + ".toml")
        if p2.exists() and p2.is_file():
            return p2
    raise FileNotFoundError(f"Could not find config file: {arg} (tried '{p}' and '{p}.toml')")


def parse_simple_toml(path: Path) -> dict:
    """
    Minimal TOML reader for this project.
    Supports:
      - [section]
      - key = value (int/float/bool/"string")
      - ignores comments starting with '#'
    Only intended for the keys we need to plot the lattice.
    """
    cfg: dict[str, dict] = {}
    section = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            cfg.setdefault(section, {})
            continue
        if "=" not in line or section is None:
            continue
        k, v = [x.strip() for x in line.split("=", 1)]
        # strings
        if len(v) >= 2 and ((v[0] == '"' and v[-1] == '"') or (v[0] == "'" and v[-1] == "'")):
            val = v[1:-1]
        elif v.lower() in ("true", "false"):
            val = (v.lower() == "true")
        else:
            # number (int/float/scientific)
            try:
                if any(ch in v for ch in (".", "e", "E")):
                    val = float(v)
                else:
                    val = int(v)
            except ValueError:
                # fallback: keep as raw string
                val = v
        cfg[section][k] = val
    return cfg


def generate_hexagon_start_at_zero(bond_length: float) -> np.ndarray:
    angles_deg = np.array([180, 240, 300, 0, 60, 120], dtype=float)
    angles_rad = np.deg2rad(angles_deg)
    x = bond_length * np.cos(angles_rad)
    y = bond_length * np.sin(angles_rad)
    basehex = np.column_stack((x, y))
    basehex = basehex - basehex[0]
    return basehex


def generate_positions(size_x: int, size_y: int, formation_shape: str, triangle_rule: str):
    formation_shape = formation_shape.lower()
    triangle_rule = triangle_rule.lower()
    positions = []
    if formation_shape == "rectangle":
        positions = [(i, j) for j in range(size_y) for i in range(size_x)]
    elif formation_shape == "triangle":
        # Only "right" is supported (armchair removed)
        positions = [(i, j) for j in range(size_y) for i in range(size_x) if i + j <= size_x - 1]
    else:
        positions = [(i, j) for j in range(size_y) for i in range(size_x)]
    return positions


def build_molecule_points(hexagon: np.ndarray, size_x: int, size_y: int, formation="zigzag", formation_shape="rectangle", triangle_rule="right") -> np.ndarray:
    formation = formation.lower()
    formation_shape = formation_shape.lower()

    # Armchair removed: force zigzag
    formation = "zigzag"

    positions = generate_positions(size_x, size_y, formation_shape, triangle_rule)
    hx = hexagon.copy()

    # Match the C++ implementation.
    move_x = hx[3] - hx[5]
    move_y = hx[2] - hx[4]

    molecule = []
    for (i, j) in positions:
        molecule.append(hexagon + i * move_x + j * move_y)

    points = np.vstack(molecule)
    rounded = np.round(points, decimals=6)
    unique = np.unique(rounded, axis=0)
    return unique


def plot_points(points: np.ndarray, bond_length: float, title: str = "", label: bool = False) -> None:
    n = points.shape[0]
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=20, c="k")

    if label:
        for i, (x, y) in enumerate(points):
            plt.text(x + 0.02 * bond_length, y + 0.02 * bond_length, str(i), fontsize=7, color="red")

    # draw bonds by distance threshold
    cutoff = bond_length * 1.02
    cutoff2 = cutoff * cutoff
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            dx = xi - points[j, 0]
            dy = yi - points[j, 1]
            if dx * dx + dy * dy <= cutoff2:
                plt.plot([xi, points[j, 0]], [yi, points[j, 1]], "b-", linewidth=0.8, alpha=0.8)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python3 Ploting/plot_graphene_points.py <config.toml | configs/name>")
        return 1

    cfg_path = resolve_config_path(sys.argv[1])
    cfg = parse_simple_toml(cfg_path)
    sys_cfg = cfg.get("system", {})

    lattice = str(sys_cfg.get("lattice", "graphene")).lower()
    if lattice != "graphene":
        raise ValueError(f"system.lattice must be 'graphene' for this tool (got: {lattice})")

    formation = str(sys_cfg.get("formation", "zigzag"))
    shape = str(sys_cfg.get("formation_shape", "rectangle"))
    triangle_rule = str(sys_cfg.get("triangle_rule", "right"))
    size_x = int(sys_cfg.get("size_x", 10))
    size_y = int(sys_cfg.get("size_y", 6))

    # In configs we use lattice_const as bond length in nm (same as your other configs).
    d = float(sys_cfg.get("lattice_const", 0.142))

    hex0 = generate_hexagon_start_at_zero(d)

    # Armchair removed: always plots zigzag

    pts = build_molecule_points(hex0, size_x=size_x, size_y=size_y, formation=formation, formation_shape=shape, triangle_rule=triangle_rule)
    title = f"{cfg_path.name} | {formation=} {shape=} {triangle_rule=} | hexagons=({size_x},{size_y}) | atoms={pts.shape[0]}"
    label = ("--label" in sys.argv)
    plot_points(pts, bond_length=d, title=title, label=label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

