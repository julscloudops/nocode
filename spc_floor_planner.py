"""
SPC Click Floor Planner

Plans and visualises SPC (Stone Plastic Composite) click-lock plank
arrangements for rectangular rooms with optional rectangular obstacles
(pillars, islands, hearths, etc.).

Usage
-----
    python spc_floor_planner.py

or import and call programmatically:

    from spc_floor_planner import Room, Plank, FloorPlanner, Pattern

    room   = Room(width_mm=3780, length_mm=4390)
    plank  = Plank.standard("1220x182")
    layout = FloorPlanner(room, plank, pattern=Pattern.THIRD_OFFSET).plan()
    layout.report()
    layout.save("kitchen_layout.png")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class Pattern(str, Enum):
    STRAIGHT     = "straight"       # no stagger (rarely used)
    HALF_OFFSET  = "half_offset"    # each row offset by plank_length / 2
    THIRD_OFFSET = "third_offset"   # each row offset by plank_length / 3
    QUARTER_OFFSET = "quarter_offset"


STAGGER_FRACTIONS: dict[Pattern, float] = {
    Pattern.STRAIGHT:       0.0,
    Pattern.HALF_OFFSET:    0.5,
    Pattern.THIRD_OFFSET:   1 / 3,
    Pattern.QUARTER_OFFSET: 0.25,
}

STANDARD_PLANKS: dict[str, Tuple[float, float]] = {
    "1220x182": (1220, 182),
    "1524x228": (1524, 228),
    "1830x230": (1830, 230),
    "1220x150": (1220, 150),
    "915x152":  (915,  152),
}


@dataclass
class Plank:
    length_mm: float
    width_mm: float

    @classmethod
    def standard(cls, key: str) -> "Plank":
        if key not in STANDARD_PLANKS:
            raise ValueError(f"Unknown standard size '{key}'. Choose from: {list(STANDARD_PLANKS)}")
        l, w = STANDARD_PLANKS[key]
        return cls(length_mm=l, width_mm=w)

    @property
    def area_m2(self) -> float:
        return (self.length_mm / 1000) * (self.width_mm / 1000)


@dataclass
class Obstacle:
    """A rectangular area that cannot be covered (pillar, island, hearth…)."""
    x_mm: float       # distance from left wall (inside expansion gap)
    y_mm: float       # distance from bottom wall (inside expansion gap)
    width_mm: float
    height_mm: float
    label: str = ""


@dataclass
class Room:
    width_mm: float    # horizontal dimension
    length_mm: float   # vertical dimension
    obstacles: List[Obstacle] = field(default_factory=list)

    @property
    def area_m2(self) -> float:
        return (self.width_mm / 1000) * (self.length_mm / 1000)

    def obstacle_area_m2(self) -> float:
        return sum(
            (o.width_mm / 1000) * (o.height_mm / 1000) for o in self.obstacles
        )

    def floored_area_m2(self) -> float:
        return self.area_m2 - self.obstacle_area_m2()


# ---------------------------------------------------------------------------
# Plank cut representation
# ---------------------------------------------------------------------------

@dataclass
class PlankPiece:
    """A single placed plank piece (may be a cut piece)."""
    x_mm: float          # left edge in room coordinates
    y_mm: float          # bottom edge in room coordinates
    width_mm: float      # plank width (across direction of laying)
    length_mm: float     # plank length (along direction of laying)
    is_cut: bool = False # True when shorter than a full plank
    cut_offcut_mm: float = 0.0  # offcut length that may be reused

    @property
    def area_m2(self) -> float:
        return (self.width_mm / 1000) * (self.length_mm / 1000)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class FloorPlanner:
    """
    Lays SPC planks row-by-row along the room width (planks run left→right).
    Rows advance from the bottom wall upward.

    Parameters
    ----------
    room : Room
    plank : Plank
    pattern : Pattern
    expansion_gap_mm : float
        Gap left around the perimeter (hidden under skirting). Default 10 mm.
    waste_factor : float
        Extra material to order as a fraction (e.g. 0.10 = 10 % extra).
    """

    def __init__(
        self,
        room: Room,
        plank: Plank,
        pattern: Pattern = Pattern.THIRD_OFFSET,
        expansion_gap_mm: float = 10.0,
        waste_factor: float = 0.10,
    ):
        self.room = room
        self.plank = plank
        self.pattern = pattern
        self.gap = expansion_gap_mm
        self.waste_factor = waste_factor

    # ------------------------------------------------------------------
    # Core planning
    # ------------------------------------------------------------------

    def plan(self) -> "FloorLayout":
        pieces: List[PlankPiece] = []
        stagger = STAGGER_FRACTIONS[self.pattern] * self.plank.length_mm

        # Usable interior dimensions (inside expansion gap)
        usable_w = self.room.width_mm  - 2 * self.gap
        usable_h = self.room.length_mm - 2 * self.gap

        row_index = 0
        y = self.gap  # current row bottom edge

        while y < self.room.length_mm - self.gap:
            row_width = min(self.plank.width_mm, self.room.length_mm - self.gap - y)
            if row_width <= 0:
                break

            row_offset = (row_index * stagger) % self.plank.length_mm
            x = self.gap - row_offset  # may start before left wall

            while x < self.room.width_mm - self.gap:
                piece_x = max(x, self.gap)
                piece_end = min(x + self.plank.length_mm, self.room.width_mm - self.gap)
                piece_len = piece_end - piece_x

                if piece_len > 1:  # skip slivers < 1 mm
                    full_len = self.plank.length_mm
                    is_cut = piece_len < full_len - 0.5
                    # offcut = the remainder of the plank not placed here
                    offcut = full_len - (piece_end - x) if x >= self.gap else (piece_x - x)
                    offcut = max(0.0, full_len - piece_len - max(0.0, self.gap - x))

                    pieces.append(PlankPiece(
                        x_mm=piece_x,
                        y_mm=y,
                        width_mm=row_width,
                        length_mm=piece_len,
                        is_cut=is_cut,
                        cut_offcut_mm=offcut if is_cut else 0.0,
                    ))

                x += self.plank.length_mm

            y += self.plank.width_mm
            row_index += 1

        return FloorLayout(
            room=self.room,
            plank=self.plank,
            pattern=self.pattern,
            expansion_gap_mm=self.gap,
            waste_factor=self.waste_factor,
            pieces=pieces,
        )


# ---------------------------------------------------------------------------
# Layout result
# ---------------------------------------------------------------------------

class FloorLayout:
    def __init__(
        self,
        room: Room,
        plank: Plank,
        pattern: Pattern,
        expansion_gap_mm: float,
        waste_factor: float,
        pieces: List[PlankPiece],
    ):
        self.room = room
        self.plank = plank
        self.pattern = pattern
        self.gap = expansion_gap_mm
        self.waste_factor = waste_factor
        self.pieces = pieces

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def full_planks(self) -> int:
        return sum(1 for p in self.pieces if not p.is_cut)

    @property
    def cut_planks(self) -> int:
        return sum(1 for p in self.pieces if p.is_cut)

    @property
    def total_pieces(self) -> int:
        return len(self.pieces)

    def _full_planks_needed(self) -> int:
        """Number of full planks consumed (cut planks each use 1 full plank)."""
        return self.full_planks + self.cut_planks

    @property
    def planks_to_order(self) -> int:
        base = self._full_planks_needed()
        return math.ceil(base * (1 + self.waste_factor))

    @property
    def boxes_to_order(self) -> Tuple[int, float]:
        """Returns (boxes, planks_per_box) assuming ~2 m² per box."""
        box_area = 2.0  # m²
        planks_per_box = max(1, math.floor(box_area / self.plank.area_m2))
        boxes = math.ceil(self.planks_to_order / planks_per_box)
        return boxes, planks_per_box

    @property
    def material_waste_pct(self) -> float:
        placed_area = sum(p.area_m2 for p in self.pieces)
        full_plank_area = self._full_planks_needed() * self.plank.area_m2
        if full_plank_area == 0:
            return 0.0
        return (full_plank_area - placed_area) / full_plank_area * 100

    # ------------------------------------------------------------------
    # Text report
    # ------------------------------------------------------------------

    def report(self) -> str:
        boxes, ppb = self.boxes_to_order
        lines = [
            "=" * 56,
            "  SPC CLICK FLOOR PLAN — INSTALLATION REPORT",
            "=" * 56,
            f"  Room size          : {self.room.width_mm/1000:.2f} m × "
            f"{self.room.length_mm/1000:.2f} m",
            f"  Room area          : {self.room.area_m2:.2f} m²",
            f"  Floored area       : {self.room.floored_area_m2():.2f} m²",
            f"  Expansion gap      : {self.gap:.0f} mm",
            "",
            f"  Plank size         : {self.plank.length_mm:.0f} × "
            f"{self.plank.width_mm:.0f} mm",
            f"  Layout pattern     : {self.pattern.value.replace('_', ' ').title()}",
            "",
            f"  Full planks placed : {self.full_planks}",
            f"  Cut planks placed  : {self.cut_planks}",
            f"  Total pieces       : {self.total_pieces}",
            f"  Cut waste          : {self.material_waste_pct:.1f} %",
            "",
            f"  Planks to order    : {self.planks_to_order}  "
            f"(+{self.waste_factor*100:.0f} % buffer)",
            f"  Approx. boxes      : {boxes}  (~{ppb} planks/box, ~2 m²/box)",
            "=" * 56,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(
        self,
        title: str = "SPC Click Floor Plan",
        show: bool = True,
        save_path: Optional[str] = None,
        figsize: Tuple[float, float] = (12, 10),
    ) -> plt.Figure:
        rw = self.room.width_mm
        rl = self.room.length_mm

        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={"width_ratios": [3, 1]})
        ax, ax_info = axes

        ax.set_xlim(-50, rw + 50)
        ax.set_ylim(-50, rl + 50)
        ax.set_aspect("equal")
        ax.set_facecolor("#f5f5f0")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Width (mm)")
        ax.set_ylabel("Length (mm)")

        # Colour palette — alternating shades per row to emphasise stagger
        colours = ["#c8b99a", "#b5a585", "#d4c5a9", "#a89070"]

        row_y_vals = sorted(set(round(p.y_mm, 1) for p in self.pieces))
        row_colour = {y: colours[i % len(colours)] for i, y in enumerate(row_y_vals)}

        for piece in self.pieces:
            c = row_colour[round(piece.y_mm, 1)]
            edge = "#7a6a55"
            rect = mpatches.FancyBboxPatch(
                (piece.x_mm, piece.y_mm),
                piece.length_mm,
                piece.width_mm,
                boxstyle="square,pad=0",
                linewidth=0.5,
                edgecolor=edge,
                facecolor=c,
                alpha=0.90,
            )
            ax.add_patch(rect)

            if piece.is_cut:
                cx = piece.x_mm + piece.length_mm / 2
                cy = piece.y_mm + piece.width_mm / 2
                ax.text(cx, cy, "✂", ha="center", va="center",
                        fontsize=7, color="#444", alpha=0.6)

        # Draw obstacles
        for obs in self.room.obstacles:
            orect = mpatches.Rectangle(
                (self.gap + obs.x_mm, self.gap + obs.y_mm),
                obs.width_mm,
                obs.height_mm,
                linewidth=1.5,
                edgecolor="#333",
                facecolor="#888888",
                alpha=0.7,
                zorder=5,
            )
            ax.add_patch(orect)
            if obs.label:
                ax.text(
                    self.gap + obs.x_mm + obs.width_mm / 2,
                    self.gap + obs.y_mm + obs.height_mm / 2,
                    obs.label,
                    ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold", zorder=6,
                )

        # Room border
        border = mpatches.Rectangle(
            (0, 0), rw, rl,
            linewidth=3, edgecolor="#2c2c2c", facecolor="none", zorder=10,
        )
        ax.add_patch(border)

        # Expansion gap indicator
        gap_rect = mpatches.Rectangle(
            (0, 0), rw, rl,
            linewidth=1, edgecolor="#6699cc", facecolor="none",
            linestyle="--", zorder=9,
        )
        inner_rect = mpatches.Rectangle(
            (self.gap, self.gap),
            rw - 2 * self.gap,
            rl - 2 * self.gap,
            linewidth=0.8, edgecolor="#6699cc", facecolor="none",
            linestyle="--", zorder=9,
        )
        ax.add_patch(inner_rect)

        # Dimension annotations
        _annotate_dim(ax, 0, -30, rw, -30,
                      f"{rw/1000:.2f} m", color="#336699")
        _annotate_dim(ax, -35, 0, -35, rl,
                      f"{rl/1000:.2f} m", color="#336699", vertical=True)

        # ------------------------------------------------------------------
        # Info panel
        # ------------------------------------------------------------------
        ax_info.set_axis_off()
        boxes, ppb = self.boxes_to_order
        info_text = (
            f"ROOM\n"
            f"  {rw/1000:.2f} m × {rl/1000:.2f} m\n"
            f"  Area: {self.room.area_m2:.2f} m²\n"
            f"  Floored: {self.room.floored_area_m2():.2f} m²\n\n"
            f"PLANK\n"
            f"  {self.plank.length_mm:.0f} × {self.plank.width_mm:.0f} mm\n"
            f"  {self.plank.area_m2*1e6/1e6:.4f} m² each\n\n"
            f"PATTERN\n"
            f"  {self.pattern.value.replace('_',' ').title()}\n\n"
            f"PIECES\n"
            f"  Full : {self.full_planks}\n"
            f"  Cut  : {self.cut_planks}\n"
            f"  Total: {self.total_pieces}\n"
            f"  Waste: {self.material_waste_pct:.1f} %\n\n"
            f"ORDER\n"
            f"  Planks : {self.planks_to_order}\n"
            f"  Boxes  : {boxes}\n"
            f"  Buffer : +{self.waste_factor*100:.0f} %\n\n"
            f"Expansion gap: {self.gap:.0f} mm"
        )
        ax_info.text(
            0.05, 0.97, info_text,
            transform=ax_info.transAxes,
            va="top", ha="left",
            fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#eef2f7",
                      edgecolor="#99aacc", linewidth=1),
        )

        # Legend
        legend_patches = [
            mpatches.Patch(facecolor=colours[0], edgecolor="#7a6a55", label="Full plank"),
            mpatches.Patch(facecolor=colours[1], edgecolor="#7a6a55", label="Full plank (alt row)"),
            mpatches.Patch(facecolor="#aaa", edgecolor="#333", label="Obstacle"),
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
                  framealpha=0.85)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Layout saved to: {save_path}")

        if show:
            plt.show()

        return fig

    def save(self, path: str) -> "FloorLayout":
        self.visualize(show=False, save_path=path)
        return self


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _annotate_dim(
    ax: plt.Axes,
    x1: float, y1: float,
    x2: float, y2: float,
    label: str,
    color: str = "#336699",
    vertical: bool = False,
) -> None:
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.2),
    )
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    rotation = 90 if vertical else 0
    ax.text(mx, my, label, ha="center", va="center",
            fontsize=8, color=color, rotation=rotation,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------

def plan_room(
    width_m: float,
    length_m: float,
    plank_key: str = "1220x182",
    pattern: Pattern = Pattern.THIRD_OFFSET,
    expansion_gap_mm: float = 10.0,
    waste_factor: float = 0.10,
    obstacles: Optional[List[Obstacle]] = None,
) -> FloorLayout:
    """Convenience one-liner for quick planning."""
    room = Room(
        width_mm=width_m * 1000,
        length_mm=length_m * 1000,
        obstacles=obstacles or [],
    )
    plank = Plank.standard(plank_key)
    return FloorPlanner(room, plank, pattern, expansion_gap_mm, waste_factor).plan()


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    print("\nRunning SPC floor planner demo — Kitchen 3.78 m × 4.39 m\n")

    layout = plan_room(
        width_m=3.78,
        length_m=4.39,
        plank_key="1220x182",
        pattern=Pattern.THIRD_OFFSET,
        expansion_gap_mm=10,
        waste_factor=0.10,
    )

    layout.report()
    layout.save("kitchen_layout.png")

    print("\n--- Comparing patterns ---")
    for pat in Pattern:
        l = plan_room(3.78, 4.39, pattern=pat, waste_factor=0.10)
        boxes, _ = l.boxes_to_order
        print(
            f"  {pat.value:<18}  pieces: {l.total_pieces:3d}  "
            f"waste: {l.material_waste_pct:5.1f} %  "
            f"order: {l.planks_to_order} planks ({boxes} boxes)"
        )

    print("\nDone. See kitchen_layout.png for the visual plan.")


if __name__ == "__main__":
    _demo()
