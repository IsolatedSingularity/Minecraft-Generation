"""Render every README visualization from deterministic inputs."""

from pathlib import Path
import time

import matplotlib

matplotlib.use('Agg')

from dragon_pathfinding import (
    create_dragon_detail_clips,
    create_dragon_pathfinding_animation,
    create_trajectory_ensemble_animation,
)
from end_dimension_overview import create_end_dimension_overview
from minecraftStructureAnalysis import create_structure_analysis
from multi_structure_generation import create_multi_structure_animation
from seed_loading import create_seed_loading_animation
from stronghold_distribution import create_stronghold_distribution
from structure_placement import create_structure_placement_animation


def _render(label, render):
    start = time.perf_counter()
    output = render()
    elapsed = time.perf_counter() - start
    print(f'{label}: {elapsed:.1f}s')
    return output


def main():
    root = Path(__file__).resolve().parents[1]
    plots = root / 'Plots'
    plots.mkdir(exist_ok=True)

    _render('End overview', lambda: create_end_dimension_overview(
        plots / 'end_dimension_overview.png'))
    _render('Structure analysis', lambda: create_structure_analysis(
        plots / 'structure_analysis.png'))
    _render('Stronghold rings', lambda: create_stronghold_distribution(
        plots / 'stronghold_rings.png'))

    _render('Dragon hero', lambda: create_dragon_pathfinding_animation(
        plots / 'dragon_pathfinding.gif'))
    _render('Dragon detail clips', lambda: create_dragon_detail_clips(plots))
    _render('Dragon ensemble', lambda: create_trajectory_ensemble_animation(
        plots / 'dragon_trajectory_ensemble.gif'))
    _render('Seed loading', lambda: create_seed_loading_animation(
        plots / 'seed_loading.gif'))
    _render('Village placement', lambda: create_structure_placement_animation(
        plots / 'structure_placement.gif'))
    _render('Nether structures', lambda: create_multi_structure_animation(
        plots / 'multi_structure_generation.gif'))


if __name__ == '__main__':
    main()
