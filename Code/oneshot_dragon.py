"""Compatibility entry point for the dragon trajectory ensemble.

The old visualization claimed an integer-overflow one-shot mechanism. That
claim was inaccurate. The replacement shows accumulated, source-shaped dragon
approach trajectories and leaves player technique details to the README.
"""

from pathlib import Path

from dragon_pathfinding import create_trajectory_ensemble_animation


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'Plots' / 'dragon_trajectory_ensemble.gif'
    create_trajectory_ensemble_animation(output)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
