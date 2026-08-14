"""Render static, non-interactive flow figures used by the root README."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from core.style import COLORS, apply_style


apply_style()


DIAGRAMS = {
    'world_generation_flow.svg': {
        'size': (12.0, 3.5),
        'nodes': {
            'seed': (0.08, 0.50, 'WORLD\nSEED'),
            'rng': (0.28, 0.50, 'JAVA RANDOM\nSTATE'),
            'noise': (0.49, 0.72, 'NOISE AND\nCLIMATE FIELDS'),
            'offsets': (0.49, 0.28, 'STRUCTURE\nCANDIDATES'),
            'terrain': (0.70, 0.72, 'BIOME AND\nTERRAIN RULES'),
            'gates': (0.70, 0.28, 'BIOME AND\nHEIGHT GATES'),
            'view': (0.91, 0.50, 'WORLD\nVISUALIZATION'),
        },
        'edges': [('seed', 'rng'), ('rng', 'noise'), ('rng', 'offsets'),
                  ('noise', 'terrain'), ('offsets', 'gates'),
                  ('terrain', 'view'), ('gates', 'view')],
    },
    'noise_composition_flow.svg': {
        'size': (12.0, 3.5),
        'nodes': {
            'broad': (0.10, 0.76, 'BROAD NOISE\nCONTINENTS'),
            'medium': (0.10, 0.50, 'MEDIUM NOISE\nREGIONS'),
            'fine': (0.10, 0.24, 'FINE NOISE\nLOCAL TEXTURE'),
            'sum': (0.38, 0.50, 'WEIGHTED\nSUM'),
            'fields': (0.65, 0.50, 'ELEVATION, CLIMATE,\nAND MOISTURE'),
            'biomes': (0.90, 0.50, 'BIOME\nCLASSIFICATION'),
        },
        'edges': [('broad', 'sum'), ('medium', 'sum'), ('fine', 'sum'),
                  ('sum', 'fields'), ('fields', 'biomes')],
    },
    'dragon_navigation_flow.svg': {
        'size': (12.0, 3.7),
        'nodes': {
            'state': (0.09, 0.72, 'CURRENT\nFIGHT STATE'),
            'crystals': (0.09, 0.28, 'LIVING\nCRYSTALS'),
            'allowed': (0.34, 0.50, 'CHOOSE\nALLOWED NODES'),
            'route': (0.59, 0.50, 'SHORTEST LEGAL\nNODE ROUTE'),
            'steer': (0.81, 0.68, 'SOURCE-SHAPED\nSTEERING'),
            'motion': (0.81, 0.27, 'POSITION AND\nDIRECTION'),
        },
        'edges': [('state', 'allowed'), ('crystals', 'allowed'),
                  ('allowed', 'route'), ('route', 'steer'),
                  ('steer', 'motion'), ('motion', 'state')],
    },
    'structure_candidate_flow.svg': {
        'size': (12.0, 3.6),
        'nodes': {
            'input': (0.09, 0.50, 'WORLD SEED,\nREGION, SALT'),
            'rng': (0.29, 0.50, '48-BIT\nJAVA RANDOM'),
            'offset': (0.50, 0.50, 'UNIFORM OR\nCENTER-BIASED OFFSET'),
            'candidate': (0.70, 0.50, 'CANDIDATE\nCHUNK'),
            'map': (0.91, 0.72, 'CANDIDATE-STAGE\nMAP'),
            'checks': (0.91, 0.28, 'LATER BIOME AND\nTERRAIN CHECKS'),
        },
        'edges': [('input', 'rng'), ('rng', 'offset'), ('offset', 'candidate'),
                  ('candidate', 'map'), ('candidate', 'checks')],
    },
}


def _draw_diagram(path, specification):
    figure, axis = plt.subplots(figsize=specification['size'])
    figure.subplots_adjust(left=0.015, right=0.985, top=0.96, bottom=0.04)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis('off')
    node_artists = {}
    for index, (key, (x, y, label)) in enumerate(specification['nodes'].items()):
        color = (COLORS['cyan'], COLORS['violet'], COLORS['coral'], COLORS['gold'])[index % 4]
        width = 0.135 if len(label) < 22 else 0.165
        height = 0.115
        node = FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle='round,pad=0.010,rounding_size=0.018',
            facecolor=COLORS['panel_alt'], edgecolor=color,
            linewidth=1.65, zorder=3,
        )
        axis.add_patch(node)
        axis.text(x, y, label, ha='center', va='center', color=COLORS['text'],
                  fontsize=8.7, fontweight='black', linespacing=1.14, zorder=4)
        node_artists[key] = node
    for start, end in specification['edges']:
        arrow = FancyArrowPatch(
            posA=specification['nodes'][start][:2],
            posB=specification['nodes'][end][:2],
            patchA=node_artists[start], patchB=node_artists[end],
            arrowstyle='-|>', mutation_scale=15.5, color='#8C9AB1',
            linewidth=1.75, connectionstyle='arc3,rad=0.0', zorder=2,
        )
        axis.add_patch(arrow)
    figure.savefig(
        path, format='svg', transparent=False,
        metadata={'Date': None, 'Creator': 'Minecraft-Generation'},
    )
    plt.close(figure)
    # Matplotlib emits trailing spaces in multi-line SVG path data.  Keep the
    # generated files deterministic and friendly to ``git diff --check``.
    svg = path.read_text(encoding='utf-8')
    path.write_text(
        '\n'.join(line.rstrip() for line in svg.splitlines()) + '\n',
        encoding='utf-8', newline='\n',
    )


def create_readme_diagrams(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for name, specification in DIAGRAMS.items():
        path = output_dir / name
        _draw_diagram(path, specification)
        outputs.append(str(path))
    return outputs


def main():
    root = Path(__file__).resolve().parents[1]
    create_readme_diagrams(root / 'Plots')


if __name__ == '__main__':
    main()
