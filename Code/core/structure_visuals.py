"""Reusable top-down schematics for Minecraft structure visualizations.

These are original, deliberately enlarged block-plan illustrations.  They are
anchored to exact candidate chunks but do not claim to reproduce a seed's
complete jigsaw/template expansion.
"""

from dataclasses import dataclass

from matplotlib.patches import Circle, Polygon, Rectangle


@dataclass(frozen=True)
class StructureSchematic:
    key: str
    label: str
    primary: str
    secondary: str
    scale: float = 1.0


STRUCTURE_SCHEMATICS = {
    item.key: item for item in (
        StructureSchematic('village', 'Village', '#D9A75F', '#F2D391', 1.15),
        StructureSchematic('desert_pyramid', 'Desert pyramid', '#E4C56D', '#D9823B', 1.05),
        StructureSchematic('jungle_pyramid', 'Jungle pyramid', '#71864B', '#A8B66C', 1.00),
        StructureSchematic('swamp_hut', 'Swamp hut', '#765744', '#A98A63', 0.92),
        StructureSchematic('pillager_outpost', 'Pillager outpost', '#6C4D3C', '#A38367', 0.94),
        StructureSchematic('igloo', 'Igloo', '#ECF4F4', '#A8CAD5', 0.90),
        StructureSchematic('woodland_mansion', 'Woodland mansion', '#5A3D32', '#8B6650', 1.25),
        StructureSchematic('ocean_monument', 'Ocean monument', '#4FA89A', '#8ED2B9', 1.20),
        StructureSchematic('shipwreck', 'Shipwreck', '#6F4B32', '#B07C49', 1.00),
        StructureSchematic('ocean_ruin', 'Ocean ruin', '#66777A', '#9DA7A3', 0.92),
        StructureSchematic('ruined_portal', 'Ruined portal', '#281633', '#B65AD2', 0.90),
        StructureSchematic('fortress', 'Nether fortress', '#6F2028', '#B8464E', 1.12),
        StructureSchematic('bastion', 'Bastion remnant', '#372C2B', '#9A7350', 1.18),
        StructureSchematic('end_city', 'End City', '#B77AB2', '#E3B5D5', 1.18),
    )
}


class StructureArtistGroup:
    """Small visibility wrapper used by animated structure maps."""

    def __init__(self, artists):
        self.artists = list(artists)

    def set_visible(self, visible):
        for artist in self.artists:
            artist.set_visible(bool(visible))

    def set_alpha(self, alpha):
        for artist in self.artists:
            artist.set_alpha(float(alpha))


def _add(ax, artists, patch, transform):
    if transform is not None:
        patch.set_transform(transform)
    ax.add_patch(patch)
    artists.append(patch)
    return patch


def draw_structure_schematic(
    ax, name, x, z, size=4.5, transform=None, zorder=10, alpha=0.98,
):
    """Draw a recognizable top-down block plan at ``(x, z)``."""
    style = STRUCTURE_SCHEMATICS[name]
    radius = float(size) * style.scale
    dark = '#11131A'
    artists = []

    def rectangle(cx, cz, width, height, color, angle=0, order=0, line=0.32):
        return _add(ax, artists, Rectangle(
            (cx - width / 2, cz - height / 2), width, height,
            angle=angle, rotation_point='center', facecolor=color,
            edgecolor=dark, linewidth=line, alpha=alpha,
            zorder=zorder + order,
        ), transform)

    def circle(cx, cz, value, color, order=0, line=0.32):
        return _add(ax, artists, Circle(
            (cx, cz), value, facecolor=color, edgecolor=dark,
            linewidth=line, alpha=alpha, zorder=zorder + order,
        ), transform)

    def polygon(points, color, order=0, line=0.32):
        return _add(ax, artists, Polygon(
            points, closed=True, facecolor=color, edgecolor=dark,
            linewidth=line, alpha=alpha, zorder=zorder + order,
        ), transform)

    if name == 'village':
        rectangle(x, z, 2.0 * radius, 0.22 * radius, '#B99A69')
        rectangle(x, z, 0.22 * radius, 1.65 * radius, '#B99A69')
        for dx, dz, width, height in (
            (-0.48, 0.42, 0.55, 0.42), (0.50, 0.40, 0.62, 0.48),
            (-0.45, -0.42, 0.48, 0.52), (0.48, -0.40, 0.54, 0.40),
        ):
            rectangle(x + dx * radius, z + dz * radius,
                      width * radius, height * radius, style.primary, order=1)
            rectangle(x + dx * radius, z + dz * radius,
                      width * radius * 0.65, height * radius * 0.66,
                      style.secondary, order=2, line=0.20)
    elif name == 'desert_pyramid':
        rectangle(x, z, 1.55 * radius, 1.55 * radius, style.primary)
        for dx, dz in ((-.58, -.58), (-.58, .58), (.58, -.58), (.58, .58)):
            rectangle(x + dx * radius, z + dz * radius,
                      .42 * radius, .42 * radius, style.secondary, order=1)
        rectangle(x, z, .72 * radius, .72 * radius, '#F0D98C', order=2)
        rectangle(x, z, .22 * radius, .22 * radius, '#D05A36', order=3)
    elif name == 'jungle_pyramid':
        for index, factor in enumerate((1.55, 1.12, .72)):
            rectangle(x, z, factor * radius, factor * radius,
                      style.primary if index != 1 else style.secondary,
                      order=index)
        rectangle(x, z - .93 * radius, .34 * radius, .36 * radius,
                  '#4A5E38', order=3)
    elif name == 'swamp_hut':
        for dx, dz in ((-.48, -.38), (.48, -.38), (-.48, .38), (.48, .38)):
            circle(x + dx * radius, z + dz * radius, .09 * radius, '#3E2C25')
        rectangle(x, z, 1.18 * radius, .86 * radius, style.secondary, order=1)
        rectangle(x, z, 1.52 * radius, .98 * radius, style.primary, order=2)
        rectangle(x + .72 * radius, z, .25 * radius, .32 * radius,
                  '#33261F', order=3)
    elif name == 'pillager_outpost':
        rectangle(x, z, 1.25 * radius, 1.25 * radius, style.primary)
        rectangle(x, z, .82 * radius, .82 * radius, style.secondary, order=1)
        for dx, dz in ((-.53, -.53), (.53, -.53), (-.53, .53), (.53, .53)):
            rectangle(x + dx * radius, z + dz * radius,
                      .22 * radius, .22 * radius, '#352820', order=2)
    elif name == 'igloo':
        circle(x, z, .72 * radius, style.primary)
        circle(x, z, .46 * radius, style.secondary, order=1)
        rectangle(x + .74 * radius, z, .58 * radius, .34 * radius,
                  style.primary, order=2)
    elif name == 'woodland_mansion':
        rectangle(x, z + .20 * radius, 1.72 * radius, 1.20 * radius,
                  style.primary)
        rectangle(x - .60 * radius, z - .55 * radius,
                  .52 * radius, .72 * radius, style.primary)
        rectangle(x + .60 * radius, z - .55 * radius,
                  .52 * radius, .72 * radius, style.primary)
        for dx in (-.55, 0, .55):
            rectangle(x + dx * radius, z + .18 * radius,
                      .32 * radius, .58 * radius, style.secondary, order=1)
    elif name == 'ocean_monument':
        polygon([
            (x, z + .92 * radius), (x + .92 * radius, z),
            (x, z - .92 * radius), (x - .92 * radius, z),
        ], style.primary)
        rectangle(x, z, 1.05 * radius, .66 * radius, style.secondary, order=1)
        for dx in (-.48, .48):
            rectangle(x + dx * radius, z, .28 * radius, .72 * radius,
                      '#387F76', order=2)
    elif name == 'shipwreck':
        polygon([
            (x - .95 * radius, z), (x - .48 * radius, z + .48 * radius),
            (x + .72 * radius, z + .32 * radius), (x + .98 * radius, z),
            (x + .72 * radius, z - .32 * radius),
            (x - .48 * radius, z - .48 * radius),
        ], style.primary)
        rectangle(x + .04 * radius, z, 1.20 * radius, .18 * radius,
                  style.secondary, order=1)
        rectangle(x - .15 * radius, z, .14 * radius, .86 * radius,
                  '#D0B27A', order=2)
    elif name == 'ocean_ruin':
        for dx, dz, factor in (
            (-.46, -.34, .52), (.28, -.42, .44), (-.20, .34, .60),
            (.48, .30, .38), (.12, .02, .32),
        ):
            rectangle(x + dx * radius, z + dz * radius,
                      factor * radius, factor * radius,
                      style.primary if factor > .45 else style.secondary)
    elif name == 'ruined_portal':
        rectangle(x - .62 * radius, z, .22 * radius, 1.45 * radius, style.primary)
        rectangle(x + .62 * radius, z, .22 * radius, 1.45 * radius, style.primary)
        rectangle(x, z + .62 * radius, 1.42 * radius, .22 * radius, style.primary)
        rectangle(x, z - .62 * radius, .82 * radius, .20 * radius, style.primary)
        rectangle(x, z, .88 * radius, .88 * radius, style.secondary, order=1)
        rectangle(x + .72 * radius, z - .72 * radius,
                  .35 * radius, .28 * radius, '#6B3A2D', order=2)
    elif name == 'fortress':
        rectangle(x, z, 1.85 * radius, .30 * radius, style.primary)
        rectangle(x, z, .30 * radius, 1.85 * radius, style.primary)
        for dx, dz in ((-.72, 0), (.72, 0), (0, -.72), (0, .72)):
            rectangle(x + dx * radius, z + dz * radius,
                      .46 * radius, .46 * radius, style.secondary, order=1)
    elif name == 'bastion':
        rectangle(x - .32 * radius, z + .18 * radius,
                  1.22 * radius, 1.18 * radius, style.primary)
        rectangle(x + .52 * radius, z - .32 * radius,
                  .78 * radius, .92 * radius, style.secondary, order=1)
        rectangle(x - .38 * radius, z - .62 * radius,
                  .55 * radius, .40 * radius, '#201A1A', order=2)
        rectangle(x + .04 * radius, z + .14 * radius,
                  .32 * radius, .72 * radius, '#C29B60', order=3)
    elif name == 'end_city':
        polygon([
            (x - .92 * radius, z),
            (x - .55 * radius, z + .38 * radius),
            (x + .58 * radius, z + .30 * radius),
            (x + 1.04 * radius, z),
            (x + .58 * radius, z - .30 * radius),
            (x - .55 * radius, z - .38 * radius),
        ], style.primary)
        polygon([
            (x - .52 * radius, z),
            (x - .22 * radius, z + .22 * radius),
            (x + .56 * radius, z + .17 * radius),
            (x + .82 * radius, z),
            (x + .56 * radius, z - .17 * radius),
            (x - .22 * radius, z - .22 * radius),
        ], style.secondary, order=1)
        rectangle(x - .42 * radius, z, .32 * radius, .50 * radius,
                  '#9B5FA0', order=2)
        circle(x + .08 * radius, z, .12 * radius, '#E8D6E7', order=3)
        rectangle(x + .08 * radius, z, .08 * radius, .72 * radius,
                  '#6D4477', order=3, line=0.22)
        polygon([
            (x + .98 * radius, z),
            (x + 1.18 * radius, z + .11 * radius),
            (x + 1.18 * radius, z - .11 * radius),
        ], '#D9C0B5', order=4)
    else:
        raise KeyError(f'No schematic registered for {name}')

    return StructureArtistGroup(artists)
