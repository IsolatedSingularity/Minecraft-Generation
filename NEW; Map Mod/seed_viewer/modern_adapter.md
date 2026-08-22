# Modern Minecraft Adapter Notes

Modern versions increase the complexity of the world source, but should not change the renderer API.

Adapter responsibilities:

- min/max build Y;
- section/paletted block-state access;
- 3D biome access;
- modern dimension types;
- custom dimensions;
- data-driven world generation;
- modern block models/textures;
- resource-pack color changes;
- biome color resolvers.

Do not hard-code Minecraft 1.21.x concepts into `ColumnPixel`.

A modern adapter can expose richer inputs, but the normalized output should remain stable.
