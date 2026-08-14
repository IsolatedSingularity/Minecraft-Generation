# MinecraftMaps Structure Viewer: What It Does and Does Not Do

Site: `https://www.minecraftmaps.com/tools/structure-viewer`

## 1. Publicly described implementation

The page explains its structure viewer in unusually direct terms:

1. Extract vanilla structure-template NBT files from the client jar.
2. Browser-fetch/decompress each gzip-compressed NBT file.
3. Read the structure's `size`, `palette`, and `blocks` fields.
4. Skip air-like/structure-void entries.
5. Render each remaining block as a cube.
6. Use map colors/fallback colors for the simple representation.
7. Optionally apply real block textures, with simplified treatment for complicated shapes.

The site emphasizes that a structure template is the **real saved template**, not an artist reconstruction.

## 2. Why it is useful

This is a good reference for:

- discovering and indexing all structure NBT files;
- reporting exact template dimensions;
- palette/block counts;
- raw `.nbt` download/export;
- a deliberately simple renderer that is easy to reproduce.

You can reproduce the core template browser from your 1.16.1 jar in a small amount of code.

## 3. Why it is not enough for your target

It is not a complete natural-structure generator.

A village/bastion/etc. consists of many templates selected/rotated/connected by generation logic. Merely displaying one NBT file cannot answer:

- where the structure starts for a world seed;
- which pieces get selected;
- how jigsaw RNG resolves;
- which processors modify blocks;
- how terrain changes placement;
- what naturally generated blocks actually appear in the final chunk.

The site itself explains that large structures are assembled from many small templates.

## 4. Rendering simplification

Its public explanation says simple full cubes are the base representation. Even when real textures are used, stairs/fences and similar shapes can be represented as full cubes with their base material texture.

That is precisely the kind of approximation you said you no longer want.

For your project, prefer the Ewan/block-model-renderer route, which resolves Minecraft blockstate/model JSON and textures and performs face culling/meshing.

## 5. Best use in your pipeline

Treat MinecraftMaps as an **NBT correctness reference**, not as the target architecture.

A useful acceptance test is:

- choose a vanilla NBT from `1.16.1.jar`;
- your parser must produce the same `size`;
- your palette must contain the same state entries;
- your block list/count must match exactly after explicitly defined air filtering;
- raw NBT hash must remain unchanged if you only view it.

Once that passes, move on to real models and natural assembly.
