# Version Evolution Relevant to Rendering

The goal is not to reproduce every Xaero release. It is to identify behavior that should be treated as modern semantics rather than inherited from a 2020 Minecraft 1.16.1 build.

## Minecraft 1.16.1-era Xaero releases

Known Forge releases include:

- Xaero World Map 1.9.0 for Minecraft 1.16.1, Aug 5 2020
- 1.10.0 for Minecraft 1.16.1, Aug 12 2020
- 1.10.1 for Minecraft 1.16.1, Aug 20 2020

1.9.0 added an option to ignore server heightmaps and changed zoom/chunk-loading behavior.
1.10.0 added the first server-side world-identification feature.
1.10.1 included a sunlight-map fix.

These builds predate much of the present cave-mode architecture. Therefore:

**Do not use the 1.16.1 Xaero JAR as the sole behavioral specification for a modern Xaero-like renderer.**

Use modern behavior as the renderer target, and use Minecraft 1.16.1 only as a world-generation/data adapter.

## Later rendering changes worth preserving conceptually

Public changelogs and compatibility work indicate later Xaero versions improved or changed:

- biome-per-pixel representation;
- biome color resolver support;
- transparent block handling;
- use of actual texture transparency;
- transparent-block height behavior;
- cave-mode and Full-cave handling;
- solid-block detection for automatic cave logic;
- map-data compression;
- lighting behavior and light-level caching.

This is exactly why the version-independent split is useful:

```text
Minecraft version adapter -> semantic columns -> modern rendering policy
```

instead of:

```text
copy 1.16.1 Xaero behavior everywhere
```
