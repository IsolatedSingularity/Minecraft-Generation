# Cubiomes Java 1.16.1 seed layer

Use upstream `Cubitect/cubiomes` and select `MC_1_16_1` explicitly. Do not use the generic `MC_1_16` alias when the target is speedrun-era Java 1.16.1.

## Build the probe

From an upstream Cubiomes checkout:

```bash
make
cc -O3 -std=c11 -fwrapv -I. /path/to/mc1161_seed_probe.c libcubiomes.a -lm -o mc1161_seed_probe
```

Then:

```bash
./mc1161_seed_probe 6090144754301628691 -5000 -5000 5000 5000 > structures.jsonl
```

The output separates a structure attempt from its Cubiomes biome viability. Treat terrain-sensitive structures as candidates until Minecraft itself confirms the chunk.

For a production viewer, expose the same APIs through either a tiny local HTTP/native service or compile Cubiomes to WASM as ezseed does. Keep this calculation layer independent from the 2D map UI.
