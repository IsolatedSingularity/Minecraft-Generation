/*
 * Minimal Java 1.16.1 structure-position probe using Cubiomes.
 *
 * Build (from a Cubiomes checkout, adjust paths as needed):
 *   make
 *   cc -O3 -std=c11 -fwrapv -I. /path/to/mc1161_seed_probe.c libcubiomes.a -lm -o mc1161_seed_probe
 *
 * Usage:
 *   ./mc1161_seed_probe <signed-world-seed> <min-block-x> <min-block-z> <max-block-x> <max-block-z>
 *
 * Output: JSON Lines. "viable" is Cubiomes' biome/structure viability result.
 * It is NOT a claim that every final block of the natural structure is known.
 * Terrain-dependent cases should be confirmed with an actual 1.16.1-generated
 * chunk when block-level exactness matters.
 */

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "generator.h"
#include "finders.h"

static int floor_div(int a, int b)
{
    int q = a / b;
    int r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) q--;
    return q;
}

static int parse_i32(const char *s, int *out)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end || v < INT32_MIN || v > INT32_MAX) return 0;
    *out = (int)v;
    return 1;
}

static int parse_seed(const char *s, uint64_t *out)
{
    char *end = NULL;
    errno = 0;
    int64_t v = strtoll(s, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)v; /* preserve Java signed-long bit pattern */
    return 1;
}

typedef struct {
    int type;
    const char *name;
} NamedStructure;

static const NamedStructure STRUCTS[] = {
    { Desert_Pyramid, "desert_pyramid" },
    { Jungle_Temple,  "jungle_temple" },
    { Swamp_Hut,      "swamp_hut" },
    { Igloo,          "igloo" },
    { Village,        "village" },
    { Ocean_Ruin,     "ocean_ruin" },
    { Shipwreck,      "shipwreck" },
    { Monument,       "monument" },
    { Mansion,        "mansion" },
    { Outpost,        "outpost" },
    { Ruined_Portal,  "ruined_portal" },
    { Ruined_Portal_N,"ruined_portal_nether" },
    { Treasure,       "buried_treasure" },
    { Fortress,       "fortress" },
    { Bastion,        "bastion" },
    { End_City,       "end_city" }
};

static int in_box(Pos p, int minx, int minz, int maxx, int maxz)
{
    return p.x >= minx && p.x <= maxx && p.z >= minz && p.z <= maxz;
}

int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr, "usage: %s <seed> <minX> <minZ> <maxX> <maxZ>\n", argv[0]);
        return 2;
    }

    uint64_t seed;
    int minx, minz, maxx, maxz;
    if (!parse_seed(argv[1], &seed) ||
        !parse_i32(argv[2], &minx) || !parse_i32(argv[3], &minz) ||
        !parse_i32(argv[4], &maxx) || !parse_i32(argv[5], &maxz)) {
        fprintf(stderr, "invalid numeric argument\n");
        return 2;
    }
    if (minx > maxx) { int t = minx; minx = maxx; maxx = t; }
    if (minz > maxz) { int t = minz; minz = maxz; maxz = t; }

    Generator g;
    int currentDim = DIM_UNDEF;

    const size_t nstructs = sizeof(STRUCTS) / sizeof(STRUCTS[0]);
    for (size_t i = 0; i < nstructs; i++) {
        StructureConfig cfg;
        if (!getStructureConfig(STRUCTS[i].type, MC_1_16_1, &cfg)) continue;

        if (currentDim != cfg.dim) {
            setupGenerator(&g, MC_1_16_1, 0);
            applySeed(&g, cfg.dim, seed);
            currentDim = cfg.dim;
        }

        const int minChunkX = floor_div(minx, 16);
        const int maxChunkX = floor_div(maxx, 16);
        const int minChunkZ = floor_div(minz, 16);
        const int maxChunkZ = floor_div(maxz, 16);
        const int minRegX = floor_div(minChunkX, cfg.regionSize);
        const int maxRegX = floor_div(maxChunkX, cfg.regionSize);
        const int minRegZ = floor_div(minChunkZ, cfg.regionSize);
        const int maxRegZ = floor_div(maxChunkZ, cfg.regionSize);

        for (int rx = minRegX; rx <= maxRegX; rx++) {
            for (int rz = minRegZ; rz <= maxRegZ; rz++) {
                Pos p;
                if (!getStructurePos(STRUCTS[i].type, MC_1_16_1, seed, rx, rz, &p)) continue;
                if (!in_box(p, minx, minz, maxx, maxz)) continue;
                int viable = isViableStructurePos(STRUCTS[i].type, &g, p.x, p.z, 0);
                int terrainSensitive = (STRUCTS[i].type == End_City);
                printf("{\"kind\":\"structure\",\"type\":\"%s\",\"dimension\":%d,\"x\":%d,\"z\":%d,\"viable\":%s,\"terrain_sensitive\":%s}\n",
                    STRUCTS[i].name, cfg.dim, p.x, p.z,
                    viable ? "true" : "false",
                    terrainSensitive ? "true" : "false");
            }
        }
    }

    /* Strongholds use their own ring/biome-search algorithm rather than the
       generic region-position API. Java 1.16.1 has 128 strongholds. */
    setupGenerator(&g, MC_1_16_1, 0);
    applySeed(&g, DIM_OVERWORLD, seed);
    StrongholdIter sh;
    initFirstStronghold(&sh, MC_1_16_1, seed);
    for (int i = 0; i < 128; i++) {
        nextStronghold(&sh, &g);
        if (in_box(sh.pos, minx, minz, maxx, maxz)) {
            printf("{\"kind\":\"stronghold\",\"index\":%d,\"dimension\":0,\"x\":%d,\"z\":%d,\"viable\":true,\"terrain_sensitive\":false}\n",
                i, sh.pos.x, sh.pos.z);
        }
    }

    /* Mineshafts are intentionally left out of the generic scan because they
       use getMineshafts() over chunk areas and can be numerous. Add them as a
       separate query path if needed, so normal map requests stay bounded. */
    return 0;
}
