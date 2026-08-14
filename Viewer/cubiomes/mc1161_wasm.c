#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "generator.h"
#include "finders.h"
#include "util.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define EXPORTED EMSCRIPTEN_KEEPALIVE
#else
#define EXPORTED
#endif

typedef struct {
    Generator generator;
    SurfaceNoise surface;
    uint64_t seed;
    int dimension;
} ViewerContext;

typedef struct {
    int32_t type;
    int32_t x;
    int32_t z;
    int32_t viable;
    int32_t terrain_sensitive;
} StructureHit;

typedef struct {
    int cubiomes_type;
    int viewer_type;
    int dimension;
} StructureEntry;

enum ViewerStructureType {
    VIEW_VILLAGE = 0,
    VIEW_DESERT_PYRAMID,
    VIEW_JUNGLE_TEMPLE,
    VIEW_SWAMP_HUT,
    VIEW_IGLOO,
    VIEW_OCEAN_RUIN,
    VIEW_SHIPWRECK,
    VIEW_MONUMENT,
    VIEW_MANSION,
    VIEW_OUTPOST,
    VIEW_RUINED_PORTAL,
    VIEW_FORTRESS,
    VIEW_BASTION,
    VIEW_END_CITY,
    VIEW_STRONGHOLD
};

static const StructureEntry STRUCTURES[] = {
    { Village,          VIEW_VILLAGE,          DIM_OVERWORLD },
    { Desert_Pyramid,   VIEW_DESERT_PYRAMID,   DIM_OVERWORLD },
    { Jungle_Temple,    VIEW_JUNGLE_TEMPLE,    DIM_OVERWORLD },
    { Swamp_Hut,        VIEW_SWAMP_HUT,        DIM_OVERWORLD },
    { Igloo,            VIEW_IGLOO,            DIM_OVERWORLD },
    { Ocean_Ruin,       VIEW_OCEAN_RUIN,       DIM_OVERWORLD },
    { Shipwreck,        VIEW_SHIPWRECK,        DIM_OVERWORLD },
    { Monument,         VIEW_MONUMENT,         DIM_OVERWORLD },
    { Mansion,          VIEW_MANSION,          DIM_OVERWORLD },
    { Outpost,          VIEW_OUTPOST,          DIM_OVERWORLD },
    { Ruined_Portal,    VIEW_RUINED_PORTAL,    DIM_OVERWORLD },
    { Ruined_Portal_N,  VIEW_RUINED_PORTAL,    DIM_NETHER },
    { Fortress,         VIEW_FORTRESS,          DIM_NETHER },
    { Bastion,          VIEW_BASTION,           DIM_NETHER },
    { End_City,         VIEW_END_CITY,          DIM_END }
};

static int floor_div(int value, int divisor)
{
    int quotient = value / divisor;
    int remainder = value % divisor;
    if (remainder < 0) quotient--;
    return quotient;
}

static int in_box(Pos position, int min_x, int min_z, int max_x, int max_z)
{
    return position.x >= min_x && position.x <= max_x
        && position.z >= min_z && position.z <= max_z;
}

EXPORTED ViewerContext *mc_create(uint32_t seed_high, uint32_t seed_low, int dimension)
{
    ViewerContext *context = (ViewerContext *)calloc(1, sizeof(ViewerContext));
    if (!context) return NULL;
    context->seed = ((uint64_t)seed_high << 32) | (uint64_t)seed_low;
    context->dimension = dimension;
    setupGenerator(&context->generator, MC_1_16_1, 0);
    applySeed(&context->generator, dimension, context->seed);
    if (dimension != DIM_NETHER)
        initSurfaceNoise(&context->surface, dimension, context->seed);
    return context;
}

EXPORTED void mc_destroy(ViewerContext *context)
{
    free(context);
}

EXPORTED int mc_biome_colors(uint8_t *output)
{
    if (!output) return 1;
    initBiomeColors((unsigned char (*)[3])output);
    return 0;
}

EXPORTED int mc_biome_tile(
    ViewerContext *context,
    int scale,
    int sample_x,
    int sample_z,
    int width,
    int height,
    int sample_y,
    int32_t *output)
{
    if (!context || !output || width <= 0 || height <= 0) return 1;
    Range range = { scale, sample_x, sample_z, width, height, sample_y, 1 };
    int *cache = allocCache(&context->generator, range);
    if (!cache) return 2;
    int result = genBiomes(&context->generator, cache, range);
    if (!result)
        memcpy(output, cache, sizeof(int32_t) * (size_t)width * (size_t)height);
    free(cache);
    return result;
}

EXPORTED int mc_height_tile(
    ViewerContext *context,
    int sample_x4,
    int sample_z4,
    int width,
    int height,
    float *output)
{
    if (!context || !output || width <= 0 || height <= 0) return 1;
    if (context->dimension == DIM_NETHER) {
        for (int i = 0; i < width * height; i++) output[i] = 127.0f;
        return 0;
    }
    return mapApproxHeight(
        output,
        NULL,
        &context->generator,
        &context->surface,
        sample_x4,
        sample_z4,
        width,
        height
    );
}

static int append_hit(
    StructureHit *output,
    int capacity,
    int count,
    int type,
    Pos position,
    int viable,
    int terrain_sensitive)
{
    if (count < capacity) {
        output[count].type = type;
        output[count].x = position.x;
        output[count].z = position.z;
        output[count].viable = viable;
        output[count].terrain_sensitive = terrain_sensitive;
    }
    return count + 1;
}

EXPORTED int mc_structures(
    ViewerContext *context,
    int min_x,
    int min_z,
    int max_x,
    int max_z,
    uint32_t enabled_mask,
    StructureHit *output,
    int capacity)
{
    if (!context || !output || capacity <= 0) return -1;
    if (min_x > max_x) { int value = min_x; min_x = max_x; max_x = value; }
    if (min_z > max_z) { int value = min_z; min_z = max_z; max_z = value; }

    int count = 0;
    const size_t total = sizeof(STRUCTURES) / sizeof(STRUCTURES[0]);
    for (size_t index = 0; index < total; index++) {
        const StructureEntry entry = STRUCTURES[index];
        if (entry.dimension != context->dimension) continue;
        if (!(enabled_mask & (1u << entry.viewer_type))) continue;

        StructureConfig config;
        if (!getStructureConfig(entry.cubiomes_type, MC_1_16_1, &config)) continue;
        const int min_chunk_x = floor_div(min_x, 16);
        const int max_chunk_x = floor_div(max_x, 16);
        const int min_chunk_z = floor_div(min_z, 16);
        const int max_chunk_z = floor_div(max_z, 16);
        const int min_region_x = floor_div(min_chunk_x, config.regionSize);
        const int max_region_x = floor_div(max_chunk_x, config.regionSize);
        const int min_region_z = floor_div(min_chunk_z, config.regionSize);
        const int max_region_z = floor_div(max_chunk_z, config.regionSize);

        for (int region_x = min_region_x; region_x <= max_region_x; region_x++) {
            for (int region_z = min_region_z; region_z <= max_region_z; region_z++) {
                Pos position;
                if (!getStructurePos(
                    entry.cubiomes_type,
                    MC_1_16_1,
                    context->seed,
                    region_x,
                    region_z,
                    &position
                )) continue;
                if (!in_box(position, min_x, min_z, max_x, max_z)) continue;
                const int viable = isViableStructurePos(
                    entry.cubiomes_type,
                    &context->generator,
                    position.x,
                    position.z,
                    0
                );
                count = append_hit(
                    output,
                    capacity,
                    count,
                    entry.viewer_type,
                    position,
                    viable,
                    entry.cubiomes_type == End_City
                );
            }
        }
    }

    if (
        context->dimension == DIM_OVERWORLD
        && (enabled_mask & (1u << VIEW_STRONGHOLD))
    ) {
        StrongholdIter iterator;
        initFirstStronghold(&iterator, MC_1_16_1, context->seed);
        for (int index = 0; index < 128; index++) {
            nextStronghold(&iterator, &context->generator);
            if (in_box(iterator.pos, min_x, min_z, max_x, max_z)) {
                count = append_hit(
                    output,
                    capacity,
                    count,
                    VIEW_STRONGHOLD,
                    iterator.pos,
                    1,
                    0
                );
            }
        }
    }

    return count <= capacity ? count : -count;
}

EXPORTED int mc_structure_stride(void)
{
    return (int)sizeof(StructureHit);
}
