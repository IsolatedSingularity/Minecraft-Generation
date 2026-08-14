#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cubiomes/mc1161_wasm.c"

#define CHECK(condition, message) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s\n", message); \
        return 1; \
    } \
} while (0)

static int contains_hit(
    const StructureHit *hits,
    int count,
    int type,
    int x,
    int z
)
{
    for (int index = 0; index < count; index++) {
        if (hits[index].type == type && hits[index].x == x && hits[index].z == z)
            return 1;
    }
    return 0;
}

int main(void)
{
    CHECK(mc_structure_stride() == (int)sizeof(StructureHit), "structure ABI stride");

    uint8_t colors[256][3];
    CHECK(mc_biome_colors(&colors[0][0]) == 0, "Cubiomes biome palette export");
    CHECK(
        colors[ocean][0] == 0x00 && colors[ocean][1] == 0x00 && colors[ocean][2] == 0x70,
        "Cubiomes dark ocean palette"
    );
    CHECK(
        colors[deep_ocean][0] == 0x00 && colors[deep_ocean][1] == 0x00 && colors[deep_ocean][2] == 0x30,
        "Cubiomes deep-ocean palette"
    );

    ViewerContext *overworld = mc_create(0, 42, DIM_OVERWORLD);
    CHECK(overworld != NULL, "create Overworld context");

    const int coordinates[][2] = {
        { 0, 0 }, { 100, 100 }, { -100, 40 }, { 1000, -700 }, { -128, 0 }
    };
    const int expected[] = { 12, 34, 12, 3, 30 };
    for (int index = 0; index < 5; index++) {
        int32_t biome = -1;
        CHECK(
            mc_biome_tile(
                overworld,
                4,
                coordinates[index][0],
                coordinates[index][1],
                1,
                1,
                64,
                &biome
            ) == 0,
            "generate Overworld biome sample"
        );
        if (biome != expected[index]) {
            fprintf(
                stderr,
                "FAIL: Java 1.16.1 Overworld biome oracle at (%d, %d): expected %d, got %d\n",
                coordinates[index][0],
                coordinates[index][1],
                expected[index],
                biome
            );
            return 1;
        }
    }

    StructureHit hits[512];
    int village_count = mc_structures(
        overworld,
        0,
        0,
        511,
        511,
        1u << VIEW_VILLAGE,
        hits,
        512
    );
    CHECK(village_count >= 1, "seed 42 village candidate query");
    if (!contains_hit(hits, village_count, VIEW_VILLAGE, 16, 16)) {
        fprintf(stderr, "FAIL: seed 42 region 0 village candidate; got");
        for (int index = 0; index < village_count; index++)
            fprintf(stderr, " (%d,%d)", hits[index].x, hits[index].z);
        fputc('\n', stderr);
        return 1;
    }

    StructureHit repeat[512];
    int repeat_count = mc_structures(
        overworld,
        0,
        0,
        511,
        511,
        1u << VIEW_VILLAGE,
        repeat,
        512
    );
    CHECK(repeat_count == village_count, "repeat structure count");
    CHECK(
        memcmp(hits, repeat, (size_t)village_count * sizeof(StructureHit)) == 0,
        "deterministic structure results"
    );

    int stronghold_count = mc_structures(
        overworld,
        -40000,
        -40000,
        40000,
        40000,
        1u << VIEW_STRONGHOLD,
        hits,
        512
    );
    CHECK(stronghold_count == 128, "all 128 Java 1.16.1 strongholds");
    mc_destroy(overworld);

    ViewerContext *nether = mc_create(0, 42, DIM_NETHER);
    CHECK(nether != NULL, "create Nether context");
    int32_t nether_biome = -1;
    CHECK(
        mc_biome_tile(nether, 1, 0, 0, 1, 1, 64, &nether_biome) == 0,
        "generate Nether biome sample"
    );
    CHECK(nether_biome == 171, "Java 1.16.1 Nether biome oracle");

    float nether_heights[256];
    CHECK(
        mc_height_tile(nether, 1, 0, 0, 16, 16, nether_heights) == 0,
        "generate Nether navigable-surface sample"
    );
    float nether_min = 128, nether_max = -1;
    for (int index = 0; index < 256; index++) {
        if (nether_heights[index] < nether_min) nether_min = nether_heights[index];
        if (nether_heights[index] > nether_max) nether_max = nether_heights[index];
    }
    CHECK(nether_min >= 31 && nether_max <= 122, "Nether surface below roof and above lava");
    CHECK(nether_min < nether_max, "Nether density surface has local relief");
    float nether_overview[16];
    CHECK(
        mc_height_tile(nether, 16, -8, -8, 4, 4, nether_overview) == 0,
        "generate Nether 1:16 overview relief"
    );
    float nether_overview_exact = -1;
    CHECK(mc_height_tile(nether, 1, -128, -128, 1, 1, &nether_overview_exact) == 0, "generate exact Nether overview point");
    CHECK(fabs(nether_overview[0] - nether_overview_exact) < 0.001f, "Nether 1:16 sample matches its exact block point");

    int nether_count = mc_structures(
        nether,
        0,
        0,
        431,
        431,
        (1u << VIEW_FORTRESS) | (1u << VIEW_BASTION),
        hits,
        512
    );
    CHECK(nether_count >= 1, "seed 42 Nether candidate query");
    if (!contains_hit(hits, nether_count, VIEW_BASTION, 0, 0)) {
        fprintf(stderr, "FAIL: seed 42 region 0 bastion candidate; got");
        for (int index = 0; index < nether_count; index++)
            fprintf(stderr, " type=%d(%d,%d)", hits[index].type, hits[index].x, hits[index].z);
        fputc('\n', stderr);
        return 1;
    }
    mc_destroy(nether);

    ViewerContext *end = mc_create(0, 42, DIM_END);
    CHECK(end != NULL, "create End context");
    float end_center = -1, end_gap = -1;
    CHECK(mc_height_tile(end, 1, 0, 0, 1, 1, &end_center) == 0, "generate End center height");
    CHECK(mc_height_tile(end, 1, 500, 0, 1, 1, &end_gap) == 0, "generate End gap height");
    CHECK(end_center > 0, "End central island density at origin");
    CHECK(end_gap <= 0, "End void gap before outer islands");
    float end_overview[16];
    CHECK(
        mc_height_tile(end, 16, -2, -2, 4, 4, end_overview) == 0,
        "generate End 1:16 overview surface"
    );
    float end_center_exact = -1;
    CHECK(mc_height_tile(end, 1, 8, 8, 1, 1, &end_center_exact) == 0, "generate exact End overview center");
    CHECK(fabs(end_overview[10] - end_center_exact) < 0.001f, "End 1:16 sample matches its exact block center");
    mc_destroy(end);

    puts("PASS: Cubiomes Java 1.16.1 viewer smoke checks");
    return 0;
}
