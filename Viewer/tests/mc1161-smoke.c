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

    puts("PASS: Cubiomes Java 1.16.1 viewer smoke checks");
    return 0;
}
