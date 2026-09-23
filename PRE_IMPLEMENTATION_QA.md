# PRE-IMPLEMENTATION ADVERSARIAL QA

**Audit date:** 2026-09-23  
**Repository:** `IsolatedSingularity/Minecraft-Generation`  
**Reviewed branch:** `main`  
**Reviewed baseline:** `f76465c3691222a71ece02611199d9b029ce1b30`  
**Purpose:** failure discovery and agent handoff before substantial further implementation.

> **Next agent:** read the P0 findings and acceptance criteria below before extending generation logic. Do not treat a green current suite as proof of Java 1.16.1 parity.

## Scope and known-good baseline

The audit inspected the repository tree, root and Viewer READMEs, active Python generation code, Cubiomes/WASM wrapper and worker, viewer smoke tests, workflows, requirements, generated-artifact paths, prior terrain audit, archives/research surfaces, recent history, and the current Bloc Fantome Java-world verification approach.

The baseline itself is internally consistent enough to build and test under the checks that currently exist:

- latest GitHub Actions **Tests** run at the reviewed HEAD passed **33/33** Python tests on Ubuntu / CPython 3.12;
- the Viewer has source-level smoke tests for Java 1.16.1 Cubiomes/WASM behavior;
- the signed 64-bit seed path in Seed Atlas is sound: decimal input is parsed with `BigInt`, carried as text, split to two 32-bit words, and reconstructed as 64 bits in WASM;
- Seed Atlas discards stale structure responses by request token, and changing seed/dimension terminates and recreates workers.

Those positives do **not** establish exact chunk generation or end-to-end viewer correctness.

No implementation code was changed by this audit.

## Severity summary

### P0: resolve before substantial further generation work

#### P0-1. Python random-spread structure placement is wrong for 10 of 11 Overworld families

**Status:** numerically reproduced from the current Python algorithm and statically proven against the repository's vendored Cubiomes implementation.

`Code/core/structures.py` defaults `StructureConfig.uniform = False`, where `False` means the code averages two bounded RNG draws per axis, i.e. triangular placement. Only `OCEAN_MONUMENT` sets `uniform=True`.

Vendored Cubiomes states and implements the opposite vanilla contract for Java 1.16.1:

- normal feature structures such as villages, temples, huts, igloos, shipwrecks, ocean ruins, ruined portals, and outposts use **uniform** `getFeaturePos`;
- monuments and mansions use **triangular** `getLargeStructurePos`;
- End cities use the large/triangular path.

The Python implementation therefore has the wrong distribution for every small Overworld family and for ocean monuments. Woodland mansions and End cities happen to use the correct triangular rule. Nether ruined portals are also affected; the custom fortress/bastion shared path is separate and uses the correct uniform offsets.

There is a direct internal contradiction at seed 42, region (0,0):

- current Python test expects village chunk offset **(1,20)**;
- the actual Java RNG bounded draws are `1, 1, 22, 19`;
- uniform vanilla placement therefore selects **(1,1)**;
- `Viewer/tests/mc1161-smoke.c`, through Cubiomes, already expects that same village at block origin **(16,16)**;
- current Python monument test expects **(8,16)**, but Cubiomes large-structure placement averages the four draws and yields chunk offset **(12,12)**.

**Why it matters:** active plots and analyses call this helper. `structure_placement.py`, `minecraftStructureAnalysis.py`, the outpost village-exclusion gate, and Nether ruined-portal visualization inherit the error. The unit suite currently locks the wrong values in place.

**Smallest acceptance criterion:** create a table-driven differential test against Cubiomes or an official Java oracle for every supported structure family across positive and negative regions and multiple seeds. At minimum, seed 42 region (0,0) must give village chunk (1,1) and monument chunk (12,12). Do not update golden plots until the placement oracle passes.

**Timing:** fix now, test-first.

#### P0-2. There is no single parity contract joining Python, Cubiomes/WASM, and final Java chunks

**Status:** architectural risk with one already-proven divergence, P0-1.

The project has three materially different generation surfaces:

1. bespoke Python ports used by README plots and analyses;
2. vendored Cubiomes C/WASM used by Seed Atlas;
3. procedural/showcase structure assembly used by the 3D viewer.

They are not kept aligned by a shared differential oracle. P0-1 demonstrates the practical consequence: two repository-local implementations can both be called exact while returning different coordinates.

The current Seed Atlas accuracy boundary is also explicitly below final chunks. Overworld relief is approximate surface shading, the Nether is a density/cave-floor view, the End is density-derived, and structure markers generally stop at candidate/viability semantics. These are legitimate visualizations when labeled as such, but they are not a final block-state simulator.

**Bloc Fantome comparison:** its newer world-map pipeline now has the missing kind of ground truth. It reads official Java 1.16.1 Anvil chunks, rejects wrong DataVersion or incomplete chunks, retains source block states, hashes chunk evidence, and independently re-reads source chunks to verify exported cells. Its current QA record reports 3,567,537 retained cells across 17 captures with zero source-state mismatches, while separately accounting for authored decoration. Bloc Fantome also clearly labels artistic terrain slices separately from imported-Java exactness.

**Why it matters:** extending approximate generation before defining the parity boundary will deepen the split and make later correction more expensive.

**Smallest acceptance criterion:** before adding another supposedly exact generation layer, establish a differential gate that can answer, for selected seeds/chunks, which outputs are:
- exact source block/state data;
- exact candidate/biome/base-density data;
- intentionally modeled presentation data.

Use the working Bloc Fantome Anvil import/verification approach as reference evidence rather than re-inventing another approximation.

**Timing:** resolve the accuracy contract before the next substantial generation phase. This does not require rewriting the viewer first.

### P1: concrete bugs and high-value adversarial tests

#### P1-1. Seed Atlas structure queries can exceed their fixed result buffer inside the UI's allowed viewport

**Status:** strong static/control-flow bug.

`Viewer/src/seed-map/main.js` accepts structure queries whose X and Z spans are each up to **140,000 blocks**. The worker allocates exactly **32,768** `StructureHit` records.

One 32-chunk-spacing family alone has about one candidate per 512 x 512 blocks. A roughly 93,000-block square can therefore exceed 32,768 candidates for only that family. With all Overworld families enabled, the combined expected candidate count crosses 32,768 around a square only in the high-20k-block range, well inside the UI limit.

The C loop does not stop scanning after capacity is exceeded. It keeps counting the full query and returns a negative count afterward, so a large query can both waste CPU and end as `Structure query failed: Structure result capacity exceeded`.

**Smallest acceptance criterion:** select all Overworld structure layers and exercise approximately 25k, 30k, 90k, and 140k square views. No accepted view should overflow, freeze, or silently drop results. Cancellation/rapid-pan behavior must remain responsive.

**Timing:** before relying on large-area structure browsing.

#### P1-2. Production WASM has a source/generated-binary CI blind spot

**Status:** statically proven workflow gap.

Production Seed Atlas loads committed generated `Viewer/src/seed-map/generated/mc1161.wasm`. The repository contains:
- `Viewer/cubiomes/mc1161_wasm.c`;
- `Viewer/scripts/build-wasm.ps1`;
- `Viewer/scripts/test-wasm.ps1`;
- a useful `Viewer/tests/mc1161-smoke.c` test.

However, the Pages workflow does **not** run `npm run test:wasm`, does not rebuild WASM, and does not compare the committed generated binary against source. The Viewer workflow also runs on pushes to `main`, not pull requests. A PR can therefore pass the Python workflow while breaking Viewer behavior, and wrapper/vendor source can drift from the production binary.

The C smoke test is materially valuable: it checks known Overworld biome points, a seed-42 village position, all 128 strongholds, Nether biome/surface behavior, and multi-scale Nether/End sampling. It is simply outside required CI.

**Smallest acceptance criterion:** on PRs that touch Viewer/Cubiomes/generated WASM, run the relevant Viewer smoke suite and a reproducible WASM source/binary test or rebuild check.

**Timing:** fix before the next Cubiomes/WASM change.

#### P1-3. Current Python tests contain false-confidence checks

**Status:** reproduced from test source.

Important examples:

- `test_expanded_structure_catalog_and_distribution_examples` hard-codes the incorrect village/monument placement values from P0-1, so the green suite actively protects wrong behavior.
- `test_region_and_population_seed_are_deterministic` checks the population seed by comparing one call with an identical second call. That proves repeatability, not correctness.
- the original-JAR biome/terrain oracle regression currently covers only a handful of fixed points: five Overworld biome samples, four Nether biome samples, and four Overworld heights.
- Python stronghold regression proves ring counts but not stronghold coordinates against a Java/Cubiomes oracle.
- chunk-status tests verify the repository's modeled presentation snapshots, not Minecraft's real asynchronous generation scheduler.

**Smallest acceptance criterion:** replace self-comparisons and implementation-derived expected values with independent oracle vectors. Add negative coordinates, region boundaries, large coordinates, signed-long seed edges, and randomized differential samples.

**Timing:** before correcting P0-1 so the tests become a safety net rather than a blocker.

#### P1-4. Generated plots can be stale while CI remains green

**Status:** strong static workflow gap.

`Code/render_all.py` is manual. The test workflow does not regenerate the README visualizations. Asset tests validate dimensions, decodability, frame behavior, and references of already-committed files, but not whether those files came from current source.

P0-1 therefore can coexist with green tests and apparently valid structure GIFs.

**Smallest acceptance criterion:** have a cheap provenance/staleness check for source-derived golden plots, or explicitly mark which generated artifacts are not CI-verified. At least regenerate affected structure artifacts after the corrected placement oracle passes.

**Timing:** before publishing/relying on regenerated scientific visuals.

#### P1-5. Seed Atlas worker crashes can leave requests permanently pending

**Status:** strong static/control-flow bug; runtime trigger still needs browser reproduction.

`resetWorker()` installs `worker.onmessage` but no `worker.onerror` or `worker.onmessageerror` handler. Each `callWorker()` stores a promise in the global `pending` map and only removes it when a normal message arrives or when the user later changes seed/dimension and `resetWorker()` rejects everything.

If a worker dies because of a WASM trap, allocation failure, browser worker failure, or malformed transferable/message state, its outstanding tile/biome/structure request has no completion path. The UI can therefore retain a loading tile/status indefinitely, leak the pending entry, and keep dispatching later work to a dead worker.

**Smallest acceptance criterion:** deliberately terminate or throw inside one worker while tile, biome hover, and structure requests are outstanding. Every promise must reject promptly, the dead worker must be removed/replaced or the pool must fail closed, and later requests must still settle.

**Timing:** fix before treating the worker pool as production-robust, especially alongside P1-1 large-view stress tests.

### P2: cleanup, ambiguity, and deferred risks

#### P2-1. Multiple tracked surfaces look authoritative to an autonomous agent

**Status:** governance/source-of-truth risk.

Current root contains active code beside:
- `archives/` with a historical `AGENT_HANDOFF.md`;
- `.jenova-research/` documents that predate current tests/CI and contain stale claims;
- `Code/minecraftStructureAnalysis_backup.py`;
- older executable scripts such as `Code/minecraftGeneration.py`, which still writes to `~/Downloads`;
- `NEW; Map Mod/`, a recent Xaero research/specification tree that is not the current implementation.

The stale Jenova checklist even says there should be no backup files while the backup remains tracked.

Do not delete historical evidence during the QA pass. Future cleanup should make active versus historical/research status unambiguous.

#### P2-2. Agent handoff conventions are partly gitignored

**Status:** governance risk.

`.gitignore` ignores `AGENTS.md`, `*.agent.md`, `*.instructions.md`, `audit-*.md`, and `OPEN_INACCURACIES.md`. A future agent can create apparently canonical instructions or an audit that never becomes part of the repository.

This file deliberately uses the tracked root name `PRE_IMPLEMENTATION_QA.md`.

#### P2-3. Dependency and platform reproducibility are weak

**Status:** credible maintenance risk.

`requirements.txt` uses lower bounds without a lock or upper bounds. The latest successful CI installed current releases rather than a frozen environment. README says Python 3.10+, but CI covers only Python 3.12 on Ubuntu. Viewer setup/build scripts are heavily PowerShell-oriented, while browser behavior itself has no end-to-end test matrix.

Acceptance: at minimum distinguish the supported runtime range from the single CI runtime, and add another runtime/platform only where current support claims require it.

#### P2-4. `main` has no branch protection / required-check guard

**Status:** repository-governance risk observed at audit baseline.

The default branch was reported as unprotected. This compounds the Viewer PR blind spot because a green or absent check is not enforced before changes land.

#### P2-5. Seed Atlas hash coordinates accept malformed/non-finite values

**Status:** credible hypothesis requiring browser reproduction.

Seed parsing is strict, but hash restoration uses `Number(... ) || 0` for X/Z. Values such as `Infinity` are truthy and can reach initial map view state. Goto input checks finiteness, so restore and interactive input do not share the same validation contract.

Acceptance: malformed, NaN, Infinity, huge finite, and world-edge hash URLs should restore to a bounded deterministic state without exceptions.

#### P2-6. Persisted occlusion-cache identity is only a 32-bit FNV fingerprint

**Status:** low-probability but real cache-collision risk.

`usePacks.js` describes `sourcesIdentity()` as using "full content hashes", but `fnvHash()` is a single 32-bit FNV-1a value plus byte length. That key is used by `useBuild.js` to load persisted IndexedDB occlusion data. Two same-length pack byte streams with the same 32-bit FNV value can therefore share a persisted cache key even though their model data differ.

This is not a practical concern for ordinary vanilla assets, but it is an avoidable stale/wrong-cache path for arbitrary local or remote packs and contradicts the stronger provenance language in the comment.

**Smallest acceptance criterion:** either use a collision-resistant content identity for persisted cache keys or store enough metadata/content verification to reject mismatched cache entries. A deliberately constructed same-length hash collision must not import the other pack's occlusion data.

**Timing:** defer unless persistent custom-pack caching is important in the next phase; do not describe the current key as a full content hash.

#### P2-7. Malformed NBT lengths are not defensively bounded before allocation/iteration

**Status:** credible local-input robustness risk; not reproduced here.

`Viewer/src/nbt.js` trusts signed list/array lengths and string lengths while advancing its cursor, allocating arrays, and iterating payloads. Truncated, negative-length, or intentionally huge NBT can therefore fall through to generic `DataView`/array errors or excessive work instead of a bounded parser rejection. The same viewer accepts local JAR/resource-pack inputs, so corrupt files are a realistic failure mode even without hostile intent.

**Smallest acceptance criterion:** fuzz truncated roots, negative/huge list and array lengths, invalid tag IDs, gzip truncation, palette indexes outside range, and structures with absurd sizes. Fail quickly with a controlled parse error and no persistent state mutation.

**Timing:** deferred unless the next phase expands arbitrary file import, but include in viewer hardening.

#### P2-8. Seed Atlas map extent exceeds Java 1.16.1's maximum world-border radius

**Status:** statically proven UI/accuracy-boundary mismatch.

`Viewer/src/seed-map/main.js` sets `WORLD_LIMIT = 33_554_432` and uses that as the OpenLayers extent. Java 1.16.1's `WorldBorder` has a dedicated maximum-world-border-radius contract, and the vanilla maximum/default hard radius is 29,999,984 blocks. The 33,554,432 value is instead a familiar noise/precision scale in this codebase, not the vanilla playable world-border limit.

This means Seed Atlas can present several million blocks of coordinates beyond the normal Java 1.16.1 world-border domain without explaining that it has switched from "world map" semantics to extrapolated generator/math semantics.

Reference: Fabric Yarn 1.16.1 `WorldBorder` API and Mojang issue MC-70076.

**Smallest acceptance criterion:** either clamp ordinary map navigation/querying to the Java 1.16.1 world-border radius, or explicitly expose an "extended mathematical sampling" mode that labels coordinates outside it. Boundary tests should cover exactly ±29,999,984 and the first coordinate beyond the accepted range.

**Timing:** deferred UI/accuracy cleanup unless world-edge behavior is part of the next phase.

#### P2-9. Showcase randomness must remain quarantined from future parity work

**Status:** architectural guardrail, not a current defect by itself.

The 3D Viewer legitimately contains `Math.random()` defaults in loot/procedural/showcase paths. Existing documentation already distinguishes showcase assembly from world-seed-exact natural generation. Future agents must not reuse those paths as if they were Java world RNG.

Acceptance: any feature promoted to "world-seed exact" must have explicit deterministic seed provenance and an independent oracle.

## Adversarial test matrix for the next agent

| Area | Minimal hostile cases | Required evidence |
|---|---|---|
| Structure placement | all supported families; seeds 0, 42, -1 and signed-long edges; positive/negative regions; spacing boundaries | Python vs Cubiomes/Java coordinate equality |
| Outpost gates | village exclusion just inside/outside 10 chunks; negative coordinates | Java/Cubiomes oracle |
| Strongholds | all 128 for several seeds | exact coordinates, not only ring counts |
| Seed transport | 0, -1, ±2^53 neighborhood, Long.MIN/MAX, overflow, malformed decimal | same 64-bit seed reaches WASM |
| Structure viewport | all-layer 25k/30k/90k/140k views; rapid pan/cancel | no overflow, stale results, or long uninterruptible scan |
| Worker state | seed/dimension change while terrain and structures are in flight; repeated toggles | old requests cannot paint new state |
| WASM provenance | rebuild from source; execute C/WASM smoke vectors | production binary matches tested source |
| Java chunk parity | selected official 1.16.1 Anvil chunks, missing/corrupt/partial/wrong DataVersion cases | block/state/biome/structure comparison and explicit rejection |
| Terrain claims | biome/base density vs final block state | every output labeled by fidelity tier |
| Generated plots | clean regeneration after source changes | affected artifacts derive from current code |
| Browser restore | malformed hash, huge coords, refresh/back/forward | deterministic bounded UI state |
| File/tool paths | run from documented working directories on supported OSes | no silent CWD/private-path dependency |
| Dependencies | supported Python versions plus clean npm install/build | reproducible documented environment |
| Showcase RNG | reload/retry and seeded procedural inputs | no accidental claim of world-seed parity |

## What remains unverified

The audit did **not** claim a full runtime reproduction of the browser UI. The current environment could inspect GitHub source/history/Actions but could not clone the repository into a network-isolated local runner. The latest green GitHub Actions run was inspected instead.

Also unverified here:

- the private `Game Reference/` and `.oracle-bin/` corpus;
- a fresh official Java 1.16.1 server generation run;
- the current committed WASM binary rebuilt byte-for-byte from wrapper/vendor source;
- real-browser performance/memory behavior for very large Seed Atlas views;
- malformed/corrupt remote JAR/ZIP/NBT handling in the full Viewer;
- accessibility and keyboard/screen-reader behavior;
- Windows/macOS runtime coverage;
- exact final-chunk parity beyond the limited existing oracle vectors.

## CI relevance verdict

Current CI is useful for regression of the repository's existing mathematical models, asset invariants, and selected Viewer build/smoke surfaces. It is **not** a reliable gate for the risks found here.

Most importantly, the Python suite is green while protecting an incorrect structure-placement contract, and required CI does not execute the repository's strongest Cubiomes/WASM smoke test or any final-Java-chunk differential test.

Treat the suite as a baseline regression layer, not as evidence that Minecraft Generation is currently exact.
