# Next Steps

## Immediate

1. Run `tools/Find-XaeroJars.ps1`.
2. Acquire modern common Xaero 1.44.2 with `Acquire-XaeroResearchInputs.ps1`.
3. Run full inspection on the modern JAR.
4. If available, inspect Xaero 1.10.1 / Minecraft 1.16.1 too.
5. Generate `comparison-modern-vs-1.16.1.md`.

## Then send back

The most valuable files to send back for a second-pass reconstruction are:

```text
outputs/inspection-modern/xaero_classes.txt
outputs/inspection-modern/class_methods.jsonl
outputs/inspection-modern/class_strings.jsonl

and either:
outputs/inspection-modern/javap/xaero.map.MapWriter.txt
outputs/inspection-modern/javap/xaero.map.region.MapPixel.txt

or the corresponding locally decompiled methods.
```

With those, the remaining inferred parts can be replaced by concrete call graphs and formulas.
