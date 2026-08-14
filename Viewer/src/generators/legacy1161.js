import { rnd, statePicker } from "../transforms.js"

// Small block builder used for structures that vanilla 1.16.1 generates in
// Java rather than loading from NBT. Coordinates below follow the mapped Java
// sources directly; surrounding terrain and downward terrain supports are
// intentionally excluded from the showcase.
function builder() {
  const { palette, stateFor } = statePicker()
  const cells = new Map()
  const key = (x, y, z) => `${x},${y},${z}`
  const put = (name, x, y, z, properties, nbt) => {
    const k = key(x, y, z)
    if (name === "minecraft:air" || name === "minecraft:cave_air") cells.delete(k)
    else cells.set(k, { name, properties, nbt })
  }
  const fill = (name, x0, y0, z0, x1, y1, z1, properties) => {
    for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) for (let x = x0; x <= x1; x++) put(name, x, y, z, properties)
  }
  const outline = (outer, inner, x0, y0, z0, x1, y1, z1, outerProps, innerProps) => {
    for (let y = y0; y <= y1; y++) for (let z = z0; z <= z1; z++) for (let x = x0; x <= x1; x++) {
      const edge = x === x0 || x === x1 || y === y0 || y === y1 || z === z0 || z === z1
      put(edge ? outer : inner, x, y, z, edge ? outerProps : innerProps)
    }
  }
  const finish = (size, anchor) => {
    let minY = Infinity, maxY = -Infinity
    for (const k of cells.keys()) { const y = Number(k.split(",")[1]); minY = Math.min(minY, y); maxY = Math.max(maxY, y) }
    const blocks = [...cells.entries()].map(([k, cell]) => {
      const [x, y, z] = k.split(",").map(Number)
      const out = { state: stateFor(cell.name, cell.properties), pos: [x, y - minY, z] }
      if (cell.nbt) out.nbt = cell.nbt
      return out
    })
    return { structure: { size: [size[0], maxY - minY + 1, size[2]], palette, blocks, entities: [], anchor: anchor ? [anchor[0], anchor[1] - minY, anchor[2]] : undefined }, maxDepth: 1 }
  }
  return { put, fill, outline, finish }
}

const stair = facing => ({ facing, half: "bottom", shape: "straight", waterlogged: "false" })

// Block-for-block geometry port of DesertTempleGenerator.generate (1.16.1).
export async function runDesertPyramid1161() {
  const { put, fill, outline, finish } = builder()
  const S = "minecraft:sandstone", C = "minecraft:cut_sandstone", H = "minecraft:chiseled_sandstone", A = "minecraft:air"
  outline(S, S, 0, -4, 0, 20, 0, 20)
  for (let i = 1; i <= 9; i++) {
    outline(S, S, i, i, i, 20 - i, i, 20 - i)
    outline(A, A, i + 1, i, i + 1, 19 - i, i, 19 - i)
  }
  fill(S, 0, -5, 0, 20, -5, 20)
  outline(S, A, 0, 0, 0, 4, 9, 4); fill(S, 1, 10, 1, 3, 10, 3)
  put("minecraft:sandstone_stairs", 2, 10, 0, stair("north")); put("minecraft:sandstone_stairs", 2, 10, 4, stair("south")); put("minecraft:sandstone_stairs", 0, 10, 2, stair("east")); put("minecraft:sandstone_stairs", 4, 10, 2, stair("west"))
  outline(S, A, 16, 0, 0, 20, 9, 4); fill(S, 17, 10, 1, 19, 10, 3)
  put("minecraft:sandstone_stairs", 18, 10, 0, stair("north")); put("minecraft:sandstone_stairs", 18, 10, 4, stair("south")); put("minecraft:sandstone_stairs", 16, 10, 2, stair("east")); put("minecraft:sandstone_stairs", 20, 10, 2, stair("west"))
  outline(S, A, 8, 0, 0, 12, 4, 4); fill(A, 9, 1, 0, 11, 3, 4)
  for (const [x, y] of [[9,1],[9,2],[9,3],[10,3],[11,3],[11,2],[11,1]]) put(C, x, y, 1)
  outline(S, A, 4, 1, 1, 8, 3, 3); fill(A, 4, 1, 2, 8, 2, 2)
  outline(S, A, 12, 1, 1, 16, 3, 3); fill(A, 12, 1, 2, 16, 2, 2)
  fill(S, 5, 4, 5, 15, 4, 15); fill(A, 9, 4, 9, 11, 4, 11)
  for (const [x, z] of [[8,8],[12,8],[8,12],[12,12]]) fill(C, x, 1, z, x, 3, z)
  fill(S, 1, 1, 5, 4, 4, 11); fill(S, 16, 1, 5, 19, 4, 11)
  fill(S, 6, 7, 9, 6, 7, 11); fill(S, 14, 7, 9, 14, 7, 11)
  fill(C, 5, 5, 9, 5, 7, 11); fill(C, 15, 5, 9, 15, 7, 11)
  for (const [x,y,z] of [[5,5,10],[5,6,10],[6,6,10],[15,5,10],[15,6,10],[14,6,10]]) put(A,x,y,z)
  fill(A,2,4,4,2,6,4); fill(A,18,4,4,18,6,4)
  for (const [x,y,z] of [[2,4,5],[2,3,4],[18,4,5],[18,3,4]]) put("minecraft:sandstone_stairs",x,y,z,stair("north"))
  fill(S,1,1,3,2,2,3); fill(S,18,1,3,19,2,3)
  put(S,1,1,2); put(S,19,1,2); put("minecraft:sandstone_slab",1,2,2,{ type:"bottom", waterlogged:"false" }); put("minecraft:sandstone_slab",19,2,2,{ type:"bottom", waterlogged:"false" })
  put("minecraft:sandstone_stairs",2,1,2,stair("west")); put("minecraft:sandstone_stairs",18,1,2,stair("east"))
  fill(S,4,3,5,4,3,17); fill(S,16,3,5,16,3,17); fill(A,3,1,5,4,2,16); fill(A,15,1,5,16,2,16)
  for (let z=5; z<=17; z+=2) { put(C,4,1,z); put(H,4,2,z); put(C,16,1,z); put(H,16,2,z) }
  for (const [x,z] of [[10,7],[10,8],[9,9],[11,9],[8,10],[12,10],[7,10],[13,10],[9,11],[11,11],[10,12],[10,13]]) put("minecraft:orange_terracotta",x,0,z)
  put("minecraft:blue_terracotta",10,0,10)
  const glyph = [C,"minecraft:orange_terracotta",C,C,"minecraft:orange_terracotta",C,"minecraft:orange_terracotta",H,"minecraft:orange_terracotta",C,"minecraft:orange_terracotta",C,"minecraft:orange_terracotta","minecraft:orange_terracotta","minecraft:orange_terracotta",C,C,C]
  for (const x of [0,20]) for (let y=2, i=0; y<=8; y++) for (let z=1; z<=3; z++,i++) put(glyph[i],x,y,z)
  for (const x of [2,18]) for (let y=2, i=0; y<=8; y++) for (let dx=-1; dx<=1; dx++,i++) put(glyph[i],x+dx,y,0)
  fill(C,8,4,0,12,6,0); put(A,8,6,0); put(A,12,6,0); put("minecraft:orange_terracotta",9,5,0); put(H,10,5,0); put("minecraft:orange_terracotta",11,5,0)
  fill(C,8,-14,8,12,-11,12); fill(H,8,-10,8,12,-10,12); fill(C,8,-9,8,12,-9,12); fill(S,8,-8,8,12,-1,12); fill(A,9,-11,9,11,-1,11)
  put("minecraft:stone_pressure_plate",10,-11,10); outline("minecraft:tnt",A,9,-13,9,11,-13,11)
  for (const [x,z,ox,oz] of [[8,10,7,10],[12,10,13,10],[10,8,10,7],[10,12,10,13]]) {
    put(A,x,-11,z); put(A,x,-10,z); put(H,ox,-10,oz); put(C,ox,-11,oz)
    put("minecraft:chest",ox,-11,oz,{ facing: x===8?"west":x===12?"east":z===8?"north":"south", type:"single", waterlogged:"false" },{ id:"minecraft:chest", LootTable:"minecraft:chests/desert_pyramid" })
  }
  return finish([21, 30, 21], [10, 0, 10])
}

// Direct 1.16.1 JungleTempleGenerator shell and redstone/chest rooms. Its
// CobblestoneRandomizer is preserved (40% cobblestone, 60% mossy).
export async function runJungleTemple1161(_load, { seed = 0x11610001 } = {}) {
  const { put, fill, finish } = builder()
  const random = rnd(seed)
  const stone = () => random() < 0.4 ? "minecraft:cobblestone" : "minecraft:mossy_cobblestone"
  const rfill = (x0,y0,z0,x1,y1,z1) => { for(let y=y0;y<=y1;y++) for(let z=z0;z<=z1;z++) for(let x=x0;x<=x1;x++) put(stone(),x,y,z) }
  const air = (x0,y0,z0,x1,y1,z1) => fill("minecraft:air",x0,y0,z0,x1,y1,z1)
  rfill(0,-4,0,11,0,14)
  for (const b of [[2,1,2,9,2,2],[2,1,12,9,2,12],[2,1,3,2,2,11],[9,1,3,9,2,11],[1,3,1,10,6,1],[1,3,13,10,6,13],[1,3,2,1,6,12],[10,3,2,10,6,12],[2,3,2,9,3,12],[2,6,2,9,6,12],[3,7,3,8,7,11],[4,8,4,7,8,10]]) rfill(...b)
  for (const b of [[3,1,3,8,2,11],[4,3,6,7,3,9],[2,4,2,9,5,12],[4,6,5,7,6,9],[5,7,6,6,7,8],[5,1,2,6,2,2],[5,2,12,6,2,12],[5,5,1,6,5,1],[5,5,13,6,5,13]]) air(...b)
  for (const [x,y,z] of [[1,5,5],[10,5,5],[1,5,9],[10,5,9]]) put("minecraft:air",x,y,z)
  for (const z of [0,14]) for (const x of [2,4,7,9]) rfill(x,4,z,x,5,z)
  rfill(5,6,0,6,6,0)
  for(const x of [0,11]) { for(let z=2;z<=12;z+=2) rfill(x,4,z,x,5,z); rfill(x,6,5,x,6,5); rfill(x,6,9,x,6,9) }
  for(const [x,z] of [[2,2],[9,2],[2,12],[9,12]]) rfill(x,7,z,x,9,z)
  for(const [x,z] of [[4,4],[7,4],[4,10],[7,10]]) rfill(x,9,z,x,9,z); rfill(5,9,7,6,9,7)
  for(const [x,y,z,f] of [[5,9,6,"north"],[6,9,6,"north"],[5,9,8,"south"],[6,9,8,"south"],[4,0,0,"north"],[5,0,0,"north"],[6,0,0,"north"],[7,0,0,"north"],[4,1,8,"north"],[4,2,9,"north"],[4,3,10,"north"],[7,1,8,"north"],[7,2,9,"north"],[7,3,10,"north"],[4,4,5,"east"],[7,4,5,"west"]]) put("minecraft:cobblestone_stairs",x,y,z,stair(f))
  for(let k=0;k<4;k++){ put("minecraft:cobblestone_stairs",5,-k,6+k,stair("south")); put("minecraft:cobblestone_stairs",6,-k,6+k,stair("south")); air(5,-k,7+k,6,-k,9+k) }
  air(1,-3,12,10,-1,13); air(1,-3,1,3,-1,13); air(1,-3,1,9,-1,5)
  for(let z=1;z<=13;z+=2) rfill(1,-3,z,1,-2,z); for(let z=2;z<=12;z+=2) rfill(1,-1,z,3,-1,z)
  rfill(2,-2,1,5,-2,1); rfill(7,-2,1,9,-2,1); rfill(6,-3,1,6,-3,1); rfill(6,-1,1,6,-1,1)
  const hook=(f)=>({facing:f,attached:"true",powered:"false"}); const wire=(props)=>({north:"none",east:"none",south:"none",west:"none",...props})
  put("minecraft:tripwire_hook",1,-3,8,hook("east")); put("minecraft:tripwire_hook",4,-3,8,hook("west")); put("minecraft:tripwire",2,-3,8,{east:"true",west:"true",attached:"true",powered:"false",disarmed:"false"}); put("minecraft:tripwire",3,-3,8,{east:"true",west:"true",attached:"true",powered:"false",disarmed:"false"})
  for(let z=2;z<=7;z++) put("minecraft:redstone_wire",5,-3,z,wire({north:"side",south:"side"})); put("minecraft:redstone_wire",5,-3,1,wire({north:"side",west:"side"})); put("minecraft:redstone_wire",4,-3,1,wire({east:"side",west:"side"}))
  put("minecraft:dispenser",3,-2,1,{facing:"north",triggered:"false"},{id:"minecraft:dispenser",LootTable:"minecraft:chests/jungle_temple_dispenser"}); put("minecraft:vine",3,-2,2,{south:"true"})
  put("minecraft:tripwire_hook",7,-3,1,hook("north")); put("minecraft:tripwire_hook",7,-3,5,hook("south")); for(let z=2;z<=4;z++) put("minecraft:tripwire",7,-3,z,{north:"true",south:"true",attached:"true",powered:"false",disarmed:"false"})
  put("minecraft:dispenser",9,-2,3,{facing:"west",triggered:"false"},{id:"minecraft:dispenser",LootTable:"minecraft:chests/jungle_temple_dispenser"}); put("minecraft:chest",8,-3,3,{facing:"north",type:"single",waterlogged:"false"},{id:"minecraft:chest",LootTable:"minecraft:chests/jungle_temple"})
  for(const [x,y,z] of [[9,-3,2],[8,-3,1],[4,-3,5],[5,-2,5],[5,-1,5],[6,-3,5],[7,-2,5],[7,-1,5],[8,-3,5],[10,-2,9]]) put("minecraft:mossy_cobblestone",x,y,z)
  rfill(9,-1,1,9,-1,5); air(8,-3,8,10,-1,10); for(const x of [8,9,10]) put("minecraft:chiseled_stone_bricks",x,-2,11)
  for(const x of [8,9,10]) put("minecraft:lever",x,-2,12,{face:"wall",facing:"north",powered:"false"})
  rfill(8,-3,8,8,-3,10); rfill(10,-3,8,10,-3,10); put("minecraft:sticky_piston",9,-2,8,{facing:"up",extended:"false"}); put("minecraft:sticky_piston",10,-2,8,{facing:"west",extended:"false"}); put("minecraft:sticky_piston",10,-1,8,{facing:"west",extended:"false"}); put("minecraft:repeater",10,-2,10,{facing:"north",delay:"1",locked:"false",powered:"false"})
  put("minecraft:chest",9,-3,10,{facing:"north",type:"single",waterlogged:"false"},{id:"minecraft:chest",LootTable:"minecraft:chests/jungle_temple"})
  return finish([12, 14, 15], [6, 0, 7])
}
