"""Java Random primitives used by Java Edition 1.16.1 simulations."""


class MinecraftLCG:
    """Bit-compatible implementation of java.util.Random."""

    MULTIPLIER = 0x5DEECE66D
    ADDEND = 0xB
    MODULUS = 2**48
    MASK = MODULUS - 1

    def __init__(self, seed):
        self.initial_seed = int(seed)
        self.seed = (self.initial_seed ^ self.MULTIPLIER) & self.MASK

    def set_seed(self, seed):
        self.initial_seed = int(seed)
        self.seed = (self.initial_seed ^ self.MULTIPLIER) & self.MASK
        return self

    def next_bits(self, bits):
        self.seed = (self.MULTIPLIER * self.seed + self.ADDEND) & self.MASK
        return self.seed >> (48 - bits)

    def next_int(self, bound=None):
        if bound is None:
            value = self.next_bits(32)
            return value if value < (1 << 31) else value - (1 << 32)
        if bound <= 0:
            raise ValueError('bound must be positive')
        if (bound & (bound - 1)) == 0:
            return (bound * self.next_bits(31)) >> 31

        bits = self.next_bits(31)
        value = bits % bound
        while ((bits - value + bound - 1) & 0xFFFFFFFF) >= 0x80000000:
            bits = self.next_bits(31)
            value = bits % bound
        return value

    def next_float(self):
        return self.next_bits(24) / float(1 << 24)

    def next_double(self):
        return (
            (self.next_bits(26) << 27) + self.next_bits(27)
        ) / float(1 << 53)

    def next_long(self):
        high = self.next_int()
        low = self.next_int()
        return to_signed_long((high << 32) + low)

    def advance(self, steps):
        for _ in range(int(steps)):
            self.next_bits(1)
        return self.seed


def to_signed_long(value):
    """Wrap an integer to Java's signed 64-bit long."""
    value &= 0xFFFFFFFFFFFFFFFF
    return value if value < (1 << 63) else value - (1 << 64)


def generate_region_seed(world_seed, region_x, region_z, salt):
    """Match the large-feature region seed used by Java 1.16.1."""
    return to_signed_long(
        int(world_seed)
        + int(region_x) * 341873128712
        + int(region_z) * 132897987541
        + int(salt)
    )


def generate_population_seed(world_seed, block_x, block_z):
    """Match ChunkRandom.setPopulationSeed for Java 1.16.1."""
    random = MinecraftLCG(world_seed)
    x_multiplier = random.next_long() | 1
    z_multiplier = random.next_long() | 1
    mixed = to_signed_long(
        int(block_x) * x_multiplier + int(block_z) * z_multiplier
    )
    return to_signed_long(mixed ^ int(world_seed))


VILLAGE_SALT = 10387312
FORTRESS_SALT = 30084232
MONUMENT_SALT = 10387313
MANSION_SALT = 10387319
STRONGHOLD_SALT = 0
