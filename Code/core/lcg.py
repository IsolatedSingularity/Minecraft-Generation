"""
Linear Congruential Generator Implementation

Implements Java's Random class LCG for authentic Minecraft RNG.
"""

import numpy as np


class MinecraftLCG:
    """
    Minecraft's Linear Congruential Generator implementation.
    
    Follows Java's Random class specification exactly:
    X_{n+1} = (a * X_n + c) mod m
    
    where a = 0x5DEECE66D, c = 0xB, m = 2^48
    """
    
    # Java Random LCG constants
    MULTIPLIER = 0x5DEECE66D
    ADDEND = 0xB
    MODULUS = 2**48
    
    def __init__(self, seed):
        """Initialize with a world seed."""
        self.seed = (seed ^ self.MULTIPLIER) & (self.MODULUS - 1)
        self.initial_seed = seed
        
    def next_bits(self, bits):
        """Generate next n bits of randomness."""
        self.seed = (self.MULTIPLIER * self.seed + self.ADDEND) % self.MODULUS
        return self.seed >> (48 - bits)
    
    def next_int(self, bound=None):
        """Generate random integer, optionally bounded."""
        if bound is None:
            return self.next_bits(32)
        
        if bound <= 0:
            raise ValueError("bound must be positive")
            
        # Power of 2 optimization
        if (bound & (bound - 1)) == 0:
            return (bound * self.next_bits(31)) >> 31
        
        # Rejection sampling for uniform distribution
        bits = self.next_bits(31)
        val = bits % bound
        while bits - val + (bound - 1) < 0:
            bits = self.next_bits(31)
            val = bits % bound
        return val
    
    def next_float(self):
        """Generate random float in [0, 1)."""
        return self.next_bits(24) / float(1 << 24)
    
    def next_double(self):
        """Generate random double in [0, 1)."""
        return ((self.next_bits(26) << 27) + self.next_bits(27)) / float(1 << 53)


def generate_region_seed(world_seed, region_x, region_z, salt):
    """
    Generate region seed using Minecraft's exact algorithm.
    
    This formula is used for structure placement (villages, temples, etc.)
    
    Parameters
    ----------
    world_seed : int
        The world seed
    region_x : int
        Region X coordinate
    region_z : int  
        Region Z coordinate
    salt : int
        Structure-specific salt value
        
    Returns
    -------
    int
        32-bit region seed
    """
    return (world_seed + 
            region_x * region_x * 4987142 + 
            region_x * 5947611 + 
            region_z * region_z * 4392871 + 
            region_z * 389711 + 
            salt) & 0xFFFFFFFF


# Common structure salts
VILLAGE_SALT = 10387312
FORTRESS_SALT = 30084232
MONUMENT_SALT = 10387313
MANSION_SALT = 10387319
STRONGHOLD_SALT = 0  # Strongholds use different algorithm
