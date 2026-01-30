"""
Noise Generation Functions

Implements Perlin-like noise for terrain and biome generation.
"""

import numpy as np


def simple_perlin_noise(x, y, seed=0):
    """
    Simple 2D noise function as a replacement for pnoise2.
    
    Uses multi-frequency sine waves with position-based randomness
    to approximate Perlin noise characteristics.
    
    Parameters
    ----------
    x : float
        X coordinate in noise space
    y : float
        Y coordinate in noise space
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    float
        Noise value roughly in range [-1, 1]
    """
    # Multi-frequency noise using sine waves
    freq1, freq2, freq3 = 0.01, 0.05, 0.1
    amp1, amp2, amp3 = 1.0, 0.5, 0.25
    
    noise = (amp1 * np.sin(freq1 * x + seed * 0.1) * np.cos(freq1 * y) +
             amp2 * np.sin(freq2 * x) * np.cos(freq2 * y + seed * 0.05) +
             amp3 * np.sin(freq3 * x + seed * 0.02) * np.cos(freq3 * y))
    
    # Add position-based randomness
    hash_val = hash((int(x/10), int(y/10), seed)) % 1000000
    np.random.seed(hash_val)
    noise += 0.3 * (np.random.random() - 0.5)
    
    return noise / 2.0


def generate_octave_noise(x_coords, y_coords, num_octaves=6, seed=0):
    """
    Generate multi-octave noise field.
    
    Parameters
    ----------
    x_coords : array-like
        X coordinates grid
    y_coords : array-like  
        Y coordinates grid
    num_octaves : int
        Number of noise octaves to combine
    seed : int
        Random seed
        
    Returns
    -------
    ndarray
        2D noise field
    """
    X, Y = np.meshgrid(x_coords, y_coords)
    noise_field = np.zeros_like(X, dtype=float)
    
    for octave in range(num_octaves):
        frequency = 2 ** octave / 1000.0
        amplitude = 1.0 / (2 ** octave)
        
        octave_noise = np.array([
            [simple_perlin_noise(i * frequency, j * frequency, seed + octave * 100)
             for i in x_coords] for j in y_coords
        ])
        
        noise_field += amplitude * octave_noise
    
    return np.clip(noise_field, -1, 1)
