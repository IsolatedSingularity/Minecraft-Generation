import java.awt.image.BufferedImage;
import java.io.File;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.stream.IntStream;
import javax.imageio.ImageIO;

/**
 * Export a compact Minecraft 1.16.1 Overworld surface sample from the original
 * server JAR.  This helper deliberately uses reflection so it can be compiled
 * without redistributing any Mojang classes.
 *
 * Pixel layout: red = raw biome id, green = WORLD_SURFACE_WG height,
 * blue = format version (1), alpha = 255.
 */
public final class VanillaTerrainCache {
    private static Class<?> cls(String name) throws Exception {
        return Class.forName(name);
    }

    private static Object invokeStatic(Class<?> owner, String name, Class<?>[] types, Object... args)
            throws Exception {
        Method method = owner.getDeclaredMethod(name, types);
        method.setAccessible(true);
        return method.invoke(null, args);
    }

    private static Object field(Object target, String name) throws Exception {
        Field value = target.getClass().getDeclaredField(name);
        value.setAccessible(true);
        return value.get(target);
    }

    private static Object staticField(Class<?> owner, String name) throws Exception {
        Field value = owner.getDeclaredField(name);
        value.setAccessible(true);
        return value.get(null);
    }

    private static Method method(Class<?> owner, String name, int parameters) {
        for (Method candidate : owner.getDeclaredMethods()) {
            if (candidate.getName().equals(name) && candidate.getParameterCount() == parameters) {
                candidate.setAccessible(true);
                return candidate;
            }
        }
        throw new IllegalArgumentException(owner.getName() + "." + name + "/" + parameters);
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 8) {
            throw new IllegalArgumentException(
                "usage: seed minX maxX minZ maxZ width height output.png");
        }
        long seed = Long.parseLong(args[0]);
        int minX = Integer.parseInt(args[1]);
        int maxX = Integer.parseInt(args[2]);
        int minZ = Integer.parseInt(args[3]);
        int maxZ = Integer.parseInt(args[4]);
        int width = Integer.parseInt(args[5]);
        int height = Integer.parseInt(args[6]);
        File output = new File(args[7]);

        // Bootstrap the built-in registries before asking for the default
        // dimension settings and biome registry.
        invokeStatic(cls("uj"), "a", new Class<?>[0]);

        Object generator = invokeStatic(cls("cix"), "a", new Class<?>[] { long.class }, seed);
        Method baseHeight = method(generator.getClass(), "a", 3);
        Object surfaceHeightmap = staticField(cls("cio$a"), "a");

        java.lang.reflect.Constructor<?> sourceConstructor = cls("bti").getDeclaredConstructor(
            long.class, boolean.class, boolean.class);
        sourceConstructor.setAccessible(true);
        Object biomeSource = sourceConstructor.newInstance(seed, false, false);
        Object biomeLayer = field(biomeSource, "f");
        Method sampleBiome = method(biomeLayer.getClass(), "a", 2);
        Object biomeRegistry = staticField(cls("gl"), "as");
        Method rawBiomeId = null;
        for (Method candidate : biomeRegistry.getClass().getMethods()) {
            if (candidate.getName().equals("a") && candidate.getParameterCount() == 1
                    && candidate.getParameterTypes()[0].equals(Object.class)
                    && candidate.getReturnType().equals(int.class)) {
                rawBiomeId = candidate;
                break;
            }
        }
        if (rawBiomeId == null) {
            throw new IllegalStateException("Could not find biome raw-ID method");
        }

        int[] pixels = new int[width * height];
        final Method biomeIdMethod = rawBiomeId;
        IntStream.range(0, height).parallel().forEach(pixelZ -> {
            int blockZ = Math.round(minZ + pixelZ * (maxZ - minZ) / (float)(height - 1));
            for (int pixelX = 0; pixelX < width; pixelX++) {
                int blockX = Math.round(minX + pixelX * (maxX - minX) / (float)(width - 1));
                try {
                    Object biome = sampleBiome.invoke(
                        biomeLayer, Math.floorDiv(blockX, 4), Math.floorDiv(blockZ, 4));
                    int biomeId = ((Number)biomeIdMethod.invoke(biomeRegistry, biome)).intValue();
                    int surface = ((Number)baseHeight.invoke(
                        generator, blockX, blockZ, surfaceHeightmap)).intValue();
                    int packed = 0xff000000 | ((biomeId & 0xff) << 16) | ((surface & 0xff) << 8) | 1;
                    pixels[(height - 1 - pixelZ) * width + pixelX] = packed;
                } catch (ReflectiveOperationException error) {
                    throw new RuntimeException(error);
                }
            }
        });
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        image.setRGB(0, 0, width, height, pixels, 0, width);
        output.getParentFile().mkdirs();
        ImageIO.write(image, "png", output);
        System.out.printf("wrote %s (%dx%d), seed=%d, x=%d..%d, z=%d..%d%n",
            output, width, height, seed, minX, maxX, minZ, maxZ);
    }
}
