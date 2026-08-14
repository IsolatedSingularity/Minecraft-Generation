import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;

/**
 * Read-only parity oracle for the bundled obfuscated Minecraft 1.16.1 server.
 *
 * Compile this file against no Minecraft classes, then put the original
 * server JAR on the runtime class path.  Reflection keeps the audit helper
 * independent of a particular decompiler or mapping namespace.
 */
public final class VanillaBiomeOracle {
    private VanillaBiomeOracle() {}

    public static void main(String[] arguments) throws Exception {
        long seed = arguments.length == 0 ? 42L : Long.parseLong(arguments[0]);
        Class<?> bootstrapClass = Class.forName("uj");
        Method bootstrap = bootstrapClass.getDeclaredMethod("a");
        bootstrap.setAccessible(true);
        bootstrap.invoke(null);
        Class<?> sourceClass = Class.forName("bti");
        Constructor<?> constructor = sourceClass.getDeclaredConstructor(
            long.class, boolean.class, boolean.class
        );
        constructor.setAccessible(true);
        Object source = constructor.newInstance(seed, false, false);

        Field layerField = sourceClass.getDeclaredField("f");
        layerField.setAccessible(true);
        Object layer = layerField.get(source);
        Method sample = layer.getClass().getDeclaredMethod("a", int.class, int.class);
        sample.setAccessible(true);

        Class<?> registryClass = Class.forName("gl");
        Field biomeField = registryClass.getDeclaredField("as");
        biomeField.setAccessible(true);
        Object registry = biomeField.get(null);
        Method rawId = null;
        for (Method method : registry.getClass().getMethods()) {
            if (
                method.getName().equals("a")
                && method.getParameterCount() == 1
                && method.getParameterTypes()[0].equals(Object.class)
                && method.getReturnType().equals(int.class)
            ) {
                rawId = method;
                break;
            }
        }
        if (rawId == null) {
            throw new IllegalStateException("Could not find biome raw-ID method");
        }

        int[][] points;
        if (arguments.length > 2 && (arguments.length - 1) % 2 == 0) {
            points = new int[(arguments.length - 1) / 2][2];
            for (int index = 1; index < arguments.length; index += 2) {
                points[(index - 1) / 2][0] = Integer.parseInt(arguments[index]);
                points[(index - 1) / 2][1] = Integer.parseInt(arguments[index + 1]);
            }
        } else {
            points = new int[][] {
                {0, 0}, {1, 0}, {0, 1}, {100, 100}, {-100, 40},
                {1000, -700}, {-4096, -4096}, {4096, 4096},
            };
        }
        for (int[] point : points) {
            Object biome = sample.invoke(layer, point[0], point[1]);
            int id = (Integer) rawId.invoke(registry, biome);
            System.out.printf("biome:%d,%d=%d%n", point[0], point[1], id);
        }

        Class<?> netherSourceClass = Class.forName("btc");
        Method defaultNether = netherSourceClass.getDeclaredMethod("d", long.class);
        defaultNether.setAccessible(true);
        Object netherSource = defaultNether.invoke(null, seed);
        Method netherSample = netherSourceClass.getDeclaredMethod(
            "b", int.class, int.class, int.class
        );
        netherSample.setAccessible(true);
        for (int[] point : points) {
            Object biome = netherSample.invoke(netherSource, point[0], 0, point[1]);
            int id = (Integer) rawId.invoke(registry, biome);
            System.out.printf("nether:%d,%d=%d%n", point[0], point[1], id);
        }

        Class<?> settingsClass = Class.forName("cix");
        Method defaultOverworld = settingsClass.getDeclaredMethod("a", long.class);
        defaultOverworld.setAccessible(true);
        Object generator = defaultOverworld.invoke(null, seed);
        Class<?> heightTypeClass = Class.forName("cio$a");
        Field worldSurfaceField = heightTypeClass.getDeclaredField("a");
        worldSurfaceField.setAccessible(true);
        Object worldSurface = worldSurfaceField.get(null);
        Method height = generator.getClass().getDeclaredMethod(
            "a", int.class, int.class, heightTypeClass
        );
        height.setAccessible(true);
        for (int[] point : points) {
            int value = (Integer) height.invoke(
                generator, point[0], point[1], worldSurface
            );
            System.out.printf("height:%d,%d=%d%n", point[0], point[1], value);
        }
    }
}
