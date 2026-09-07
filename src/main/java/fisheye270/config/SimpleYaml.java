package fisheye270.config;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Tiny YAML subset reader (maps, scalars, comments). No SnakeYAML JAR required,
 * so {@code javac -cp opencv-490.jar} is enough.
 */
public final class SimpleYaml {
    private SimpleYaml() {}

    public static Map<String, Object> load(Path path) throws IOException {
        return parse(Files.readAllLines(path, StandardCharsets.UTF_8));
    }

    static Map<String, Object> parse(List<String> raw) {
        List<String> lines = new ArrayList<>();
        for (String line : raw) {
            String s = stripComment(line);
            if (s.isBlank()) continue;
            if (s.trim().startsWith("-")) continue; // flow/block lists unused by AppConfig
            lines.add(s);
        }
        int[] idx = {0};
        return parseMap(lines, idx, indentOf(lines.isEmpty() ? "" : lines.get(0)));
    }

    static Map<String, Object> parseMap(List<String> lines, int[] idx, int minIndent) {
        Map<String, Object> map = new LinkedHashMap<>();
        while (idx[0] < lines.size()) {
            String line = lines.get(idx[0]);
            int ind = indentOf(line);
            if (ind < minIndent) break;
            if (ind > minIndent && map.isEmpty()) {
                minIndent = ind;
            } else if (ind > minIndent) {
                break;
            }
            idx[0]++;
            String trimmed = line.trim();
            int colon = trimmed.indexOf(':');
            if (colon < 0) continue;
            String key = trimmed.substring(0, colon).trim();
            String rest = trimmed.substring(colon + 1).trim();
            if (rest.isEmpty()) {
                if (idx[0] < lines.size() && indentOf(lines.get(idx[0])) > ind) {
                    map.put(key, parseMap(lines, idx, indentOf(lines.get(idx[0]))));
                } else {
                    map.put(key, new LinkedHashMap<String, Object>());
                }
            } else {
                map.put(key, scalar(rest));
            }
        }
        return map;
    }

    static Object scalar(String raw) {
        if ("null".equals(raw) || "~".equals(raw)) return null;
        if ("true".equalsIgnoreCase(raw)) return Boolean.TRUE;
        if ("false".equalsIgnoreCase(raw)) return Boolean.FALSE;
        if ((raw.startsWith("\"") && raw.endsWith("\"")) || (raw.startsWith("'") && raw.endsWith("'"))) {
            return raw.substring(1, raw.length() - 1);
        }
        try {
            if (raw.contains(".") || raw.contains("e") || raw.contains("E")) {
                return Double.parseDouble(raw);
            }
            return Long.parseLong(raw);
        } catch (NumberFormatException e) {
            return raw;
        }
    }

    static int indentOf(String line) {
        int i = 0;
        while (i < line.length() && line.charAt(i) == ' ') i++;
        return i;
    }

    static String stripComment(String line) {
        boolean quote = false;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"') quote = !quote;
            if (c == '#' && !quote) return line.substring(0, i);
        }
        return line;
    }
}
