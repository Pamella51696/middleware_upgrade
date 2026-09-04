package fisheye270;

/**
 * Drop-in entry point matching the original VideoStreamingServer main:
 *   java ... VideoStreamingServer [port] [front] [left] [right] [rear]
 * or Maven:
 *   mvn -q exec:java -Dexec.args="9090 front.mov left.mov right.mov rear.mov"
 */
public final class VideoStreamingServer {
    public static void main(String[] args) throws Exception {
        if (args.length == 0) {
            App.main(new String[]{"serve", "9090"});
            return;
        }
        String a0 = args[0];
        if (a0.matches("synth|phase1|phase2|phase3|compare|compare-projections|align|run|serve")) {
            App.main(args);
            return;
        }
        // numeric port → original calling convention
        String[] forwarded = new String[args.length + 1];
        forwarded[0] = "serve";
        System.arraycopy(args, 0, forwarded, 1, args.length);
        App.main(forwarded);
    }
}
