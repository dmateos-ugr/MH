const std = @import("std");
const utils = @import("utils.zig");
const ref_algorithms = @import("ref_algorithms.zig");
const busquedaLocalP1 = @import("bl.zig").busquedaLocalP1;
const genetic = @import("genetic.zig");
const Random = std.rand.Random;
const Allocator = std.mem.Allocator;
const Example = utils.Example;
const print = utils.print;

const AlgorithmFn = *const fn (training_set: []const Example, allocator: Allocator, rnd: Random) error{OutOfMemory}![]const f64;

const Algorithm = struct {
    func: AlgorithmFn,
    name: []const u8,
};

const AlgorithmResult = struct {
    fitness_data: utils.FitnessData,
    time: u64,
};

const Args = struct {
    seed: ?usize,
    dataset: ?[]const u8,
};

fn showArgsHelp() void {
    print(
        \\Usage:
        \\    p1 [-h] [-s] dataset
        \\
        \\Available options:
        \\    -h, --help      Display this help and exit.
        \\    -s, --seed <n>  The seed to use for random number generation.
        \\    dataset         The dataset to use: diabetes, ozone-320, or spectf-heart.
        \\
    , .{});
}

fn parseArgs() ?Args {
    var args = Args{
        .seed = null,
        .dataset = null,
    };

    var args_it = std.process.args();
    _ = args_it.skip();
    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            return null;
        } else if (std.mem.eql(u8, arg, "--seed") or std.mem.eql(u8, arg, "-s")) {
            const arg_seed = args_it.next() orelse {
                print("Argument '{s}' specified, but no seed was provided\n\n", .{arg});
                return null;
            };
            const seed = std.fmt.parseInt(usize, arg_seed, 0) catch |err| {
                print("Failed to parse seed argument '{s}' with {}\n\n", .{ arg_seed, err });
                return null;
            };
            args.seed = seed;
        } else {
            if (std.mem.startsWith(u8, arg, "-")) {
                print("Unknown argument '{s}'\n\n", .{arg});
                return null;
            }
            if (args.dataset != null) {
                print("Unexpected argument '{s}'\n\n", .{arg});
                return null;
            } else {
                // Check dataset is valid
                const options = [_][]const u8{ "diabetes", "ozone-320", "spectf-heart" };
                for (&options) |option| {
                    if (std.mem.eql(u8, arg, option))
                        break;
                } else {
                    print("Unknown dataset: '{s}'\n\n", .{arg});
                    return null;
                }
                args.dataset = arg;
            }
        }
    }
    if (args.dataset == null) {
        print("Missing argument: dataset\n\n", .{});
        return null;
    }
    return args;
}

pub fn main() !void {
    const args = parseArgs() orelse {
        showArgsHelp();
        return;
    };
    const rng_seed = args.seed orelse @bitCast(usize, std.time.microTimestamp()) % 100;
    print("Dataset: {s}\n", .{args.dataset.?});
    print("Seed: {}\n", .{rng_seed});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rng = std.rand.DefaultPrng.init(rng_seed);
    const rnd = rng.random();

    const n_threads = try utils.initThreadPool(allocator);
    defer utils.deinitThreadPool();
    print("Number of threads: {}\n", .{n_threads});

    const partitions = try utils.readPartitions(args.dataset.?, allocator);
    defer utils.freePartitions(partitions, allocator);

    const algorithms = [_]Algorithm{
        // .{ .func = busquedaLocalP1, .name = "BUSQUEDA LOCAL" },
        // .{ .func = ref_algorithms.greedy, .name = "GREEDY" },
        // .{ .func = ref_algorithms.algOriginal1NN, .name = "1NN" },
        .{ .func = genetic.AGG_Arit, .name = "AGG Arit" },
        .{ .func = genetic.AGG_BLX, .name = "AGG BLX" },
        .{ .func = genetic.AGE_BLX, .name = "AGE BLX" },
        .{ .func = genetic.AGE_Arit, .name = "AGE Arit" },
        .{ .func = genetic.AM_Best, .name = "AM Best" },
        .{ .func = genetic.AM_All, .name = "AM All" },
        .{ .func = genetic.AM_Rand, .name = "AM Rand" },
    };

    for (algorithms) |algorithm| {
        var result_average = std.mem.zeroes(AlgorithmResult);

        print("[ALGORITMO {s}]\n", .{algorithm.name});
        print("              %_clas   %_red  Fitness   T (s)\n", .{});
        for (0..partitions.len) |i| {
            const training_set = try utils.joinPartitions(partitions, i, allocator);
            defer allocator.free(training_set);
            const test_set = partitions[i];

            const time_start = std.time.milliTimestamp();
            const w = try algorithm.func(training_set, allocator, rnd);
            defer allocator.free(w);
            const time = std.time.milliTimestamp() - time_start;
            const fitness_data = utils.getFitnessData(w, test_set, training_set);

            result_average.fitness_data.fitness += fitness_data.fitness;
            result_average.fitness_data.tasa_clas += fitness_data.tasa_clas;
            result_average.fitness_data.tasa_red += fitness_data.tasa_red;
            result_average.time += @intCast(u64, time);

            print("Particion {}:  {d:6.3}  {d:6.3}  {d:7.3}  {d:6.3}\n", .{
                i + 1,
                fitness_data.tasa_clas,
                fitness_data.tasa_red,
                fitness_data.fitness,
                @intToFloat(f64, time) / 1000,
            });
        }

        // Print average
        result_average.fitness_data.fitness /= @intToFloat(f64, partitions.len);
        result_average.fitness_data.tasa_clas /= @intToFloat(f64, partitions.len);
        result_average.fitness_data.tasa_red /= @intToFloat(f64, partitions.len);
        result_average.time /= partitions.len;
        print("Media:        {d:6.3}  {d:6.3}  {d:7.3}  {d:6.3}\n\n", .{
            result_average.fitness_data.tasa_clas,
            result_average.fitness_data.tasa_red,
            result_average.fitness_data.fitness,
            @intToFloat(f64, result_average.time) / 1000,
        });
    }
}

test {
    std.testing.refAllDecls(@This());
}

// Having a 100% reduction rate weight means the classifier assigns to each
// test element the class of the first element in the training set
test {
    const allocator = std.testing.allocator;

    _ =try utils.initThreadPool(allocator);
    defer utils.deinitThreadPool();

    const partitions = try utils.readPartitions("diabetes", allocator);
    defer utils.freePartitions(partitions, allocator);

    const training_set = try utils.joinPartitions(partitions, 0, allocator);
    defer allocator.free(training_set);
    const test_set = partitions[0];

    const w = try allocator.alloc(f64, training_set[0].attributes.len);
    defer allocator.free(w);
    std.mem.set(f64, w, 0);

    const tasa_clas = utils.tasaClas(w, test_set, training_set);

    var n: usize = 0;
    for (test_set) |test_example| {
        if (std.mem.eql(u8, test_example.class, training_set[0].class))
            n += 1;
    }
    const tasa_clas_calc = 100.0 * @intToFloat(f64, n) / @intToFloat(f64, test_set.len);

    try std.testing.expectEqual(tasa_clas_calc, tasa_clas);
}