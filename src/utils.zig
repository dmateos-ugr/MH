const std = @import("std");
const Random = std.rand.Random;
const Allocator = std.mem.Allocator;

// Threshold for considering an attribute as discarded
const MIN_WEIGHT = 0.1;

// Alpha used in: fitness = alpha·tasa_clas + (1-alpha)·tasa_red
const ALPHA = 0.8;

const N_PARTITIONS = 5;

const MOV_VARIANCE = 0.3;
const MOV_STDDEV = std.math.sqrt(MOV_VARIANCE);

pub fn print(comptime format: []const u8, args: anytype) void {
    const stdout = std.io.getStdOut().writer();
    stdout.print(format, args) catch unreachable;
}

pub const Example = struct {
    attributes: []f64,
    class: []const u8,

    pub fn init(attributes: []f64, class: []const u8) !Example {
        return Example{
            .attributes = attributes,
            .class = class,
        };
    }

    pub fn deinit(self: Example, allocator: Allocator) void {
        allocator.free(self.attributes);
        allocator.free(self.class);
    }

    pub fn distance(self: Example, other: Example) f64 {
        var ret: f64 = 0;
        for (self.attributes, other.attributes) |attr1, attr2| {
            const dist = attr1 - attr2;
            ret += dist * dist;
        }
        return ret;
    }

    pub fn distanceWeighted(self: Example, other: Example, w: []const f64) f64 {
        return distanceWeightedRaw(w.len, self.attributes.ptr, other.attributes.ptr, w.ptr);
    }

    fn distanceWeightedRaw(
        n: usize,
        noalias attributes1: [*]const f64,
        noalias attributes2: [*]const f64,
        noalias w: [*]const f64,
    ) f64 {
        var ret: f64 = 0;
        for (attributes1[0..n], attributes2[0..n], w[0..n]) |attr1, attr2, weight| {
            if (weight < MIN_WEIGHT) continue;
            const dist = attr1 - attr2;
            ret += dist * dist * weight;
        }
        return ret;
    }

    const SIMD_LENGTH = std.simd.suggestVectorSize(f64) orelse 4;
    const SimdType = @Vector(SIMD_LENGTH, f64);
    const MIN_WEIGHT_SIMD = @splat(SIMD_LENGTH, @as(f64, MIN_WEIGHT));

    fn loadSimd(mem: [*]const f64) SimdType {
        var result = @splat(SIMD_LENGTH, @as(f64, 0.0));
        comptime var i: usize = 0;
        inline while (i < SIMD_LENGTH) : (i += 1) {
            result[i] = mem[i];
        }
        return result;
    }

    pub fn distanceWeightedSimd(self: Example, other: Example, w: []const f64) f64 {
        return distanceWeightedSimdRaw(w.len, self.attributes.ptr, other.attributes.ptr, w.ptr);
    }

    fn distanceWeightedSimdRaw(
        n: usize,
        noalias attributes1: [*]const f64,
        noalias attributes2: [*]const f64,
        noalias w: [*]const f64,
    ) f64 {
        var ret: f64 = 0;
        var i: usize = 0;
        const n_sequentially = n % SIMD_LENGTH;
        const n_in_chunks = n - n_sequentially;

        // Process attributes in chunks of SIMD_LENGTH
        while (i < n_in_chunks) : (i += SIMD_LENGTH) {
            // Load values from memory
            const attrs1 = loadSimd(attributes1 + i);
            const attrs2 = loadSimd(attributes2 + i);
            const weights = loadSimd(w + i);

            // Create a vector of booleans indicating whether each weight is
            // discarded or not
            const discarded_weights = weights < MIN_WEIGHT_SIMD;

            // Create a vector that has 0 for discarded weights, and the
            // original weight for the rest
            const zeroes = @splat(SIMD_LENGTH, @as(f64, 0));
            const final_weights = @select(f64, discarded_weights, zeroes, weights);

            const dist = attrs1 - attrs2;
            const values = dist * dist * final_weights;
            ret += @reduce(.Add, values);
        }

        // Process the rest sequentially
        if (n_sequentially > 0) {
            ret += distanceWeightedRaw(
                n_sequentially,
                attributes1 + n_in_chunks,
                attributes2 + n_in_chunks,
                w + n_in_chunks,
            );
        }

        return ret;
    }
};

fn readExamplesFromFile(filename: []const u8, allocator: Allocator) ![]Example {
    const file = try std.fs.cwd().openFile(filename, .{});
    const reader = file.reader();

    // First part of the file: count number of attributes
    var num_attributes: usize = 0;
    var buf: [1024]u8 = undefined;
    while (try reader.readUntilDelimiterOrEof(&buf, '\n')) |line| {
        if (std.mem.startsWith(u8, line, "@attribute")) {
            num_attributes += 1;
        } else if (std.mem.startsWith(u8, line, "@data")) {
            break;
        }
    }

    // Last one is the class
    num_attributes -= 1;

    // Second part of the files: read examples
    var examples = std.ArrayList(Example).init(allocator);
    while (try reader.readUntilDelimiterOrEof(&buf, '\n')) |line| {
        const num_commas = std.mem.count(u8, line, ",");
        if (num_commas != num_attributes) {
            print("Error reading file: invalid example '{s}' (expected {} attributes + class, found {})\n", .{ line, num_attributes, num_commas });
            return error.InvalidExample;
        }

        const attributes = try allocator.alloc(f64, num_attributes);
        var i: usize = 0;
        var iter = std.mem.tokenize(u8, line, ",");
        while (iter.next()) |token| : (i += 1) {
            if (i < num_attributes) {
                const f = std.fmt.parseFloat(f64, token) catch unreachable;
                attributes[i] = f;
            } else {
                std.debug.assert(iter.peek() == null);
                const class = try allocator.dupe(u8, token);
                const example = try Example.init(attributes, class);
                try examples.append(example);
            }
        }
        std.debug.assert(i == num_attributes + 1);
    }

    return examples.toOwnedSlice();
}

fn normalizeExamples(partitions: [N_PARTITIONS][]Example) void {
    const n = partitions[0][0].attributes.len;

    for (0..n) |i| {
        // Calculate max and min of the attribute
        var min = partitions[0][0].attributes[i];
        var max = min;
        for (partitions) |partition| {
            for (partition) |example| {
                min = std.math.min(min, example.attributes[i]);
                max = std.math.max(max, example.attributes[i]);
            }
        }

        // Normalize attribute in every example
        for (partitions) |partition| {
            for (partition) |example| {
                const new_value = (example.attributes[i] - min) / (max - min);
                std.debug.assert(0 <= new_value and new_value <= 1);
                example.attributes[i] = new_value;
            }
        }
    }
}

pub fn joinPartitions(partitions: [N_PARTITIONS][]const Example, skip_idx: usize, allocator: Allocator) ![]Example {
    var used_partitions: [N_PARTITIONS - 1][]const Example = undefined;
    for (0..partitions.len) |i| {
        if (i == skip_idx) continue;
        const i_save = if (i < skip_idx) i else i - 1;
        used_partitions[i_save] = partitions[i];
    }

    return try std.mem.concat(allocator, Example, &used_partitions);
}

pub fn readPartitions(dataset: []const u8, allocator: Allocator) ![N_PARTITIONS][]Example {
    var partitions: [N_PARTITIONS][]Example = undefined;
    for (&partitions, 0..) |*partition, i| {
        const path = try std.fmt.allocPrint(allocator, "Instancias_APC/{s}_{}.arff", .{ dataset, i + 1 });
        defer allocator.free(path);
        partition.* = readExamplesFromFile(path, allocator) catch |err| switch (err) {
            error.FileNotFound => {
                print("Failed to read file {s}\n", .{path});
                std.process.exit(1);
            },
            else => return err,
        };
    }

    normalizeExamples(partitions);

    return partitions;
}

pub fn freePartitions(partitions: [N_PARTITIONS][]Example, allocator: Allocator) void {
    for (partitions) |partition| {
        for (partition) |example| {
            example.deinit(allocator);
        }
        allocator.free(partition);
    }
}

pub fn mov(w: []f64, i: usize, rnd: Random) void {
    const z_i = rndNorm(rnd, 0, MOV_STDDEV);
    const result = w[i] + z_i;
    const result_truncated = std.math.max(0, std.math.min(1, result));
    w[i] = result_truncated;
}

fn rndNorm(rnd: Random, mean: f64, stddev: f64) f64 {
    return rnd.floatNorm(f64) * stddev + mean;
}

pub fn createRandomSolution(n: usize, allocator: Allocator, rnd: Random) ![]f64 {
    const w = try allocator.alloc(f64, n);
    for (w) |*weight| {
        weight.* = rnd.float(f64); // range [0, 1)
    }
    return w;
}

pub const FitnessData = struct {
    tasa_clas: f64,
    tasa_red: f64,
    fitness: f64,
};

pub fn getFitnessData(w: []const f64, test_set: []const Example, training_set: []const Example) FitnessData {
    const tasa_clas = tasaClas(w, test_set, training_set);
    const tasa_red = tasaRed(w);
    const fitness = ALPHA * tasa_clas + (1 - ALPHA) * tasa_red;
    return FitnessData{
        .tasa_clas = tasa_clas,
        .tasa_red = tasa_red,
        .fitness = fitness,
    };
}

pub fn getFitness(w: []const f64, test_set: []const Example, training_set: []const Example) f64 {
    return getFitnessData(w, test_set, training_set).fitness;
}

var g_thread_pool_inited = false;
var g_thread_pool: std.Thread.Pool = undefined;
var g_thread_results: []usize = undefined;

pub fn initThreadPool(allocator: Allocator, n_threads_arg: ?u32) !usize {
    try g_thread_pool.init(.{
        .allocator = allocator,
        .n_jobs = n_threads_arg,
    });
    const n_threads = g_thread_pool.threads.len;
    g_thread_results = try allocator.alloc(usize, n_threads);
    g_thread_pool_inited = true;
    return n_threads;
}

pub fn deinitThreadPool() void {
    g_thread_pool.allocator.free(g_thread_results);
    g_thread_pool.deinit();
    g_thread_pool_inited = false;
}

pub const tasaClas = tasaClasParallel;
// pub const tasaClas = tasaClasSequential;

pub fn tasaClasSequential(w: []const f64, test_set: []const Example, training_set: []const Example) f64 {
    // This is a pointer and length comparison. Doesn't work if test_set is a
    // *copy* of training_set, instead of the same slice, but works for now.
    const leave_one_out = std.meta.eql(test_set, training_set);

    // Classify every example in test_set using training_set and weights w
    var well_classified: usize = 0;
    for (test_set, 0..) |example, i| {
        const skip_idx = if (leave_one_out) i else null;
        const class = classifier1NN(example, training_set, skip_idx, w);
        if (std.mem.eql(u8, class, example.class))
            well_classified += 1;
    }

    return 100.0 * @intToFloat(f64, well_classified) / @intToFloat(f64, test_set.len);
}

// This can't be used together with the parallel version of getFitnesses
pub fn tasaClasParallel(w: []const f64, test_set: []const Example, training_set: []const Example) f64 {
    std.debug.assert(g_thread_pool_inited);

    // This is a pointer and length comparison. Doesn't work if test_set is a
    // *copy* of training_set, instead of the same slice, but works for now.
    const leave_one_out = std.meta.eql(test_set, training_set);

    var wait_group = std.Thread.WaitGroup{};
    var n_threads = g_thread_pool.threads.len;
    var size = test_set.len / n_threads;
    if (size == 0) {
        size = 1;
        n_threads = test_set.len;
    }

    for (g_thread_results[0..n_threads]) |*result| {
        result.* = std.math.maxInt(usize);
    }

    // Launch threads
    for (0..n_threads) |i| {
        const start_idx = size * i;
        const end_idx = if (i == n_threads - 1) test_set.len else size * (i + 1);
        // print("thread {}: {}-{}\n", .{ i, start_idx, end_idx });
        wait_group.start();
        g_thread_pool.spawn(workerTasaClas, .{
            w,
            test_set,
            training_set,
            leave_one_out,
            start_idx,
            end_idx,
            &wait_group,
            &g_thread_results[i],
        }) catch unreachable;
    }

    // Wait for them to finish
    g_thread_pool.waitAndWork(&wait_group);

    // Compute final result
    var well_classified: usize = 0;
    for (g_thread_results[0..n_threads]) |result| {
        std.debug.assert(result != std.math.maxInt(usize));
        well_classified += result;
    }
    return 100.0 * @intToFloat(f64, well_classified) / @intToFloat(f64, test_set.len);
}

fn workerTasaClas(
    w: []const f64,
    test_set: []const Example,
    training_set: []const Example,
    leave_one_out: bool,
    start_idx: usize,
    end_idx: usize,
    waiting_group: *std.Thread.WaitGroup,
    result: *usize,
) void {
    // Classify every example in test_set using training_set and weights w
    var well_classified: usize = 0;
    for (test_set[start_idx..end_idx], start_idx..end_idx) |example, i| {
        const skip_idx = if (leave_one_out) i else null;
        const class = classifier1NN(example, training_set, skip_idx, w);
        if (std.mem.eql(u8, class, example.class))
            well_classified += 1;
    }

    result.* = well_classified;
    waiting_group.finish();
}

pub fn tasaRed(weights: []const f64) f64 {
    var discarded: usize = 0;
    for (weights) |weight| {
        if (weight < MIN_WEIGHT) {
            discarded += 1;
        }
    }
    return 100.0 * @intToFloat(f64, discarded) / @intToFloat(f64, weights.len);
}

test "tasaRed 100" {
    const w = try std.testing.allocator.alloc(f64, 8);
    defer std.testing.allocator.free(w);
    @memset(w, 0);
    const tasa_red = tasaRed(w);
    try std.testing.expectEqual(@as(f64, 100), tasa_red);
    // print("{}\n", .{tasa_red});
}

pub fn getFitnesses(
    solutions: []const []const f64,
    training_set: []const Example,
    fitnesses: []f64,
) void {
    for (solutions, fitnesses) |w, *fitness| {
        fitness.* = getFitness(w, training_set, training_set);
    }
}

fn classifier1NN(e: Example, set: []const Example, skip_idx: ?usize, w: []const f64) []const u8 {
    var class_min: []const u8 = undefined;
    var dist_min = std.math.floatMax(f64);
    for (set, 0..) |example, i| {
        if (skip_idx == i)
            continue;
        const dist = e.distanceWeightedSimd(example, w);
        // const dist2 = e.distanceWeighted(example, w);
        // if (!std.math.approxEqRel(f64, dist, dist2, std.math.sqrt(std.math.floatEps(f64)))) {
        //     print("{d} {d}\n", .{dist, dist2});
        // }
        if (dist < dist_min) {
            dist_min = dist;
            class_min = example.class;
        }
    }
    return class_min;
}
