const std = @import("std");
const Random = std.rand.Random;
const Allocator = std.mem.Allocator;

const Example = struct {
    attributes: []f64,
    class: []const u8,

    pub fn init(attributes: []f64, class: []const u8) !Example {
        return Example{
            .attributes = attributes,
            .class = class,
        };
    }

    fn deinit(self: Example, allocator: Allocator) void {
        allocator.free(self.attributes);
        allocator.free(self.class);
    }

    fn distance(self: Example, other: Example, _: void) f64 {
        var ret: f64 = 0;
        for (self.attributes, other.attributes) |attr1, attr2| {
            const dist = attr1 - attr2;
            ret += dist * dist;
        }
        return ret;
    }

    fn distanceWeighted(self: Example, other: Example, w: []const f64) f64 {
        var ret: f64 = 0;
        for (self.attributes, other.attributes, w) |attr1, attr2, weight| {
            if (weight < MIN_WEIGHT) continue;
            const dist = attr1 - attr2;
            ret += dist * dist * weight;
        }
        return ret;
    }
};

const MIN_WEIGHT = 0.1;

pub fn print(comptime format: []const u8, args: anytype) void {
    const stdout = std.io.getStdOut().writer();
    stdout.print(format, args) catch unreachable;
}

pub fn readExamplesFromFile(filename: []const u8, allocator: Allocator) ![]Example {
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

    normalizeExamples(examples.items);

    return examples.toOwnedSlice();
}

fn normalizeExamples(examples: []Example) void {
    const n = examples[0].attributes.len;
    for (0..n) |i| {
        // Calculate max and min of the attribute
        var min = examples[0].attributes[i];
        var max = min;
        for (examples[1..]) |example| {
            min = std.math.min(min, example.attributes[i]);
            max = std.math.max(max, example.attributes[i]);
        }

        // Normalize attribute in every example
        for (examples) |example| {
            const new_value = (example.attributes[i] - min) / (max - min);
            std.debug.assert(0 <= new_value and new_value <= 1);
            example.attributes[i] = new_value;
        }
    }
}

const ALPHA = 0.8;

pub fn getFitness(w: []const f64, test_set: []const Example, training_set: []const Example) f64 {
    return ALPHA * tasaClas(w, test_set, training_set) + (1 - ALPHA) * tasaRed(w);
}

fn tasaClas(w: []const f64, test_set: []const Example, training_set: []const Example) f64 {
    // Classify every example in test_set using training_set and weights w
    const leave_one_out = std.meta.eql(test_set, training_set);

    var well_classified: usize = 0;
    for (test_set, 0..) |example, i| {
        const skip_idx = if (leave_one_out) i else null;
        const class = classifier1NN(example, training_set, skip_idx, Example.distanceWeighted, w);
        if (std.mem.eql(u8, class, example.class))
            well_classified += 1;
    }

    return 100.0 * @intToFloat(f64, well_classified) / @intToFloat(f64, test_set.len);
}

fn tasaRed(weights: []const f64) f64 {
    var discarded: usize = 0;
    for (weights) |weight| {
        if (weight < MIN_WEIGHT) {
            discarded += 1;
        }
    }
    return 100.0 * @intToFloat(f64, discarded) / @intToFloat(f64, weights.len);
}

fn classifier1NN(e: Example, set: []const Example, skip_idx: ?usize, comptime distanceFn: anytype, ctx: anytype) []const u8 {
    var c_min: []const u8 = undefined;
    var d_min = std.math.floatMax(f64);
    for (set, 0..) |example, i| {
        if (skip_idx != null and skip_idx.? == i)
            continue;
        const d = distanceFn(e, example, ctx);
        if (d < d_min) {
            d_min = d;
            c_min = example.class;
        }
    }
    return c_min;
}

const MOV_VARIANCE = 0.3;
const MOV_STDDEV = std.math.sqrt(MOV_VARIANCE);

fn mov(w: []f64, i: usize, rnd: Random) void {
    const z_i = rndNorm(rnd, 0, MOV_STDDEV);
    const result = w[i] + z_i;
    const result_truncated = std.math.max(0, std.math.min(1, result));
    w[i] = result_truncated;
}

fn rndNorm(rnd: Random, mean: f64, stddev: f64) f64 {
    return rnd.floatNorm(f64) * stddev + mean;
}

fn busquedaLocal(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    const n = training_set[0].attributes.len;

    // First solution
    const w = try allocator.alloc(f64, n);
    errdefer allocator.free(w);
    for (w) |*weight| {
        weight.* = rnd.float(f64); // range 0, 1
    }
    var current_fitness = getFitness(w, training_set, training_set);

    // Mutation of w
    const w_mut = try allocator.alloc(f64, n);
    defer allocator.free(w_mut);

    // Indexes
    const indexes = try allocator.alloc(usize, n);
    defer allocator.free(indexes);
    for (indexes, 0..) |*index, i| {
        index.* = i;
    }
    rnd.shuffle(usize, indexes);

    var neighbours: usize = 0;
    var iter: usize = 0;
    while (neighbours < 20 * n and iter < 15000) : ({
        iter += 1;
        neighbours += 1;
    }) {
        // Mutate w into w_mut
        std.mem.copy(f64, w_mut, w);
        mov(w_mut, indexes[iter % n], rnd);

        // Evaluate w_mut, classifying every example in training_set using leave-one-out
        const fitness = getFitness(w_mut, training_set, training_set);
        if (fitness > current_fitness) {
            // Replace w with w_mut
            current_fitness = fitness;
            std.mem.copy(f64, w, w_mut);
            rnd.shuffle(usize, indexes);
            neighbours = 0;
        }
    }

    // print("iterations: {}, neighbours: {}\n", .{iter, neighbours});

    return w;
}

fn abs(f: f64) f64 {
    return if (f < 0) -f else f;
}

fn greedy(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    _ = rnd; // rnd is unused, since greedy is deterministic

    const n = training_set[0].attributes.len;

    // First solution
    const w = try allocator.alloc(f64, n);
    errdefer allocator.free(w);
    std.mem.set(f64, w, 0);

    for (training_set, 0..) |example, i_example| {
        // Get the closest enemy and closest friend to `example`
        var closest_enemy_i: usize = 0;
        var closest_friend_i: usize = 0;
        var closest_enemy_distance = std.math.floatMax(f64);
        var closest_friend_distance = std.math.floatMax(f64);
        for (training_set, 0..) |other, i| {
            if (i == i_example) continue;
            const dist = example.distance(other, {});
            if (std.mem.eql(u8, example.class, other.class)) {
                if (dist < closest_friend_distance) {
                    closest_friend_distance = dist;
                    closest_friend_i = i;
                }
            } else {
                if (dist < closest_enemy_distance) {
                    closest_enemy_distance = dist;
                    closest_enemy_i = i;
                }
            }
        }

        // Add the difference to the closest enemy and substract the difference
        // to the closest friend
        const enemy_attrs = training_set[closest_enemy_i].attributes;
        const friend_attrs = training_set[closest_friend_i].attributes;
        for (w, example.attributes, enemy_attrs, friend_attrs) |*weight, attr, enemy_attr, friend_attr| {
            weight.* += abs(attr - enemy_attr) - abs(attr - friend_attr);
        }
    }

    // Normalize w
    const w_max = std.mem.max(f64, w);
    for (w) |*weight| {
        weight.* = if (weight.* < 0) 0 else weight.* / w_max;
    }

    return w;
}

fn joinPartitions(partitions: [5][]const Example, skip_idx: usize, allocator: Allocator) ![]Example {
    var used_partitions: [4][]const Example = undefined;
    for (0..partitions.len) |i| {
        if (i == skip_idx) continue;
        const i_save = if (i < skip_idx) i else i - 1;
        used_partitions[i_save] = partitions[i];
    }

    return try std.mem.concat(allocator, Example, &used_partitions);
}

fn readPartitions(comptime name: []const u8, allocator: Allocator) ![5][]Example {
    var examples_partitions: [5][]Example = undefined;
    inline for (&examples_partitions, 0..) |*examples, i| {
        const path = std.fmt.comptimePrint("Instancias_APC/{s}_{}.arff", .{ name, i + 1 });
        examples.* = try readExamplesFromFile(path, allocator);
    }
    return examples_partitions;
}

const RNG_SEED = 16;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var rng = std.rand.DefaultPrng.init(RNG_SEED);
    const rnd = rng.random();

    const partitions = try readPartitions("diabetes", allocator);
    defer {
        for (partitions) |partition| {
            for (partition) |example| {
                example.deinit(allocator);
            }
            allocator.free(partition);
        }
    }

    for (0..5) |i| {
        print("[PARTICION {}]\n", .{i+1});
        const training_set = try joinPartitions(partitions, i, allocator);
        defer allocator.free(training_set);
        const test_set = partitions[i];
        print("training: {}, test: {} \n", .{training_set.len, test_set.len});
        for (0..5) |j| {
            print("{d} {s}\n", .{training_set[j].attributes, training_set[j].class});
        }
        _ = rnd;

        // const w_bl = try busquedaLocal(training_set, allocator, rnd);
        // defer allocator.free(w_bl);
        // const fitness_bl = getFitness(w_bl, test_set, training_set);
        // print("BL: fitness {d}, tasa_red {d}, tasaClas {d}\n", .{
        //     fitness_bl,
        //     tasaRed(w_bl),
        //     tasaClas(w_bl, test_set, training_set),
        // });

        // const w_greedy = try greedy(training_set, allocator, rnd);
        // defer allocator.free(w_greedy);
        // const fitness_greedy = getFitness(w_greedy, test_set, training_set);
        // print("GREEDY: fitness {d}, tasa_red {d}, tasaClas {d}\n", .{
        //     fitness_greedy,
        //     tasaRed(w_greedy),
        //     tasaClas(w_greedy, test_set, training_set),
        // });

        // print("w greedy: ", .{});
        // for (w_greedy) |weight| {
        //     print("{d} ", .{weight});
        // }
        // print("\n", .{});

        print("\n", .{});
    }

    // std.mem.copy(Example, examples, examples1);
    // std.mem.copy(Example, examples[examples1.len..], examples2);
    // std.mem.copy(Example, examples[examples1.len + examples2.len..], examples3);
    // std.mem.copy(Example, examples[examples1.len + examples2.len + examples3.len..], examples4);

    // _ = w;

}
