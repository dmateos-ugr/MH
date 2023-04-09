const std = @import("std");
const Random = std.rand.Random;
const Allocator = std.mem.Allocator;

// Threshold for considering an attribute as discarded
const MIN_WEIGHT = 0.1;

// Alpha used in: fitness = alpha·tasa_clas + (1-alpha)·tasa_red
const ALPHA = 0.8;

const N_PARTITIONS = 5;

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
        var ret: f64 = 0;
        for (self.attributes, other.attributes, w) |attr1, attr2, weight| {
            if (weight < MIN_WEIGHT) continue;
            const dist = attr1 - attr2;
            ret += dist * dist * weight;
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

pub fn getFitness(w: []const f64, test_set: []const Example, training_set: []const Example) f64 {
    return ALPHA * tasaClas(w, test_set, training_set) + (1 - ALPHA) * tasaRed(w);
}

pub fn tasaClas(w: []const f64, test_set: []const Example, training_set: []const Example) f64 {
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

pub fn tasaRed(weights: []const f64) f64 {
    var discarded: usize = 0;
    for (weights) |weight| {
        if (weight < MIN_WEIGHT) {
            discarded += 1;
        }
    }
    return 100.0 * @intToFloat(f64, discarded) / @intToFloat(f64, weights.len);
}

fn classifier1NN(e: Example, set: []const Example, skip_idx: ?usize, w: []const f64) []const u8 {
    var class_min: []const u8 = undefined;
    var dist_min = std.math.floatMax(f64);
    for (set, 0..) |example, i| {
        if (skip_idx != null and skip_idx.? == i)
            continue;
        const dist = e.distanceWeighted(example, w);
        if (dist < dist_min) {
            dist_min = dist;
            class_min = example.class;
        }
    }
    return class_min;
}
