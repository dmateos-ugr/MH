const std = @import("std");
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
        const n = self.attributes.len;
        std.debug.assert(other.attributes.len == n);
        var ret: f64 = 0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const dist = self.attributes[i] - other.attributes[i];
            ret += dist * dist;
        }
        return ret;
    }

    fn distanceWeighted(self: Example, other: Example, w: []const f64) f64 {
        const n = self.attributes.len;
        std.debug.assert(other.attributes.len == n and w.len == n);
        var ret: f64 = 0;
        var i: usize = 0;
        while (i < self.attributes.len) : (i += 1) {
            if (w[i] < MIN_WEIGHT) continue;
            const dist = self.attributes[i] - other.attributes[i];
            ret += dist * dist * w[i];
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
            print("Error reading file: invalid example '{s}' (expected {} attributes + class, found {})\n", .{line, num_attributes, num_commas});
            return error.InvalidExample;
        }

        const attributes = try allocator.alloc(f64, num_attributes);
        var i: usize = 0;
        var iter = std.mem.tokenize(u8, line, ",");
        while (iter.next()) |token| : (i += 1) {
            // std.debug.print("token: {s}\n", .{token});
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
    var i: usize = 0;
    while (i < n) : (i += 1) {
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

pub fn fitness(w: []const f64) f64 {
    return ALPHA*tasa_clas(w) + (1 - ALPHA)*tasa_red(w);
}

fn tasa_clas(w: []const f64) f64 {
    _ = w;

}

fn tasa_red(weights: []const f64) f64 {
    var discarded: usize = 0;
    for (weights) |weight| {
        if (weight < MIN_WEIGHT) {
            discarded += 1;
        }
    }
    return 100 * discarded / weights.len;
}


fn classifier1NN(e: Example, set: []const Example, skip_idx: ?usize, comptime distanceFn: anytype, ctx: anytype) []const u8 {
    var c_min: []const u8 = undefined;
    var d_min = std.math.floatMax(f64);
    for (set) |example, i| {
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


pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    var rng = std.rand.DefaultPrng.init(0);
    const rnd = rng.random();

    var w: [8]f64 = undefined;
    for (w) |*weight| {
        weight.* = rnd.float(f64);
    }

    std.debug.print("{any}\n", .{w});


    const examples = try readExamplesFromFile("Instancias_APC/diabetes_1.arff", allocator);
    // const examples2 = try readExamplesFromFile("Instancias_APC/diabetes_2.arff", allocator);



    // _ = examples;
    for (examples) |example, i| {
        const result1 = classifier1NN(example, examples, i, Example.distanceWeighted, &w);
        const result2 = classifier1NN(example, examples, i, Example.distance, {});
        // _ = result1;
        // _ = result2;
        if (!std.mem.eql(u8, result1, example.class))
            std.debug.print("failed1 {}\n", .{i});
        if (!std.mem.eql(u8, result2, example.class))
            std.debug.print("failed2 {}\n", .{i});
    }


    for (examples) |example| {
        example.deinit(allocator);
    }
    allocator.free(examples);

}
