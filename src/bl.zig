const std = @import("std");
const utils = @import("utils.zig");
const Random = std.rand.Random;
const Allocator = std.mem.Allocator;
const Example = utils.Example;

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

// Auxiliar struct to keep track of which components of the weight we are
// mutating. Indexes should be shuffled when we have iterated all of them, or
// when the mutation yielded a higher fitness.
const IndexIterator = struct {
    allocator: Allocator,
    indexes: []usize,
    rnd: Random,
    i: usize,

    pub fn init(allocator: Allocator, n: usize, rnd: Random) !IndexIterator {
        const indexes = try allocator.alloc(usize, n);
        for (indexes, 0..) |*index, i| {
            index.* = i;
        }
        rnd.shuffle(usize, indexes);
        return IndexIterator{
            .allocator = allocator,
            .indexes = indexes,
            .rnd = rnd,
            .i = 0,
        };
    }

    pub fn deinit(self: *IndexIterator) void {
        self.allocator.free(self.indexes);
        self.* = undefined;
    }

    pub fn next(self: *IndexIterator) usize {
        const ret = self.indexes[self.i];
        self.i += 1;
        if (self.i == self.indexes.len)
            self.reset();
        return ret;
    }

    pub fn reset(self: *IndexIterator) void {
        self.rnd.shuffle(usize, self.indexes);
        self.i = 0;
    }
};

pub fn busquedaLocal(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    const n = training_set[0].attributes.len;

    // First solution
    const w = try allocator.alloc(f64, n);
    errdefer allocator.free(w);
    for (w) |*weight| {
        weight.* = rnd.float(f64); // range [0, 1)
    }
    var current_fitness = utils.getFitness(w, training_set, training_set);

    // Mutation of w
    const w_mut = try allocator.alloc(f64, n);
    defer allocator.free(w_mut);

    // Indexes that we'll be mutating on each iteration
    var indexes = try IndexIterator.init(allocator, n, rnd);
    defer indexes.deinit();

    var neighbours: usize = 0;
    var iter: usize = 0;
    while (neighbours < 20 * n and iter < 15000) : (iter += 1) {
        // Mutate w into w_mut
        std.mem.copy(f64, w_mut, w);
        mov(w_mut, indexes.next(), rnd);

        // Evaluate w_mut, classifying every example in training_set using leave-one-out
        const fitness = utils.getFitness(w_mut, training_set, training_set);
        if (fitness > current_fitness) {
            // Replace w with w_mut
            current_fitness = fitness;
            std.mem.copy(f64, w, w_mut);
            indexes.reset();
            neighbours = 0;
        } else neighbours += 1;
    }

    // print("iterations: {}, neighbours: {}\n", .{ iter, neighbours });

    return w;
}
