const std = @import("std");
const utils = @import("utils.zig");
const Random = std.rand.Random;
const Allocator = std.mem.Allocator;
const Example = utils.Example;

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

pub fn busquedaLocalP1(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    const n = training_set[0].attributes.len;
    const w = try allocator.alloc(f64, n);
    errdefer allocator.free(w);
    for (w) |*weight| {
        weight.* = rnd.float(f64); // range [0, 1)
    }

    _ = try busquedaLocal(w, training_set, allocator, rnd, .{
        .max_iters = 15000,
        .max_neighbours_per_attribute = 20,
    });

    return w;
}

const LocalSearchConfig = struct {
    max_iters: usize,
    max_neighbours_per_attribute: usize,
};

// Performs local search with `w` as initial solution. The final solution is
// held in `w`. Returns the number of iterations performed.
pub fn busquedaLocal(
    w: []f64,
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
    config: LocalSearchConfig, // TODO probar a poner comptime
) !usize {
    const n = w.len;
    const max_neighbours = config.max_neighbours_per_attribute * n;

    // First solution
    var current_fitness = utils.getFitness(w, training_set, training_set);

    // Mutation of w
    const w_mut = try allocator.alloc(f64, n);
    defer allocator.free(w_mut);

    // Indexes that we'll be mutating on each iteration
    var indexes = try IndexIterator.init(allocator, n, rnd);
    defer indexes.deinit();

    var neighbours: usize = 0;
    var iter: usize = 0;
    while (neighbours < max_neighbours and iter < config.max_iters) : (iter += 1) {
        // Mutate w into w_mut
        std.mem.copy(f64, w_mut, w);
        utils.mov(w_mut, indexes.next(), rnd);

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

    // utils.print("iterations: {}, neighbours: {}\n", .{ iter, neighbours });

    return iter;
}
