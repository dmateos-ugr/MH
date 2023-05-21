const std = @import("std");
const utils = @import("utils.zig");
const busquedaLocal = @import("bl.zig").busquedaLocal;
const Random = std.rand.Random;
const Allocator = std.mem.Allocator;
const Example = utils.Example;

const MAX_EVALUATIONS = 15000;
const ALPHA_BLX = 0.3;
const PROB_CROSS = 0.7;
const PROB_MUTATION = 0.1;
const PROB_BL = 0.1;
const N_POPULATION = 50;
const N_CROSS_PAIRS = @floatToInt(usize, PROB_CROSS * N_POPULATION / 2.0);
const N_MUTATIONS = @floatToInt(usize, PROB_MUTATION * N_POPULATION);
const N_BL = @floatToInt(usize, PROB_BL * N_POPULATION);
comptime {
    std.debug.assert(N_CROSS_PAIRS == 17);
    std.debug.assert(N_MUTATIONS == 5);
}


fn randFloatRange(min: f64, max: f64, rnd: Random) f64 {
    std.debug.assert(min <= max);
    const range_length = max - min;
    const ret = rnd.float(f64) * range_length + min;
    std.debug.assert(min <= ret and ret <= max);
    return ret;
}

const CrossFn = fn (w1: []f64, w2: []f64, rnd: Random) void;

fn crossBLX(w1: []f64, w2: []f64, rnd: Random) void {
    for (w1, w2) |*weight1, *weight2| {
        const min = std.math.min(weight1.*, weight2.*);
        const max = std.math.max(weight1.*, weight2.*);
        const explore_range = ALPHA_BLX * (max - min);
        const inf = std.math.max(min - explore_range, 0);
        const sup = std.math.min(max + explore_range, 1);
        weight1.* = randFloatRange(inf, sup, rnd);
        weight2.* = randFloatRange(inf, sup, rnd);
    }
}

fn crossArithmetic(w1: []f64, w2: []f64, rnd: Random) void {
    const alpha = rnd.float(f64);
    const one_minus_alpha = 1 - alpha;
    for (w1, w2) |*weight1, *weight2| {
        weight1.* = alpha * weight1.* + one_minus_alpha * weight2.*;
        weight2.* = alpha * weight2.* + one_minus_alpha * weight1.*;
    }
}

pub fn AGG_BLX(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    return agg(training_set, allocator, rnd, crossBLX, .None);
}

pub fn AGG_Arit(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    return agg(training_set, allocator, rnd, crossArithmetic, .None);
}

pub fn AM_All(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    return agg(training_set, allocator, rnd, crossBLX, .All);
}

pub fn AM_Rand(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    return agg(training_set, allocator, rnd, crossBLX, .Rand);
}

pub fn AM_Best(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    return agg(training_set, allocator, rnd, crossBLX, .Best);
}

fn binaryTournament(fitnesses: *const [N_POPULATION]f64, rnd: Random) usize {
    const idx1 = rnd.uintLessThan(usize, fitnesses.len);
    var idx2 = rnd.uintLessThan(usize, fitnesses.len);
    while (idx1 == idx2)
        idx2 = rnd.uintLessThan(usize, fitnesses.len);
    const idx_win = if (fitnesses[idx1] > fitnesses[idx2]) idx1 else idx2;
    return idx_win;
}

/// Populate `new_population` using binary tournament.
fn aggSeleccion(
    population: *const [N_POPULATION][]f64,
    fitnesses: *const [N_POPULATION]f64,
    new_population: *const [N_POPULATION][]f64,
    rnd: Random,
) void {
    for (new_population) |element| {
        const i = binaryTournament(fitnesses, rnd);
        @memcpy(element, population[i]);
    }
}

/// Cross `N_CROSS_PAIRS` in `new_population` using `cross_fn`.
fn aggCruce(
    new_population: *const [N_POPULATION][]f64,
    comptime cross_fn: CrossFn,
    rnd: Random,
) void {
    for (0..N_CROSS_PAIRS) |i| {
        const element1 = new_population[2 * i];
        const element2 = new_population[2 * i + 1];
        cross_fn(element1, element2, rnd);
    }
}

/// Mutate `N_MUTATIONS` elements of `new_population`.
fn aggMutacion(
    new_population: *const [N_POPULATION][]f64,
    rnd: Random,
) void {
    const n = new_population[0].len;
    for (0..N_MUTATIONS) |_| {
        const i = rnd.uintLessThan(usize, new_population.len);
        const i_weight = rnd.uintLessThan(usize, n);
        utils.mov(new_population[i], i_weight, rnd);
    }
}

/// Replaces worst solution in `new_population` with best solution in
/// `population`, updating its fitness.
fn aggElitismo(
    population: *const [N_POPULATION][]f64,
    fitnesses: *const [N_POPULATION]f64,
    new_population: *const [N_POPULATION][]f64,
    new_fitnesses: *[N_POPULATION]f64,
) void {
    // Find index of solution with min fitness in `new_population`, and max
    // fitness in `population`
    var new_min_fitness_idx: usize = 0;
    var old_max_fitness_idx: usize = 0;
    for (new_fitnesses, fitnesses, 0..) |new_fitness, fitness, i| {
        if (new_fitness < new_fitnesses[new_min_fitness_idx])
            new_min_fitness_idx = i;
        if (fitness > fitnesses[old_max_fitness_idx])
            old_max_fitness_idx = i;
    }

    // If the worst in `new_population` is better than the best in `population`,
    // don't do anything
    if (new_fitnesses[new_min_fitness_idx] >= fitnesses[old_max_fitness_idx])
        return;

    // Replace the worst in `new_population` with the best in `population`
    @memcpy(new_population[new_min_fitness_idx], population[old_max_fitness_idx]);
    new_fitnesses[new_min_fitness_idx] = fitnesses[old_max_fitness_idx];
}

const MemeticType = enum {
    None,
    All,
    Rand,
    Best,
};

/// Perform local search on a subset of `population` according to
/// `memetic_type`, updating `fitnesses` and `evaluations`.
fn memeticBusquedaLocal(
    training_set: []const Example,
    population: *[N_POPULATION][]f64,
    fitnesses: *[N_POPULATION]f64,
    evaluations: *usize,
    comptime memetic_type: MemeticType,
    allocator: Allocator,
    rnd: Random,
) !void {
    const subset_size = switch (memetic_type) {
        .All => N_POPULATION,
        .Rand, .Best => N_BL,
        .None => unreachable,
    };

    if (memetic_type == .Best) {
        sortPopulationBestFirst(population, fitnesses);
    } else if (memetic_type == .Rand) {
        // Place subset_size random items at the beginning. Those are the ones
        // we'll be performing local search on.
        for (0..subset_size) |i| {
            const i_rnd = rnd.uintLessThan(usize, population.len);
            std.mem.swap(f64, &fitnesses[i], &fitnesses[i_rnd]);
            std.mem.swap([]f64, &population[i], &population[i_rnd]);
        }
    }

    // Perform local search on the first subset_size elements, updating
    // fitnesses and evaluations
    for (population[0..subset_size], fitnesses[0..subset_size]) |element, *fitness| {
        const result = try busquedaLocal(element, training_set, allocator, rnd, .{
            .max_iters = std.math.maxInt(usize), // unlimited
            .max_neighbours_per_attribute = 2,
        });
        std.debug.assert(fitness.* <= result.fitness);
        fitness.* = result.fitness;
        evaluations.* += result.evaluations;
        if (evaluations.* >= MAX_EVALUATIONS)
            break;
    }
}

fn agg(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
    comptime cross_fn: CrossFn,
    comptime memetic_type: MemeticType,
) ![]const f64 {
    const n = training_set[0].attributes.len;

    var population1: [N_POPULATION][]f64 = undefined;
    var population2: [N_POPULATION][]f64 = undefined;
    var fitnesses1: [N_POPULATION]f64 = undefined;
    var fitnesses2: [N_POPULATION]f64 = undefined;

    var population = &population1;
    var fitnesses = &fitnesses1;
    var new_population = &population2;
    var new_fitnesses = &fitnesses2;

    var generations: usize = 1;

    // Allocate memory for populations, initializing the current one randomly
    for (population, new_population) |*element1, *element2| {
        element1.* = try utils.createRandomSolution(n, allocator, rnd);
        element2.* = try allocator.alloc(f64, n);
    }
    defer for (population, new_population) |element1, element2| {
        allocator.free(element1);
        allocator.free(element2);
    };

    // Get fitness of current population
    utils.getFitnesses(population, training_set, fitnesses);

    var evaluations: usize = population.len; // we have already done one round of evaluations
    while (evaluations < MAX_EVALUATIONS) {
        // Seleccion: populates `new_population` with elements of `population`
        aggSeleccion(population, fitnesses, new_population, rnd);

        // Cruce: crosses elements of `new_population`
        aggCruce(new_population, cross_fn, rnd);

        // Mutacion: mutates elements of `new_population`
        aggMutacion(new_population, rnd);

        // Evaluate `new_population`
        utils.getFitnesses(new_population, training_set, new_fitnesses);
        evaluations += new_population.len;

        // Local search for memetic algorithms: improves a subset of the
        // solutions in `new_population` using local search
        if (memetic_type != .None and (generations % 10) == 0) {
            try memeticBusquedaLocal(
                training_set,
                new_population,
                new_fitnesses,
                &evaluations,
                memetic_type,
                allocator,
                rnd,
            );
        }

        // Elitismo: replace worst solution in `new_population` with best
        // solution in `population`
        aggElitismo(population, fitnesses, new_population, new_fitnesses);

        // Reemplazamiento: make new_population the current population
        std.mem.swap(*[N_POPULATION][]f64, &population, &new_population);
        std.mem.swap(*[N_POPULATION]f64, &fitnesses, &new_fitnesses);

        if (comptime memetic_type != .None) generations += 1;
    }

    const max_fitness_idx = std.mem.indexOfMax(f64, fitnesses);
    return allocator.dupe(f64, population[max_fitness_idx]);
}

pub fn AGE_BLX(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    return age(training_set, allocator, rnd, crossBLX);
}

pub fn AGE_Arit(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    return age(training_set, allocator, rnd, crossArithmetic);
}

const SortContext = struct {
    population: [][]const f64,
    fitnesses: []f64,

    pub fn lessThan(self: SortContext, idx1: usize, idx2: usize) bool {
        return self.fitnesses[idx1] > self.fitnesses[idx2];
    }

    pub fn swap(self: *SortContext, idx1: usize, idx2: usize) void {
        std.mem.swap([]const f64, &self.population[idx1], &self.population[idx2]);
        std.mem.swap(f64, &self.fitnesses[idx1], &self.fitnesses[idx2]);
    }
};

fn sortPopulationBestFirst(population: [][]const f64, fitnesses: []f64) void {
    var sort_context = SortContext{ .population = population, .fitnesses = fitnesses };
    std.sort.insertionSortContext(population.len, &sort_context);
    std.debug.assert(std.sort.isSorted(f64, fitnesses, {}, std.sort.desc(f64)));
}

test "sortPopulationBestFirst" {
    const allocator = std.testing.allocator;
    var rng = std.rand.DefaultPrng.init(0);
    const rnd = rng.random();

    _ = try utils.initThreadPool(allocator, null);
    defer utils.deinitThreadPool();
    const partitions = try utils.readPartitions("ozone-320", allocator);
    defer utils.freePartitions(partitions, allocator);
    const dataset = try std.mem.concat(allocator, Example, &partitions);
    defer allocator.free(dataset);

    const n = dataset[0].attributes.len;

    var population: [N_POPULATION][]f64 = undefined;
    var fitnesses: [N_POPULATION]f64 = undefined;
    var population_unordered: [N_POPULATION][]f64 = undefined;
    var fitnesses_unordered: [N_POPULATION]f64 = undefined;

    // Get random population, and assign fitness from 0 to N_POPULATION - 1
    for (&population) |*element| {
        element.* = try utils.createRandomSolution(n, allocator, rnd);
    }
    defer for (population) |element| {
        allocator.free(element);
    };
    for (&fitnesses, 0..) |*fitness, i| {
        fitness.* = @intToFloat(f64, i);
    }

    // Copy
    @memcpy(&population_unordered, &population);
    @memcpy(&fitnesses_unordered, &fitnesses);

    // Shuffle population
    for (0..population.len) |idx1| {
        const idx2 = rnd.uintLessThan(usize, population.len);
        std.mem.swap([]f64, &population[idx1], &population[idx2]);
        std.mem.swap(f64, &fitnesses[idx1], &fitnesses[idx2]);
    }

    // Sort it
    sortPopulationBestFirst(&population, &fitnesses);

    // Check that fitnesses are ordered, and that populations have been switched too
    for (0..fitnesses.len - 1) |i| {
        try std.testing.expect(fitnesses[i] >= fitnesses[i + 1]);
        const value = fitnesses.len - i - 1;
        try std.testing.expect(fitnesses[i] == @intToFloat(f64, value));
        try std.testing.expect(std.mem.eql(f64, population[i], population_unordered[value]));
    }
}

/// Populate `two_elements` using binary tournament twice.
fn ageSeleccion(
    population: *const [N_POPULATION][]f64,
    fitnesses: *const [N_POPULATION]f64,
    two_elements: *const [2][]f64,
    rnd: Random,
) void {
    for (two_elements) |element| {
        const idx_win = binaryTournament(fitnesses, rnd);
        @memcpy(element, population[idx_win]);
    }
}

/// Mutate `two_elements` with a probability of `PROB_MUTATION`.
fn ageMutacion(
    two_elements: *const [2][]f64,
    rnd: Random,
) void {
    const n = two_elements[0].len;
    for (two_elements) |element| {
        if (rnd.float(f64) < PROB_MUTATION) {
            const i_weight = rnd.uintLessThan(usize, n);
            utils.mov(element, i_weight, rnd);
        }
    }
}

/// Keep best two among `two_elements` and worst two elements of `population`.
fn ageReemplazamiento(
    population: *[N_POPULATION][]f64,
    fitnesses: *[N_POPULATION]f64,
    two_elements: *const [2][]f64,
    training_set: []const Example,
) void {
    var two_fitnesses: [2]f64 = undefined;
    utils.getFitnesses(two_elements, training_set, &two_fitnesses);

    // Sort `two elements` and worst two of `population`. Population is in
    // descending order according to their fitnesses, so worst two are last two.
    var four_elements = [4][]f64{
        population[population.len - 1],
        population[population.len - 2],
        two_elements[0],
        two_elements[1],
    };
    var four_fitnesses = [4]f64{
        fitnesses[fitnesses.len - 1],
        fitnesses[fitnesses.len - 2],
        two_fitnesses[0],
        two_fitnesses[1],
    };
    sortPopulationBestFirst(&four_elements, &four_fitnesses);

    // Replace worst two of `population` with best two of `four_elements`, and
    // keep population sorted.
    std.mem.copyForwards(f64, population[population.len - 1], four_elements[0]);
    std.mem.copyForwards(f64, population[population.len - 2], four_elements[1]);
    fitnesses[population.len - 1] = four_fitnesses[0];
    fitnesses[population.len - 2] = four_fitnesses[1];
    sortPopulationBestFirst(population, fitnesses);
}

fn age(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
    comptime cross_fn: CrossFn,
) ![]const f64 {
    const n = training_set[0].attributes.len;

    var population: [N_POPULATION][]f64 = undefined;
    var fitnesses: [N_POPULATION]f64 = undefined;
    var two_elements: [2][]f64 = undefined;

    // Allocate memory for the population, initializing it randomly
    for (&population) |*element| {
        element.* = try utils.createRandomSolution(n, allocator, rnd);
    }
    defer for (population) |element| {
        allocator.free(element);
    };
    for (&two_elements) |*element| {
        element.* = try allocator.alloc(f64, n);
    }
    defer for (two_elements) |element| {
        allocator.free(element);
    };

    // Get fitness of current population and sort it according to it
    utils.getFitnesses(&population, training_set, &fitnesses);
    sortPopulationBestFirst(&population, &fitnesses);

    var evaluations: usize = population.len; // we have already done one round of evaluations
    while (evaluations < MAX_EVALUATIONS) {
        // Seleccion: populates `two_elements`
        ageSeleccion(&population, &fitnesses, &two_elements, rnd);

        // Cruce: cross `two_elements`
        cross_fn(two_elements[0], two_elements[1], rnd);

        // Mutacion
        ageMutacion(&two_elements, rnd);

        // Reemplazamiento: keep best two among `two_elements` and worst two of
        // `population`
        ageReemplazamiento(&population, &fitnesses, &two_elements, training_set);
        evaluations += 2;
    }

    // Return the element with highest fitness
    std.debug.assert(std.mem.indexOfMax(f64, &fitnesses) == 0);
    return allocator.dupe(f64, population[0]);
}
