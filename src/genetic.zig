const std = @import("std");
const utils = @import("utils.zig");
const Random = std.rand.Random;
const Allocator = std.mem.Allocator;
const Example = utils.Example;

fn randFloatRange(min: f64, max: f64, rnd: Random) f64 {
    std.debug.assert(min <= max);
    const range_length = max - min;
    const ret = rnd.float(f64) * range_length + min;
    std.debug.assert(min <= ret and ret <= max);
    return ret;
}

const CrossFn = fn (w1: []f64, w2: []f64, rnd: Random) void;

// TODO perf probar bucle con indice
const ALPHA_BLX = 0.3;
fn crossBLX(w1: []f64, w2: []f64, rnd: Random) void {
    for (w1, w2) |*weight1, *weight2| {
        const min = std.math.min(weight1.*, weight2.*);
        const max = std.math.max(weight1.*, weight2.*);
        const explore_range = ALPHA_BLX * (max - min);
        const inf = min - explore_range;
        const sup = max + explore_range;
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

const N_POPULATION = 50;
const N_CROSS_PAIRS = @floatToInt(usize, 0.7 * N_POPULATION / 2.0);
const N_MUTATIONS = @floatToInt(usize, 0.1 * N_POPULATION);
comptime {
    std.debug.assert(N_CROSS_PAIRS == 17);
    std.debug.assert(N_MUTATIONS == 5);
}

pub fn AGG_BLX(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    return agg_aux(training_set, allocator, rnd, crossBLX);
}

pub fn AGG_Arit(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    return agg_aux(training_set, allocator, rnd, crossArithmetic);
}

// TODO modificar bl para que use esto?
const WeightFitness = struct {
    w: []f64,
    fitness: f64,

    fn initRandom(training_set: []const Example, allocator: Allocator, rnd: Random) !WeightFitness {
        const n = training_set[0].attributes.len;
        const w = try utils.createRandomSolution(n, allocator, rnd);
        return init(w, training_set);
    }

    // fn initZero(training_set: []const Example, allocator: Allocator) !WeightFitness {
    //     const n = training_set[0].attributes.len;
    //     const w = try allocator.alloc(f64, n);
    //     for (w) |*weight| {
    //         weight.* = 0;
    //     }
    //     return init(w, training_set);
    // }

    fn init(w: []f64, training_set: []const Example) WeightFitness {
        return .{
            .w = w,
            .fitness = utils.getFitness(w, training_set, training_set),
        };
    }

    fn deinit(self: WeightFitness, allocator: Allocator) void {
        allocator.free(self.w);
    }

    fn updateFitness(self: *WeightFitness, training_set: []const Example) void {
        self.fitness = utils.getFitness(self.w, training_set, training_set);
    }
};

fn agg_aux2(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
    comptime cross_fn: CrossFn,
) ![]const f64 {
    const n = training_set[0].attributes.len;

    var population: [N_POPULATION]WeightFitness = undefined;
    var new_population: [N_POPULATION]WeightFitness = undefined;
    for (&population) |*element| {
        element.* = try WeightFitness.initRandom(training_set, allocator, rnd);
    }
    // TODO defer

    var evaluations: usize = 0;
    while (evaluations < 15000) : (evaluations += new_population.len) {
        // Seleccion
        for (&new_population) |*element| {
            const idx1 = rnd.uintLessThan(usize, population.len);
            const idx2 = rnd.uintLessThan(usize, population.len);
            element.* = if (population[idx1].fitness > population[idx2].fitness)
                population[idx1]
            else
                population[idx2];
        }

        // Cruce
        for (0..N_CROSS_PAIRS) |i| {
            const element1 = new_population[2 * i];
            const element2 = new_population[2 * i + 1];
            cross_fn(element1.w, element2.w, rnd);
        }

        // Mutacion
        for (0..N_MUTATIONS) |_| {
            const i = rnd.uintLessThan(usize, new_population.len);
            const i_weight = rnd.uintLessThan(usize, n);
            utils.mov(new_population[i].w, i_weight, rnd);
        }

        // Reemplazamiento
        var new_min_fitness_idx: usize = 0;
        var old_max_fitness_idx: usize = 0;
        for (&new_population, &population, 0..) |*new_element, *element, i| {
            // TODO probablemente sea mejor un array de fitness y a volar
            new_element.updateFitness(training_set);
            if (new_element.fitness < new_population[new_min_fitness_idx].fitness)
                new_min_fitness_idx = i;

            if (element.fitness > population[old_max_fitness_idx].fitness)
                old_max_fitness_idx = i;
        }

        new_population[new_min_fitness_idx] = population[old_max_fitness_idx];
        std.mem.copy(WeightFitness, &population, &new_population);

        // utils.print("{}\n", .{evaluations});
    }

    var max_fitness_idx: usize = 0;
    for (population, 0..) |element, i| {
        if (element.fitness > population[max_fitness_idx].fitness)
            max_fitness_idx = i;
    }

    return population[max_fitness_idx].w;
}


fn agg_aux(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
    comptime cross_fn: CrossFn,
) ![]const f64 {
    const n = training_set[0].attributes.len;

    var population: [N_POPULATION][]f64 = undefined;
    var fitnesses: [N_POPULATION]f64 = undefined;
    var new_population: [N_POPULATION][]f64 = undefined;
    var new_fitnesses: [N_POPULATION]f64 = undefined;

    // Initial population
    for (&population, &fitnesses) |*element, *fitness| {
        element.* = try utils.createRandomSolution(n, allocator, rnd);
        fitness.* = utils.getFitness(element.*, training_set, training_set);
    }
    // TODO defer

    var evaluations: usize = 0;
    while (evaluations < 15000) : (evaluations += new_population.len) {
        // Seleccion
        for (&new_population) |*element| {
            const idx1 = rnd.uintLessThan(usize, population.len);
            const idx2 = rnd.uintLessThan(usize, population.len);
            element.* = if (fitnesses[idx1] > fitnesses[idx2])
                population[idx1]
            else
                population[idx2];
        }

        // Cruce
        for (0..N_CROSS_PAIRS) |i| {
            const element1 = new_population[2 * i];
            const element2 = new_population[2 * i + 1];
            cross_fn(element1, element2, rnd);
        }

        // Mutacion
        for (0..N_MUTATIONS) |_| {
            const i = rnd.uintLessThan(usize, new_population.len);
            const i_weight = rnd.uintLessThan(usize, n);
            utils.mov(new_population[i], i_weight, rnd);
        }

        // Reemplazamiento
        var new_min_fitness_idx: usize = 0;
        var old_max_fitness_idx: usize = 0;
        for (new_population, &new_fitnesses, fitnesses, 0..) |new_element, *new_fitness, fitness, i| {
            new_fitness.* = utils.getFitness(new_element, training_set, training_set);
            if (new_fitness.* < new_fitnesses[new_min_fitness_idx])
                new_min_fitness_idx = i;

            if (fitness > fitnesses[old_max_fitness_idx])
                old_max_fitness_idx = i;
        }

        new_population[new_min_fitness_idx] = population[old_max_fitness_idx];
        new_fitnesses[new_min_fitness_idx] = fitnesses[old_max_fitness_idx];
        std.mem.copy([]f64, &population, &new_population);
        std.mem.copy(f64, &fitnesses, &new_fitnesses);
    }

    var max_fitness_idx: usize = 0;
    for (fitnesses, 0..) |fitness, i| {
        if (fitness > fitnesses[max_fitness_idx])
            max_fitness_idx = i;
    }

    return population[max_fitness_idx];
}
