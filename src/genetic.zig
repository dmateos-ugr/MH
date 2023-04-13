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

fn agg_aux(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
    comptime cross_fn: CrossFn,
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

    // Initial population
    for (population) |*element| {
        element.* = try utils.createRandomSolution(n, allocator, rnd);
    }
    defer for (population) |element| {
        allocator.free(element);
    };
    utils.getFitnesses(population, training_set, fitnesses);

    var evaluations: usize = population.len; // we have already done one round of evaluations
    while (evaluations < 15000) : (evaluations += new_population.len) {
        // Seleccion
        for (new_population) |*element| {
            const idx1 = rnd.uintLessThan(usize, population.len);
            const idx2 = rnd.uintLessThan(usize, population.len);
            const idx_win = if (fitnesses[idx1] > fitnesses[idx2]) idx1 else idx2;
            element.* = try allocator.dupe(f64, population[idx_win]);
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
        utils.getFitnesses(new_population, training_set, new_fitnesses);
        var new_min_fitness_idx: usize = 0;
        var old_max_fitness_idx: usize = 0;
        for (new_fitnesses, fitnesses, 0..) |new_fitness, fitness, i| {
            if (new_fitness < new_fitnesses[new_min_fitness_idx])
                new_min_fitness_idx = i;
            if (fitness > fitnesses[old_max_fitness_idx])
                old_max_fitness_idx = i;
        }
        allocator.free(new_population[new_min_fitness_idx]);
        new_population[new_min_fitness_idx] = try allocator.dupe(f64, population[old_max_fitness_idx]);
        new_fitnesses[new_min_fitness_idx] = fitnesses[old_max_fitness_idx];

        // Free elements of previous population
        for (population) |element| {
            allocator.free(element);
        }

        // Make new_population the current population
        std.mem.swap(*[N_POPULATION][]f64, &population, &new_population);
        std.mem.swap(*[N_POPULATION]f64, &fitnesses, &new_fitnesses);
    }

    const max_fitness_idx = std.mem.indexOfMax(f64, fitnesses);
    return allocator.dupe(f64, population[max_fitness_idx]);
}
