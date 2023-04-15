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

const PROB_CROSS = 0.7;
const PROB_MUTATION = 0.1;
const N_POPULATION = 50;
const N_CROSS_PAIRS = @floatToInt(usize, PROB_CROSS * N_POPULATION / 2.0);
const N_MUTATIONS = @floatToInt(usize, PROB_MUTATION * N_POPULATION);
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
    while (evaluations < 15000) : (evaluations += new_population.len) {
        // Seleccion
        for (new_population) |element| {
            const idx1 = rnd.uintLessThan(usize, population.len);
            const idx2 = rnd.uintLessThan(usize, population.len);
            const idx_win = if (fitnesses[idx1] > fitnesses[idx2]) idx1 else idx2;
            std.mem.copy(f64, element, population[idx_win]);
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
        std.mem.copy(f64, new_population[new_min_fitness_idx], population[old_max_fitness_idx]);
        new_fitnesses[new_min_fitness_idx] = fitnesses[old_max_fitness_idx];

        // Make new_population the current population
        std.mem.swap(*[N_POPULATION][]f64, &population, &new_population);
        std.mem.swap(*[N_POPULATION]f64, &fitnesses, &new_fitnesses);
    }

    const max_fitness_idx = std.mem.indexOfMax(f64, fitnesses);
    return allocator.dupe(f64, population[max_fitness_idx]);
}

pub fn AGE_BLX(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    return age_aux(training_set, allocator, rnd, crossBLX);
}

pub fn AGE_Arit(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    return age_aux(training_set, allocator, rnd, crossArithmetic);
}

const WeightFitness = struct {
    w: []f64,
    fitness: f64,

    fn createRandom(training_set: []const Example, allocator: Allocator, rnd: Random) !WeightFitness {
        const n = training_set[0].attributes.len;
        const w = try utils.createRandomSolution(n, allocator, rnd);
        return init(w, training_set);
    }

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

fn cmpByFitness(_: void, element1: WeightFitness, element2: WeightFitness) bool {
    return element1.fitness < element2.fitness;
}

fn sortPopulation2(population: []WeightFitness) void {
    std.sort.sort(WeightFitness, population, {}, cmpByFitness);
}

fn age_aux2(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
    comptime cross_fn: CrossFn,
) ![]const f64 {
    const n = training_set[0].attributes.len;

    var population: [N_POPULATION]WeightFitness = undefined;
    var two_elements: [2]WeightFitness = undefined;

    // Allocate memory for the population, initializing it randomly
    for (&population) |*element| {
        element.* = try WeightFitness.createRandom(training_set, allocator, rnd);
    }
    defer for (population) |element| {
        element.deinit(allocator);
    };
    for (&two_elements) |*element| {
        element.w = try allocator.alloc(f64, n);
    }
    defer for (two_elements) |element| {
        element.deinit(allocator);
    };

    sortPopulation2(&population);

    var evaluations: usize = population.len; // we have already done one round of evaluations
    while (evaluations < 15000) : (evaluations += 2) {
        // Seleccion
        for (0..2) |i| {
            const idx1 = rnd.uintLessThan(usize, population.len);
            const idx2 = rnd.uintLessThan(usize, population.len);
            const idx_win = if (population[idx1].fitness > population[idx2].fitness) idx1 else idx2;
            std.mem.copy(f64, two_elements[i].w, population[idx_win].w);
            // no need to copy fitness since we are mutating those two now
            // two_elements[i].fitness = population[idx_win].fitness;
        }

        // Cruce
        cross_fn(two_elements[0].w, two_elements[1].w, rnd);

        // Mutacion
        for (two_elements) |element| {
            if (rnd.float(f64) < PROB_MUTATION) {
                const i_weight = rnd.uintLessThan(usize, n);
                utils.mov(element.w, i_weight, rnd);
            }
        }

        // Reemplazamiento. No need to copy contents here
        for (&two_elements) |*element| {
            element.updateFitness(training_set);
        }
        var four_elements = [4]WeightFitness{ population[0], population[1], two_elements[0], two_elements[1] };
        sortPopulation2(&four_elements);
        std.mem.copy(f64, population[0].w, four_elements[2].w); // best two
        std.mem.copy(f64, population[1].w, four_elements[3].w);
        population[0].fitness = four_elements[2].fitness;
        population[1].fitness = four_elements[3].fitness;
        sortPopulation2(&population);

    }

    // for (population) |element| {
    //     utils.print("{d} {d}\n", .{element.w, element.fitness});
    // }
    // utils.print("train fitness: {d}\n", .{population[population.len - 1].fitness});
    return allocator.dupe(f64, population[population.len - 1].w);
}









const SortContext = struct {
    population: [][]const f64,
    fitnesses: []f64,

    pub fn lessThan(self: SortContext, idx1: usize, idx2: usize) bool {
        return self.fitnesses[idx1] < self.fitnesses[idx2];
    }

    pub fn swap(self: *SortContext, idx1: usize, idx2: usize) void {
        std.mem.swap([]const f64, &self.population[idx1], &self.population[idx2]);
        std.mem.swap(f64, &self.fitnesses[idx1], &self.fitnesses[idx2]);
    }
};

fn sortPopulation(population: [][]const f64, fitnesses: []f64) void {
    var sort_context = SortContext{ .population = population, .fitnesses = fitnesses };
    std.sort.insertionSortContext(population.len, &sort_context);
}


fn age_aux(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
    comptime cross_fn: CrossFn,
) ![]const f64 {
    const n = training_set[0].attributes.len;

    var population: [N_POPULATION][]f64 = undefined;
    var fitnesses: [N_POPULATION]f64 = undefined;
    var two_elements: [2][]f64 = undefined;
    var two_fitnesses: [2]f64 = undefined;

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

    sortPopulation(&population, &fitnesses);

    var evaluations: usize = population.len; // we have already done one round of evaluations
    while (evaluations < 15000) : (evaluations += 2) {
        // Seleccion
        for (0..2) |i| {
            const idx1 = rnd.uintLessThan(usize, population.len);
            const idx2 = rnd.uintLessThan(usize, population.len);
            const idx_win = if (fitnesses[idx1] > fitnesses[idx2]) idx1 else idx2;
            std.mem.copy(f64, two_elements[i], population[idx_win]);
        }

        // Cruce
        cross_fn(two_elements[0], two_elements[1], rnd);

        // Mutacion
        for (two_elements) |element| {
            if (rnd.float(f64) < PROB_MUTATION) {
                const i_weight = rnd.uintLessThan(usize, n);
                utils.mov(element, i_weight, rnd);
            }
        }

        // Reemplazamiento. No need to copy contents here
        utils.getFitnesses(&two_elements, training_set, &two_fitnesses);
        var four_elements = [4][]f64{ population[0], population[1], two_elements[0], two_elements[1] };
        var four_fitnesses = [4]f64{ fitnesses[0], fitnesses[1], two_fitnesses[0], two_fitnesses[1] };
        sortPopulation(&four_elements, &four_fitnesses);

        // const fitness_greatest = utils.getFitness(four_elements[3], training_set, training_set);
        // // utils.print("greatest: {d}\n", .{fitness_greatest});
        // for (four_elements, four_fitnesses) |element, fitness| {
        //     // utils.print("{d} {d}\n", .{fitness, utils.getFitness(element, training_set, training_set)});
        //     std.debug.assert(fitness == utils.getFitness(element, training_set, training_set));
        //     std.debug.assert(fitness_greatest >= fitness);
        // }

        std.mem.copy(f64, population[0], four_elements[2]); // best two
        std.mem.copy(f64, population[1], four_elements[3]);
        fitnesses[0] = four_fitnesses[2];
        fitnesses[1] = four_fitnesses[3];
        sortPopulation(&population, &fitnesses);
    }

    // Return the element with highest fitness
    // utils.print("train fitness: {d}\n", .{fitnesses[population.len - 1]});
    return allocator.dupe(f64, population[population.len - 1]);
}
