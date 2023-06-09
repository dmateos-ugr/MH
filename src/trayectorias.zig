const std = @import("std");
const utils = @import("utils.zig");
const busquedaLocal = @import("bl.zig").busquedaLocal;
const Random = std.rand.Random;
const Allocator = std.mem.Allocator;
const Example = utils.Example;

const MAX_EVALUATIONS = 15000;
const PHI = 0.2;
const LOG_PHI = std.math.ln(PHI);
const MU = 0.3;
const FINAL_TEMPERATURE = 1.0e-4;

fn nextTemperature(tk: f64, beta: f64) f64 {
    return tk / (1 + beta * tk);
}

fn initialTemperature(fitness: f64) f64 {
    var result = -MU * fitness / LOG_PHI;
    while (result <= FINAL_TEMPERATURE) {
        result *= 10;
    }
    return result;
}

fn shouldAcceptWorseSolution(diff_fitness: f64, t: f64, rnd: Random) bool {
    std.debug.assert(diff_fitness <= 0);
    return rnd.float(f64) <= std.math.exp(diff_fitness / t);
}

// Enfriamiento Simulado
pub fn ES(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
) ![]const f64 {
    const n = training_set[0].attributes.len;
    const w = try utils.createRandomSolution(n, allocator, rnd);
    _ = try enfriamientoSimulado(w, training_set, MAX_EVALUATIONS, allocator, rnd);
    return w;
}

fn enfriamientoSimulado(
    w_best: []f64,
    training_set: []const Example,
    max_evaluations: usize,
    allocator: Allocator,
    rnd: Random,
) !f64 {
    const n = w_best.len;
    const max_vecinos = 10 * n;
    const max_exitos = @floatToInt(usize, 0.1 * @intToFloat(f64, max_vecinos));

    var fitness_best = utils.getFitness(w_best, training_set, training_set);

    const w = try allocator.dupe(f64, w_best);
    defer allocator.free(w);
    var fitness_current = fitness_best;

    const w_mut = try allocator.alloc(f64, n);
    defer allocator.free(w_mut);

    const t0 = initialTemperature(fitness_current);
    const M = max_evaluations / max_vecinos;
    const beta = (t0 - FINAL_TEMPERATURE) / (@intToFloat(f64, M) * t0 * FINAL_TEMPERATURE);

    var t = t0;
    var vecinos: usize = 0;
    var exitos: usize = 0;
    var evaluations: usize = 0;
    while (evaluations < max_evaluations) : ({
        evaluations += 1;
        vecinos += 1;
    }) {
        // Mutate w into w_mut
        @memcpy(w_mut, w);
        utils.mov(w_mut, rnd.uintLessThan(usize, n), rnd);

        const fitness_mut = utils.getFitness(w_mut, training_set, training_set);
        const diff_fitness = fitness_mut - fitness_current;
        if (diff_fitness > 0 or shouldAcceptWorseSolution(diff_fitness, t, rnd)) {
            @memcpy(w, w_mut);
            fitness_current = fitness_mut;
            if (fitness_current > fitness_best) {
                @memcpy(w_best, w);
                fitness_best = fitness_current;
            }
            exitos += 1;
        }

        if (vecinos == max_vecinos or exitos == max_exitos) {
            // Enfriamiento
            if (exitos == 0)
                break;
            t = nextTemperature(t, beta);
            vecinos = 0;
            exitos = 0;
        }
    }

    return fitness_best;
}

pub fn BMB(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
) ![]const f64 {
    const n = training_set[0].attributes.len;
    const w = try allocator.alloc(f64, n);
    defer allocator.free(w);
    const w_best = try allocator.alloc(f64, n);
    var fitness_best: f64 = 0;

    for (0..15) |_| {
        // Create random solution
        for (w) |*weight| {
            weight.* = rnd.float(f64); // range [0, 1)
        }

        // Perform local search
        const fitness = (try busquedaLocal(w, training_set, allocator, rnd, .{
            .max_iters = 1000,
        })).fitness;

        // Update best
        if (fitness > fitness_best) {
            @memcpy(w_best, w);
            fitness_best = fitness;
        }
    }

    return w_best;
}

fn ilsMut(w: []f64, rnd: Random) void {
    const n_mut = std.math.max(@floatToInt(usize, @round(@intToFloat(f64, w.len)) * 0.1), 2);
    for (0..n_mut) |_| {
        const i_weight = rnd.uintLessThan(usize, w.len);
        w[i_weight] = rnd.float(f64);
    }
}

fn ilsBl(
    w: []f64,
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
) !f64 {
    return (try busquedaLocal(w, training_set, allocator, rnd, .{
        .max_iters = 1000,
    })).fitness;
}

fn ilsEs(
    w: []f64,
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
) !f64 {
    return enfriamientoSimulado(w, training_set, 1000, allocator, rnd);
}

pub fn ILS(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
) ![]const f64 {
    return ils(training_set, ilsBl, allocator, rnd);
}

pub fn ILS_ES(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
) ![]const f64 {
    return ils(training_set, ilsEs, allocator, rnd);
}

fn ils(
    training_set: []const Example,
    comptime bl_fn: fn ([]f64, []const Example, Allocator, Random) Allocator.Error!f64,
    allocator: Allocator,
    rnd: Random,
) ![]const f64 {
    const n = training_set[0].attributes.len;
    const w = try utils.createRandomSolution(n, allocator, rnd);
    defer allocator.free(w);
    var fitness_best = try bl_fn(w, training_set, allocator, rnd);
    const w_best = try allocator.dupe(f64, w);

    for (0..14) |_| {
        @memcpy(w, w_best);
        ilsMut(w, rnd);

        const fitness = try bl_fn(w, training_set, allocator, rnd);
        if (fitness > fitness_best) {
            @memcpy(w_best, w);
            fitness_best = fitness;
        }
    }

    return w_best;
}

const KMAX = 3;

pub fn VNS(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
) ![]const f64 {
    const n = training_set[0].attributes.len;
    const w = try utils.createRandomSolution(n, allocator, rnd);
    defer allocator.free(w);
    var k: usize = 1;
    var fitness_best = (try busquedaLocal(w, training_set, allocator, rnd, .{
        .max_iters = 1000,
        .num_attributes_mutated = k,
    })).fitness;
    const w_best = try allocator.dupe(f64, w);

    for (0..14) |_| {
        @memcpy(w, w_best);
        ilsMut(w, rnd);

        const fitness = (try busquedaLocal(w, training_set, allocator, rnd, .{
            .max_iters = 1000,
            .num_attributes_mutated = k,
        })).fitness;

        if (fitness > fitness_best) {
            @memcpy(w_best, w);
            fitness_best = fitness;
            k = 1;
        } else {
            k = (k % KMAX) + 1 ;
        }
    }

    return w_best;
}
