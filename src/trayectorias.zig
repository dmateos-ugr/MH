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
    return -MU * fitness / LOG_PHI;
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
    const max_vecinos = 10 * n;
    const max_exitos = @floatToInt(usize, 0.1 * @intToFloat(f64, max_vecinos));

    const w = try utils.createRandomSolution(n, allocator, rnd);
    defer allocator.free(w);
    var fitness_current = utils.getFitness(w, training_set, training_set);

    const w_best = try allocator.dupe(f64, w);
    var fitness_best = fitness_current;

    const w_mut = try allocator.alloc(f64, n);
    defer allocator.free(w_mut);

    const t0 = initialTemperature(fitness_current);
    const M = MAX_EVALUATIONS / max_vecinos;
    const beta = (t0 - FINAL_TEMPERATURE) / (@intToFloat(f64, M) * t0 * FINAL_TEMPERATURE);

    var t = t0;
    var vecinos: usize = 0;
    var exitos: usize = 0;
    var evaluations: usize = 0;
    while (evaluations < MAX_EVALUATIONS) : ({
        evaluations += 1;
        vecinos += 1;
    }) {
        // Mutate w into w_mut
        @memcpy(w_mut, w);
        utils.mov(w_mut, rnd.uintLessThan(usize, n), rnd);

        const fitness_mut = utils.getFitness(w_mut, training_set, training_set);
        const diff_fitness = fitness_mut - fitness_current;
        // if (diff_fitness < 0)
        //     utils.print("{d}\n", .{std.math.exp(diff_fitness / (@intToFloat(f64, k) * t))});
        // Casi todos los exitos ocurren cuando diff_fitness=0. suspicious
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
            // Enfriamieno
            if (exitos == 0)
                break;
            t = nextTemperature(t, beta);
            vecinos = 0;
            exitos = 0;

            // if (t < FINAL_TEMPERATURE)
            //     utils.print("got at {} evaluations\n", .{evaluations});
        }
    }

    // utils.print("{d} {d}\n", .{t, FINAL_TEMPERATURE});

    return w_best;
}

pub fn BMB(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
) ![]const f64 {
    _ = training_set;
    _ = allocator;
    _ = rnd;
}

pub fn ILS(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
) ![]const f64 {
    _ = training_set;
    _ = allocator;
    _ = rnd;
}

pub fn ILS_ES(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
) ![]const f64 {
    _ = training_set;
    _ = allocator;
    _ = rnd;
}

pub fn VNS(
    training_set: []const Example,
    allocator: Allocator,
    rnd: Random,
) ![]const f64 {
    _ = training_set;
    _ = allocator;
    _ = rnd;
}
