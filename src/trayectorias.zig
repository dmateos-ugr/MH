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

fn nextTemperature(tk: f64, t0: f64, M: usize) f64 {
    const beta = (t0 - FINAL_TEMPERATURE) / (@intToFloat(f64, M) * t0 * FINAL_TEMPERATURE);
    return tk / (1 + beta * tk);
}

fn coste(w: []const f64) f64 {
    _ = w;
    // ???
    return 0;
}

fn initialTemperature(fitness: f64) f64 {
    return -MU * fitness / LOG_PHI;
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
    const M = MAX_EVALUATIONS / max_vecinos;

    const w = try utils.createRandomSolution(n, allocator, rnd);
    defer allocator.free(w);
    var fitness_current = utils.getFitness(w, training_set, training_set);

    const w_best = try allocator.dupe(f64, w);
    var fitness_best = fitness_current;

    const w_mut = try allocator.alloc(f64, n);
    defer allocator.free(w_mut);

    const t0 = initialTemperature(fitness_current);
    var k: usize = 0;

    var evaluations: usize = 0;

    var t = t0;
    while (t >= FINAL_TEMPERATURE) : ({
        t = nextTemperature(t, t0, M);
        k += 1;
    }) {
        var vecinos: usize = 0;
        var exitos: usize = 0;
        while (vecinos < max_vecinos and exitos < max_exitos) : (vecinos += 1) {
            // Mutate w into w_mut
            @memcpy(w_mut, w);
            utils.mov(w_mut, rnd.uintLessThan(usize, n), rnd);

            const fitness_mut = utils.getFitness(w_mut, training_set, training_set);
            const diff_fitness = fitness_mut - fitness_current;
            // if (diff_fitness < 0)
            //     utils.print("{d}\n", .{std.math.exp(diff_fitness / (@intToFloat(f64, k) * t))});
            // if (diff_fitness == 0)
            //     continue;
            // Casi todos los exitos ocurren cuando diff_fitness=0. suspicious
            if (diff_fitness > 0 or rnd.float(f64) <= std.math.exp(diff_fitness / (@intToFloat(f64, k) * t))) { // suspicious
                @memcpy(w, w_mut);
                fitness_current = fitness_mut;
                if (fitness_current > fitness_best) {
                    @memcpy(w_best, w);
                    fitness_best = fitness_current;
                }
                exitos += 1;
            }

        }
        // en la primera iteracion termina por vecinos, y despues siempre por exitos.
        // supongo que se mueve a un maximo local donde diff_fitness = 0 siempre.
        utils.print("{}/{} {}/{}\n", .{vecinos, max_vecinos, exitos, max_exitos});

        evaluations += vecinos;

        if (exitos == 0)
            break;
    }

    utils.print("{}\n", .{evaluations});

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
