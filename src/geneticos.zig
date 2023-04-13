const std = @import("std");
const utils = @import("utils.zig");
const Random = std.rand.Random;
const Allocator = std.mem.Allocator;
const Example = utils.Example;

fn randFloatRange(min: f64, max: f64, rnd: Random) f64 {}

const CrossFn = fn (w1: []const f64, w2: []const f64, rnd: Random) void;

// TODO perf probar bucle con indice
const ALPHA_BLX = 0.3;
fn crossBLX(w1: []const f64, w2: []const f64, rnd: Random) void {
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

fn crossArithmetic(w1: []const f64, w2: []const f64, rnd: Random) void {
    const alpha = rnd.float(f64);
    const one_minus_alpha = 1 - alpha;
    for (w1, w2) |*weight1, *weight2| {
        weight1.* = alpha * weight1.* + one_minus_alpha * weight2.*;
        weight2.* = alpha * weight2.* + one_minus_alpha * weight1.*;
    }
}

const N_POPULATION = 50;

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
	const population = [N_POPULATION][]const f64{};
	const population2 = [N_POPULATION][]const f64{};
	for (population, population2) |w, w2| {

	}

}
