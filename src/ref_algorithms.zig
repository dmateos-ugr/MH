const std = @import("std");
const utils = @import("utils.zig");
const Random = std.rand.Random;
const Allocator = std.mem.Allocator;
const Example = utils.Example;

fn abs(f: f64) f64 {
    return if (f < 0) -f else f;
}

pub fn greedy(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    _ = rnd; // rnd is unused, since greedy is deterministic

    const n = training_set[0].attributes.len;

    // First solution
    const w = try allocator.alloc(f64, n);
    errdefer allocator.free(w);
    @memset(w, 0);

    for (training_set, 0..) |example, i_example| {
        // Get the closest enemy and closest friend to `example`
        var closest_enemy_i: usize = 0;
        var closest_friend_i: usize = 0;
        var closest_enemy_distance = std.math.floatMax(f64);
        var closest_friend_distance = std.math.floatMax(f64);
        for (training_set, 0..) |other, i| {
            if (i == i_example) continue;
            const dist = example.distance(other);
            if (std.mem.eql(u8, example.class, other.class)) {
                if (dist < closest_friend_distance) {
                    closest_friend_distance = dist;
                    closest_friend_i = i;
                }
            } else {
                if (dist < closest_enemy_distance) {
                    closest_enemy_distance = dist;
                    closest_enemy_i = i;
                }
            }
        }

        // Add the difference to the closest enemy and substract the difference
        // to the closest friend
        const enemy_attrs = training_set[closest_enemy_i].attributes;
        const friend_attrs = training_set[closest_friend_i].attributes;
        for (w, example.attributes, enemy_attrs, friend_attrs) |*weight, attr, enemy_attr, friend_attr| {
            weight.* += abs(attr - enemy_attr) - abs(attr - friend_attr);
        }
    }

    // Normalize w
    const w_max = std.mem.max(f64, w);
    for (w) |*weight| {
        weight.* = if (weight.* < 0) 0 else weight.* / w_max;
    }

    return w;
}

pub fn algOriginal1NN(training_set: []const Example, allocator: Allocator, rnd: Random) ![]const f64 {
    _ = rnd;

    // The original 1NN algorithm simply uses a vector of weights set to 1.0
    const n = training_set[0].attributes.len;
    const w = try allocator.alloc(f64, n);
    @memset(w, 1.0);
    return w;
}
