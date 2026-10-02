entry matmul [m][n][k] (a: [m][k]f32) (b: [k][n]f32): *[m][n]f32 =
  let bt = transpose b
  in map (\a_row -> map (\b_col -> f32.sum (map2 (*) a_row b_col)) bt) a

let oftb_scale_f32 : f32 = 0.7071067811865476

let sanitize_f32 (v: f32) : f32 =
  if f32.isnan v || f32.isinf v then 0f32 else v

let clamp_f16_value (v: f32) : f32 =
  let safe = sanitize_f32 v
  in f32.max (-60000f32) (f32.min 60000f32 safe)

let clamp_f16_weight (v: f32) : f32 =
  let safe = sanitize_f32 v
  in f32.max (-65504f32) (f32.min 65504f32 safe)

let splitmix64 (value: u64) : u64 =
  let z0 = value + 0x9E3779B97F4A7C15u64
  let z1 = (z0 ^ (z0 >> 30u64)) * 0xBF58476D1CE4E5B9u64
  let z2 = (z1 ^ (z1 >> 27u64)) * 0x94D049BB133111EBu64
  in z2 ^ (z2 >> 31u64)

let fwht_stages [n] (x: [n]f32) (block: i64) (stages: i64) : [n]f32 =
  if block <= 0 || stages <= 0 then x
  else
    let (final, _) =
      loop (cur, h) = (x, 1i64) for _s < stages do
        let next =
          tabulate n (\i ->
            let o = i % block
            in if (o / h) % 2i64 == 0i64
               then (cur[i] + cur[i + h]) * oftb_scale_f32
               else (cur[i - h] - cur[i]) * oftb_scale_f32)
        in (next, h * 2i64)
    in final

let mix_radix_blocks [n] (x: [n]f32) (block: i64) (radix: i64) : [n]f32 =
  if block <= 0 || radix <= 0 then x
  else
    let coeff = 2f32 / f32.i64 radix
    in tabulate n (\i ->
      let o = i % block
      let s = loop acc = 0f32 for b < radix do acc + x[b * block + o]
      in x[i] - coeff * s)

let diffuse_row [n] (x: [n]f32) (radix: i64) (block: i64) (stages: i64) : [n]f32 =
  mix_radix_blocks (fwht_stages x block stages) block radix

let apply_diffuse [n] (x: [n]f32) (diffusion: bool) (radix: i64) (block: i64) (stages: i64) : [n]f32 =
  if diffusion then diffuse_row x radix block stages else x

let gram_sigma_max_2col [m] (w: [m][2]f32) : f32 =
  let a = reduce (+) 0f64 (map (\i ->
    let x = f64.f32 (sanitize_f32 w[i][0])
    in x * x) (iota m))
  let b = reduce (+) 0f64 (map (\i ->
    f64.f32 (sanitize_f32 w[i][0]) * f64.f32 (sanitize_f32 w[i][1])) (iota m))
  let c = reduce (+) 0f64 (map (\i ->
    let y = f64.f32 (sanitize_f32 w[i][1])
    in y * y) (iota m))
  let tr = a + c
  let diff = a - c
  let delta = diff * diff + 4f64 * b * b
  let lam = (tr + f64.sqrt delta) / 2f64
  let sig = f64.sqrt (f64.max 0f64 lam)
  in f32.max 0f32 (f32.f64 sig)

let spectral_normalize_matrix_exact [m] (w: [m][2]f32) (target: f32) : ([m][2]f32, f32, f32) =
  let safe_w = map (map sanitize_f32) w
  let sigma = gram_sigma_max_2col safe_w
  let safe_target = f32.max target 1e-6f32
  let scale = if sigma > safe_target then safe_target / sigma else 1f32
  let normalized = map (map (* scale)) safe_w
  in (normalized, sigma, sigma * scale)

entry stack_spectral_normalize_exact [layers][rows]
  (weights: *[layers][rows][2]f32)
  (target: f32)
  : (*[layers][rows][2]f32, f32, f32) =
  let results = map (\weight -> spectral_normalize_matrix_exact weight target) weights
  let normalized = map (\(weight, _, _) -> weight) results
  let before = reduce f32.max 0f32 (map (\(_, sigma, _) -> sigma) results)
  let after = reduce f32.max 0f32 (map (\(_, _, sigma) -> sigma) results)
  in (normalized, before, after)

let rsf_stack_coupling_row_ld [half]
  (row: [half*2]f32)
  (weights_s: [half][2]f16) (weights_t: [half][2]f16)
  (clip_min_f32: f32) (clip_max_f32: f32)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : ([half*2]f32, f32) =
  let x1 = row[0:half] :> [half]f32
  let x2 = row[half:half*2] :> [half]f32
  let pre = map (\d -> f32.f16 weights_s[d][0] * x2[d] + f32.f16 weights_s[d][1]) (iota half)
  let clipped = map (\sum -> f32.max clip_min_f32 (f32.min clip_max_f32 sum)) pre
  let scale = map f32.exp clipped
  let y1 = map2 (*) x1 scale
  let y2 = map2 (\x2_j j ->
    let trans = f32.f16 weights_t[j][0] * y1[j] + f32.f16 weights_t[j][1]
    in x2_j + sanitize_f32 trans) x2 (iota half)
  let o1 = map2 (\a b -> (a - b) * oftb_scale_f32) y1 y2
  let o2 = map2 (\a b -> (a + b) * oftb_scale_f32) y1 y2
  let joined = (o1 ++ o2) :> [half*2]f32
  let diffused = apply_diffuse joined diffusion radix block stages
  let out = map clamp_f16_value diffused
  let ld = f32.sum clipped
  in (out, ld)

let rsf_stack_coupling_row [half]
  (row: [half*2]f32)
  (weights_s: [half][2]f16) (weights_t: [half][2]f16)
  (clip_min_f32: f32) (clip_max_f32: f32)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : [half*2]f32 =
  let (out, _) = rsf_stack_coupling_row_ld row weights_s weights_t clip_min_f32 clip_max_f32 diffusion radix block stages
  in out

let rsf_stack_invert_row_ld [half]
  (row: [half*2]f32)
  (weights_s: [half][2]f16) (weights_t: [half][2]f16)
  (clip_min_f32: f32) (clip_max_f32: f32)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : ([half*2]f32, f32) =
  let y_d = apply_diffuse row diffusion radix block stages
  let y1p = y_d[0:half] :> [half]f32
  let y2p = y_d[half:half*2] :> [half]f32
  let u1 = map2 (\a b -> (a + b) * oftb_scale_f32) y1p y2p
  let u2 = map2 (\a b -> (b - a) * oftb_scale_f32) y1p y2p
  let x2 = map (\d ->
    let trans = f32.f16 weights_t[d][0] * u1[d] + f32.f16 weights_t[d][1]
    let safe_trans = sanitize_f32 trans
    in u2[d] - safe_trans) (iota half)
  let pre = map (\d -> f32.f16 weights_s[d][0] * x2[d] + f32.f16 weights_s[d][1]) (iota half)
  let clipped = map (\p -> f32.max clip_min_f32 (f32.min clip_max_f32 p)) pre
  let x1 = map2 (\u_j s_j -> sanitize_f32 (u_j / f32.exp s_j)) u1 clipped
  let out = map clamp_f16_value ((x1 ++ x2) :> [half*2]f32)
  let ld = f32.sum clipped
  in (out, ld)

let rsf_stack_invert_row [half]
  (row: [half*2]f32)
  (weights_s: [half][2]f16) (weights_t: [half][2]f16)
  (clip_min_f32: f32) (clip_max_f32: f32)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : [half*2]f32 =
  let (out, _) = rsf_stack_invert_row_ld row weights_s weights_t clip_min_f32 clip_max_f32 diffusion radix block stages
  in out

entry rsf_forward [n][half] (input: [n][half*2]f16)
  (weights_s: [half][2]f16) (weights_t: [half][2]f16)
  (clip_min: f16) (clip_max: f16)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : *[n][half*2]f16 =
  let raw_min = f32.f16 clip_min
  let raw_max = f32.f16 clip_max
  let clip_min_f32 = f32.min raw_min raw_max
  let clip_max_f32 = f32.max raw_min raw_max
  in map (\row ->
    let row_f32 = map f32.f16 row
    let result = rsf_stack_coupling_row row_f32 weights_s weights_t clip_min_f32 clip_max_f32 diffusion radix block stages
    in map (\v -> f16.f32 (clamp_f16_value v)) result) input

let sfd_fisher_update_core [d][e]
  (weights: [d][e]f32) (gradients: [d][e]f32)
  (momentum_state: [d][e]f32) (fisher_state: [d][e]f32)
  (learning_rate: f32) (momentum_beta: f32) (fisher_gamma: f32)
  (optimizer_step: i64) (epsilon: f32) (trust_ratio: f32) (weight_floor: f32)
  : ([d][e]f32, [d][e]f32, [d][e]f32) =
  let safe_beta = f32.max 0f32 (f32.min 0.99999994f32 (sanitize_f32 momentum_beta))
  let safe_gamma = f32.max 0f32 (f32.min 0.99999994f32 (sanitize_f32 fisher_gamma))
  let safe_eps = f32.max (sanitize_f32 epsilon) 1e-12f32
  let safe_lr = sanitize_f32 learning_rate
  let safe_trust_ratio = f32.max 0f32 (f32.min 1f32 (sanitize_f32 trust_ratio))
  let safe_floor = f32.max 0f32 (sanitize_f32 weight_floor)
  let step_f = f32.i64 (i64.max 1 optimizer_step)
  let momentum_correction = f32.max safe_eps (1f32 - safe_beta f32.** step_f)
  let fisher_correction = f32.max safe_eps (1f32 - safe_gamma f32.** step_f)
  let momentum_next = map2 (map2 (\m g ->
    let safe_m = sanitize_f32 m
    let safe_g = sanitize_f32 g
    let candidate = safe_beta * safe_m + (1f32 - safe_beta) * safe_g
    in sanitize_f32 candidate)) momentum_state gradients
  let fisher_next = map2 (map2 (\f g ->
    let safe_f = f32.max 0f32 (f32.min 1e6f32 (sanitize_f32 f))
    let safe_g = sanitize_f32 g
    let candidate = safe_gamma * safe_f + (1f32 - safe_gamma) * safe_g * safe_g
    in f32.min 1e6f32 (f32.max 0f32 (sanitize_f32 candidate)))) fisher_state gradients
  let weights_next = map3 (map3 (\w m f ->
    let safe_w = sanitize_f32 w
    let m_hat = m / momentum_correction
    let f_hat = f / fisher_correction
    let raw_step = safe_lr * m_hat / (f32.sqrt (f32.max f_hat 0f32) + safe_eps)
    let max_step = safe_trust_ratio * f32.max safe_floor (f32.abs safe_w)
    let clipped_step = f32.max (-max_step) (f32.min max_step (sanitize_f32 raw_step))
    let updated = safe_w - clipped_step
    in sanitize_f32 updated)) weights momentum_next fisher_next
  in (weights_next, momentum_next, fisher_next)

entry master_weights_to_f16_3d [layers][rows][columns] (weights: [layers][rows][columns]f32): *[layers][rows][columns]f16 =
  map (map (map (\value -> f16.f32 (clamp_f16_weight value)))) weights

let cap_fisher_diag (v: f32) : f32 =
  f32.min 1e6f32 (f32.max 0f32 (sanitize_f32 v))

let cap_fisher_off (v: f32) : f32 =
  f32.min 1e6f32 (f32.max (-1e6f32) (sanitize_f32 v))

let sfd_block_2x2_update (w: [2]f32) (g: [2]f32) (m: [2]f32) (f: [3]f32)
                         (learning_rate: f32) (momentum_beta: f32) (fisher_gamma: f32)
                         (optimizer_step: i64) (epsilon: f32) (trust_ratio: f32) (weight_floor: f32)
                         : ([2]f32, [2]f32, [3]f32) =
  let safe_beta = f32.max 0f32 (f32.min 0.99999994f32 (sanitize_f32 momentum_beta))
  let safe_gamma = f32.max 0f32 (f32.min 0.99999994f32 (sanitize_f32 fisher_gamma))
  let safe_eps = f32.max (sanitize_f32 epsilon) 1e-12f32
  let safe_lr = sanitize_f32 learning_rate
  let safe_trust = f32.max 0f32 (f32.min 1f32 (sanitize_f32 trust_ratio))
  let safe_floor = f32.max 0f32 (sanitize_f32 weight_floor)
  let gw = sanitize_f32 g[0]
  let gb = sanitize_f32 g[1]
  let mw = sanitize_f32 m[0]
  let mb = sanitize_f32 m[1]
  let fww = cap_fisher_diag f[0]
  let fwb = cap_fisher_off f[1]
  let fbb = cap_fisher_diag f[2]
  let mw_n = sanitize_f32 (safe_beta * mw + (1f32 - safe_beta) * gw)
  let mb_n = sanitize_f32 (safe_beta * mb + (1f32 - safe_beta) * gb)
  let fww_n = cap_fisher_diag (safe_gamma * fww + (1f32 - safe_gamma) * gw * gw)
  let fwb_n = cap_fisher_off (safe_gamma * fwb + (1f32 - safe_gamma) * gw * gb)
  let fbb_n = cap_fisher_diag (safe_gamma * fbb + (1f32 - safe_gamma) * gb * gb)
  let step_f = f32.i64 (i64.max 1 optimizer_step)
  let mom_corr = f32.max safe_eps (1f32 - safe_beta f32.** step_f)
  let fish_corr = f32.max safe_eps (1f32 - safe_gamma f32.** step_f)
  let mw_hat = mw_n / mom_corr
  let mb_hat = mb_n / mom_corr
  let fww_hat = fww_n / fish_corr
  let fwb_hat = fwb_n / fish_corr
  let fbb_hat = fbb_n / fish_corr
  let lam = safe_eps
  let a = fww_hat + lam
  let b = fwb_hat
  let c = fbb_hat + lam
  let det = f32.max 1e-12f32 (a * c - b * b)
  let sd = f32.sqrt det
  let s = f32.sqrt (f32.max 0f32 (a + c + 2f32 * sd))
  let alpha = 1f32 / (sd * f32.max s 1e-12f32)
  let raw_w = alpha * ((c + sd) * mw_hat - b * mb_hat) * safe_lr
  let raw_b = alpha * ((-b) * mw_hat + (a + sd) * mb_hat) * safe_lr
  let ww = sanitize_f32 w[0]
  let wb = sanitize_f32 w[1]
  let max_w = safe_trust * f32.max safe_floor (f32.abs ww)
  let max_b = safe_trust * f32.max safe_floor (f32.abs wb)
  let dw = f32.max (-max_w) (f32.min max_w (sanitize_f32 raw_w))
  let db = f32.max (-max_b) (f32.min max_b (sanitize_f32 raw_b))
  let cand_w = ww - dw
  let cand_b = wb - db
  let ww_n = if f32.isnan cand_w || f32.isinf cand_w then ww else cand_w
  let wb_n = if f32.isnan cand_b || f32.isinf cand_b then wb else cand_b
  in ([ww_n, wb_n] :> [2]f32, [mw_n, mb_n] :> [2]f32, [fww_n, fwb_n, fbb_n] :> [3]f32)

entry stack_update_sfd_block2x2_master [layers][rows]
  (master_weights: *[layers][rows][2]f32)
  (gradients: [layers][rows][2]f32)
  (momentum: *[layers][rows][2]f32)
  (fisher_blocks: *[layers][rows][3]f32)
  (lr: f32) (beta1: f32) (beta2: f32) (step: i64) (eps: f32) (trust_ratio: f32) (weight_floor: f32)
  : (*[layers][rows][2]f32, *[layers][rows][2]f32, *[layers][rows][3]f32) =
  let updates = map4 (\w g m f ->
    let trips = map4 (\wr gr mr fr ->
      sfd_block_2x2_update wr gr mr fr lr beta1 beta2 step eps trust_ratio weight_floor
    ) w g m f
    in (map (\(wr, _, _) -> wr) trips,
        map (\(_, mr, _) -> mr) trips,
        map (\(_, _, fr) -> fr) trips)
  ) master_weights gradients momentum fisher_blocks
  in (map (\(w, _, _) -> w) updates,
      map (\(_, m, _) -> m) updates,
      map (\(_, _, f) -> f) updates)

entry master_weights_to_f16_2d [rows][columns] (weights: [rows][columns]f32): *[rows][columns]f16 =
  map (map (\value -> f16.f32 (clamp_f16_weight value))) weights

entry embedding_update_sfd_master [vocab_size][dim]
  (master_weight: *[vocab_size][dim]f32) (grad_weight: [vocab_size][dim]f32)
  (momentum_state: *[vocab_size][dim]f32) (fisher_state: *[vocab_size][dim]f32)
  (learning_rate: f32) (momentum_beta: f32) (fisher_gamma: f32) (optimizer_step: i64) (epsilon: f32)
  (trust_ratio: f32) (weight_floor: f32)
  : ([vocab_size][dim]f32, [vocab_size][dim]f32, [vocab_size][dim]f32) =
  sfd_fisher_update_core master_weight grad_weight momentum_state fisher_state learning_rate momentum_beta fisher_gamma optimizer_step epsilon trust_ratio weight_floor

entry scale_matrix_f32 [rows][columns] (values: *[rows][columns]f32) (scale_factor: f32) : *[rows][columns]f32 =
  map (map (\value -> value * scale_factor)) values

entry clip_matrix_global_norm_f32 [rows][columns]
  (values: *[rows][columns]f32) (clip_norm: f32) : *[rows][columns]f32 =
  let flat_values = flatten values
  let maximum_absolute_value = reduce f32.max 0f32 (map f32.abs flat_values)
  let scaled_norm_squared =
    if maximum_absolute_value > 0f32
    then f32.sum (map (\value ->
      let scaled = value / maximum_absolute_value
      in scaled * scaled) flat_values)
    else 0f32
  let norm = maximum_absolute_value * f32.sqrt scaled_norm_squared
  let scale =
    if clip_norm > 0f32 && norm > clip_norm && norm > 1e-12f32
    then clip_norm / norm
    else 1f32
  in map (map (* scale)) values

entry embedding_forward_padded [n][batch_size][seq_len][vocab_size][dim]
  (tokens: [n]i64)
  (lengths: [batch_size]i64)
  (positions: [seq_len]i64)
  (weight: [vocab_size][dim]f16) : *[batch_size][seq_len][dim]f16 =
  let slot_indices = map2 (\position slot_index ->
    if position >= 0 && position < seq_len then position else slot_index)
    positions (iota seq_len)
  in map2 (\batch_index length ->
    map (\sequence_index ->
      let flat_index = batch_index * seq_len + sequence_index
      in if sequence_index >= 0 &&
            sequence_index < i64.max 0 (i64.min seq_len length) &&
            flat_index >= 0 &&
            flat_index < n
         then let token = tokens[flat_index]
              in if token >= 0 && token < vocab_size
                 then weight[token]
                 else replicate dim (f16.i32 0)
         else replicate dim (f16.i32 0)) slot_indices) (iota batch_size) lengths

entry embedding_backward_padded [n][batch_size][seq_len][dim][vocab_size]
  (tokens: [n]i64)
  (lengths: [batch_size]i64)
  (grad_output: [batch_size][seq_len][dim]f16)
  (grad_weight: [vocab_size][dim]f32) : *[vocab_size][dim]f32 =
  let total = batch_size * seq_len
  let limits = map (\length -> i64.max 0 (i64.min seq_len length)) lengths
  let validity = tabulate total (\flat_index ->
    let batch_index = flat_index / seq_len
    let sequence_index = flat_index % seq_len
    in if sequence_index < limits[batch_index] && flat_index < n
       then let token = tokens[flat_index]
            in token >= 0 && token < vocab_size
       else false)
  let safe_tokens = tabulate total (\flat_index ->
    if validity[flat_index] then tokens[flat_index] else -1i64)
  let masked_grads = tabulate total (\flat_index ->
    if validity[flat_index]
    then let batch_index = flat_index / seq_len
         let sequence_index = flat_index % seq_len
         in map (\v -> sanitize_f32 (f32.f16 v)) grad_output[batch_index][sequence_index]
    else replicate dim 0f32)
  let updates = hist (map2 (+)) (replicate dim 0f32) vocab_size safe_tokens masked_grads
  in map2 (map2 (+)) grad_weight updates

entry embedding_spectral_normalize [vocab_size][dim]
  (weight: *[vocab_size][dim]f32)
  (u: *[vocab_size]f32)
  (v: *[dim]f32)
  (power_iters: i64)
  (target: f32) : (*[vocab_size][dim]f32, *[vocab_size]f32, *[dim]f32, f32, f32) =
  let safe_weight = map (map sanitize_f32) weight
  let weight_t = transpose safe_weight
  let sanitized_u = map sanitize_f32 u
  let sanitized_v = map sanitize_f32 v
  let u_norm0 = f32.sqrt (f32.sum (map (\value -> value * value) sanitized_u))
  let v_norm0 = f32.sqrt (f32.sum (map (\value -> value * value) sanitized_v))
  let init_u_value = if vocab_size > 0 then 1f32 / f32.sqrt (f32.i64 vocab_size) else 0f32
  let init_v_value = if dim > 0 then 1f32 / f32.sqrt (f32.i64 dim) else 0f32
  let u_start = (if u_norm0 > 1e-12f32 then sanitized_u else replicate vocab_size init_u_value) :> [vocab_size]f32
  let v_start = (if v_norm0 > 1e-12f32 then sanitized_v else replicate dim init_v_value) :> [dim]f32
  let (final_u, final_v) =
    loop (ua, va) = (u_start, v_start) for loop_k < i64.max 1 power_iters do
      let _ = loop_k
      let raw_v = map (\column -> f32.sum (map2 (*) column ua)) weight_t
      let raw_v_norm = f32.sqrt (f32.sum (map (\value -> value * value) raw_v))
      let seeded_v = (if raw_v_norm > 1e-12f32 then map (/ raw_v_norm) raw_v else va) :> [dim]f32
      let raw_u = map (\row -> f32.sum (map2 (*) row seeded_v)) safe_weight
      let u_norm = f32.sqrt (f32.sum (map (\value -> value * value) raw_u))
      let next_u = (map (/ f32.max u_norm 1e-12f32) raw_u) :> [vocab_size]f32
      let refined_v = map (\column -> f32.sum (map2 (*) column next_u)) weight_t
      let refined_norm = f32.sqrt (f32.sum (map (\value -> value * value) refined_v))
      let next_v = (map (/ f32.max refined_norm 1e-12f32) refined_v) :> [dim]f32
      in (next_u, next_v)
  let final_u_sized = final_u :> [vocab_size]f32
  let final_v_sized = final_v :> [dim]f32
  let projected = map (\row -> f32.sum (map2 (*) row final_v_sized)) safe_weight
  let sigma = sanitize_f32 (f32.abs (f32.sum (map2 (*) final_u_sized projected)))
  let safe_target = f32.max target 1e-6f32
  let scale = if sigma > safe_target then safe_target / sigma else 1f32
  let normalized = map (map (* scale)) safe_weight
  in (normalized, copy final_u_sized, copy final_v_sized, sigma, sigma * scale)

let graph_derive_qubit_states [n] (hashes: [n]u64) : ([n]f32, [n]f32, [n]f32, [n]f32) =
  let pi = 3.14159265358979323846f32
  let two_pi = 2f32 * pi
  let inv_m = 1f32 / 1000000f32
  let secondary = map splitmix64 hashes
  let raw_re_a = map (\h -> f32.cos (two_pi * f32.u64 (h % 1000000u64) * inv_m)) hashes
  let raw_im_a = map (\h -> f32.sin (two_pi * f32.u64 ((h >> 20u64) % 1000000u64) * inv_m)) hashes
  let raw_re_b = map (\h -> f32.cos (two_pi * f32.u64 ((h >> 40u64) % 1000000u64) * inv_m)) hashes
  let raw_im_b = map (\s -> f32.sin (two_pi * f32.u64 (s % 1000000u64) * inv_m)) secondary
  let norms = map4 (\ra ia rb ib ->
    let s = ra * ra + ia * ia + rb * rb + ib * ib
    in if s > 1e-30f32 then f32.sqrt s else 1f32) raw_re_a raw_im_a raw_re_b raw_im_b
  in (map2 (/) raw_re_a norms, map2 (/) raw_im_a norms, map2 (/) raw_re_b norms, map2 (/) raw_im_b norms)

entry graph_batch_encode [n] (data_hashes: [n]u64) (seed: u64) : ([]u64, []f32, []f32, []f32, []f32, []i64, []i64) =
  let seed_mix = splitmix64 seed
  let mixed = map (\h -> splitmix64 (h ^ seed_mix)) data_hashes
  let (re_a, im_a, re_b, im_b) = graph_derive_qubit_states mixed
  let ne = n * 3
  let edge_srcs = tabulate ne (\flat_i ->
    let node_i = flat_i / 3
    let pred_k = flat_i % 3
    in if node_i > pred_k then node_i else -1i64)
  let edge_tgts = tabulate ne (\flat_i ->
    let node_i = flat_i / 3
    let pred_k = flat_i % 3
    in if node_i > pred_k then node_i - pred_k - 1 else -1i64)
  in (copy data_hashes, re_a, im_a, re_b, im_b, edge_srcs, edge_tgts)

entry embedding_sum_squares [vocab_size][dim] (source: [vocab_size][dim]f16) : f32 =
  let squared = map (\row ->
    map (\v ->
      let x = sanitize_f32 (f32.f16 v)
      in x * x) row) source
  let total = f32.sum (flatten squared)
  in sanitize_f32 total

entry rsf_stack_forward [batch_size][seq_len][half][num_layers]
  (inputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [num_layers][half][2]f16)
  (weights_t: [num_layers][half][2]f16)
  (clip_min: f16) (clip_max: f16)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : *[batch_size][seq_len][half*2]f16 =
  let raw_min = f32.f16 clip_min
  let raw_max = f32.f16 clip_max
  let clip_min_f32 = f32.min raw_min raw_max
  let clip_max_f32 = f32.max raw_min raw_max
  let flat = flatten inputs
  let out_rows = map (\row ->
    let row_f32 = map f32.f16 row
    let result = loop cur = row_f32 for l < num_layers do
      rsf_stack_coupling_row cur weights_s[l] weights_t[l] clip_min_f32 clip_max_f32 diffusion radix block stages
    in map (\v -> f16.f32 (clamp_f16_value v)) result) flat
  in copy (unflatten out_rows :> [batch_size][seq_len][half*2]f16)

entry rsf_stack_inverse [batch_size][seq_len][half][num_layers]
  (outputs: [batch_size][seq_len][half*2]f16)
  (weights_s: [num_layers][half][2]f16)
  (weights_t: [num_layers][half][2]f16)
  (clip_min: f16) (clip_max: f16)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : *[batch_size][seq_len][half*2]f16 =
  let raw_min = f32.f16 clip_min
  let raw_max = f32.f16 clip_max
  let clip_min_f32 = f32.min raw_min raw_max
  let clip_max_f32 = f32.max raw_min raw_max
  let flat = flatten outputs
  let out_rows = map (\row ->
    let row_f32 = map f32.f16 row
    let result = loop cur = row_f32 for i < num_layers do
      let l = num_layers - 1 - i
      in rsf_stack_invert_row cur weights_s[l] weights_t[l] clip_min_f32 clip_max_f32 diffusion radix block stages
    in map (\v -> f16.f32 (clamp_f16_value v)) result) flat
  in copy (unflatten out_rows :> [batch_size][seq_len][half*2]f16)

let rsf_forward_adjoint_token [half]
  (y_row: [half*2]f32)
  (g_row: [half*2]f32)
  (ws: [half][2]f16)
  (wt: [half][2]f16)
  (safe_clip_min: f32) (safe_clip_max: f32)
  (ld_shift: f32)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : ([half]f32, [half]f32, [half]f32, [half]f32, [half*2]f32, [half*2]f32, f32) =
  let d2 = half * 2
  let g_masked = map2 (\gv yv ->
    if f32.abs yv >= 60000f32 then 0f32 else sanitize_f32 gv) g_row y_row
  let y_d = apply_diffuse y_row diffusion radix block stages
  let g_d = apply_diffuse g_masked diffusion radix block stages
  let y1p = y_d[0:half] :> [half]f32
  let y2p = y_d[half:d2] :> [half]f32
  let g1p = g_d[0:half] :> [half]f32
  let g2p = g_d[half:d2] :> [half]f32
  let u1 = map2 (\a b -> (a + b) * oftb_scale_f32) y1p y2p
  let u2 = map2 (\a b -> (b - a) * oftb_scale_f32) y1p y2p
  let h1 = map2 (\a b -> (a + b) * oftb_scale_f32) g1p g2p
  let h2 = map2 (\a b -> (b - a) * oftb_scale_f32) g1p g2p
  let dy1_total = map (\j -> h1[j] + h2[j] * f32.f16 wt[j][0]) (iota half)
  let x2 = map (\dd ->
    let trans = f32.f16 wt[dd][0] * u1[dd] + f32.f16 wt[dd][1]
    in u2[dd] - sanitize_f32 trans) (iota half)
  let pre_scale = map (\dd -> f32.f16 ws[dd][0] * x2[dd] + f32.f16 ws[dd][1]) (iota half)
  let clipped = map (\p -> f32.max safe_clip_min (f32.min safe_clip_max p)) pre_scale
  let scale = map f32.exp clipped
  let x1 = map2 (\u_j s_j -> sanitize_f32 (u_j / s_j)) u1 scale
  let dx1 = map2 (\dt_j s_j -> sanitize_f32 (dt_j * s_j)) dy1_total scale
  let ds = map3 (\p dt_j u_j ->
    if p >= safe_clip_min && p <= safe_clip_max
    then sanitize_f32 (dt_j * u_j - ld_shift)
    else 0f32) pre_scale dy1_total u1
  let dx2 = map (\j -> sanitize_f32 (h2[j] + ds[j] * f32.f16 ws[j][0])) (iota half)
  let y_next = (x1 ++ x2) :> [half*2]f32
  let g_next = (dx1 ++ dx2) :> [half*2]f32
  let ld_tok = f32.sum clipped
  in (ds, h2, x2, u1, y_next, g_next, ld_tok)

let rsf_inverted_adjoint_token [half]
  (x_row: [half*2]f32)
  (g_row: [half*2]f32)
  (ws: [half][2]f16)
  (wt: [half][2]f16)
  (safe_clip_min: f32) (safe_clip_max: f32)
  (ld_shift: f32)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : ([half]f32, [half]f32, [half]f32, [half]f32, [half*2]f32, [half*2]f32, f32) =
  let d2 = half * 2
  let x1 = x_row[0:half] :> [half]f32
  let x2 = x_row[half:d2] :> [half]f32
  let g1 = map sanitize_f32 (g_row[0:half] :> [half]f32)
  let g2 = map sanitize_f32 (g_row[half:d2] :> [half]f32)
  let pre = map (\d -> f32.f16 ws[d][0] * x2[d] + f32.f16 ws[d][1]) (iota half)
  let clipped = map (\p -> f32.max safe_clip_min (f32.min safe_clip_max p)) pre
  let scale = map f32.exp clipped
  let y1 = map2 (*) x1 scale
  let y2 = map2 (\x2_j j ->
    let trans = f32.f16 wt[j][0] * y1[j] + f32.f16 wt[j][1]
    in x2_j + sanitize_f32 trans) x2 (iota half)
  let inv_scale = map (\s -> f32.exp (-s)) clipped
  let ds = map3 (\p g1_j x1_j ->
    if p >= safe_clip_min && p <= safe_clip_max
    then sanitize_f32 (-g1_j * x1_j - ld_shift)
    else 0f32) pre g1 x1
  let dx2 = map (\j -> sanitize_f32 (g2[j] + f32.f16 ws[j][0] * ds[j])) (iota half)
  let gy1 = map (\j -> sanitize_f32 (g1[j] * inv_scale[j] - f32.f16 wt[j][0] * dx2[j])) (iota half)
  let gy2 = dx2
  let dtw = map2 (\dx2_j y1_j -> sanitize_f32 (-dx2_j * y1_j)) dx2 y1
  let dtb = map (\v -> sanitize_f32 (-v)) dx2
  let o1 = map2 (\a b -> (a - b) * oftb_scale_f32) y1 y2
  let o2 = map2 (\a b -> (a + b) * oftb_scale_f32) y1 y2
  let y_joined = (o1 ++ o2) :> [half*2]f32
  let y_next = map clamp_f16_value (apply_diffuse y_joined diffusion radix block stages)
  let h1 = map2 (\a b -> (a - b) * oftb_scale_f32) gy1 gy2
  let h2 = map2 (\a b -> (a + b) * oftb_scale_f32) gy1 gy2
  let g_joined = (h1 ++ h2) :> [half*2]f32
  let g_next = apply_diffuse g_joined diffusion radix block stages
  let ld_tok = f32.sum clipped
  in (ds, dtw, dtb, y1, y_next, g_next, ld_tok)

entry rsf_stack_backward_gradients_fused [batch_size][seq_len][half][num_layers]
  (final_outputs: [batch_size][seq_len][half*2]f16)
  (targets: [batch_size][seq_len][half*2]f16)
  (originals: [batch_size][seq_len][half*2]f16)
  (lengths: [batch_size]i64)
  (weights_s: [num_layers][half][2]f16)
  (weights_t: [num_layers][half][2]f16)
  (grad_mean: bool)
  (gradient_scale: f32)
  (clip_min: f32)
  (clip_max: f32)
  (reconstruction_alpha: f32)
  (forward_scale: f32)
  (logdet_weight: f32)
  (diffusion: bool)
  (radix: i64) (block: i64) (stages: i64)
  : (*[num_layers][half][2]f32, *[num_layers][half][2]f32,
     *[batch_size][seq_len][half*2]f16,
     f32, f32, f32) =
  let d2 = half * 2
  let safe_clip_min = f32.min clip_min clip_max
  let safe_clip_max = f32.max clip_min clip_max
  let flat_final = flatten final_outputs
  let flat_targets = flatten targets
  let flat_orig = flatten originals
  let limits = map (\length -> i64.max 0 (i64.min seq_len length)) lengths
  let valid_tokens = i64.sum limits
  let count_elements = if valid_tokens > 0 then valid_tokens * d2 else 1
  let count_elements_f32 = f32.i64 count_elements
  let count_tokens_f32 = f32.max 1f32 (f32.i64 valid_tokens)
  let gradient_element_divisor = if grad_mean then count_elements_f32 else 1f32
  let gradient_token_divisor = if grad_mean then count_tokens_f32 else 1f32
  let ld_shift = sanitize_f32 (logdet_weight / gradient_token_divisor)
  let active_indices = filter (\t ->
    let b = t / seq_len
    let j = t % seq_len
    in j < limits[b]) (iota (batch_size * seq_len))
  let active_final = map (\t -> flat_final[t]) active_indices
  let active_targets = map (\t -> flat_targets[t]) active_indices
  let active_orig = map (\t -> flat_orig[t]) active_indices
  let initial_grads = map2 (\y t ->
    map2 (\yv tv ->
      let diff = f32.f16 yv - f32.f16 tv
      let safe_diff = f32.max (-100f32) (f32.min 100f32 (sanitize_f32 diff))
      in 2f32 * safe_diff / gradient_element_divisor) y t) active_final active_targets
  let y_start = map (map (\v -> sanitize_f32 (f32.f16 v))) active_final
  let gs_zero = replicate half (replicate 2 0f32)
  let (gs_stack, gt_stack, x_stack, g_stack, ld_stack) =
    loop (gs_acc, gt_acc, y_all, g_all, ld_all) =
      (replicate num_layers (copy gs_zero),
       replicate num_layers (copy gs_zero),
       y_start,
       initial_grads,
       map (\_ -> 0f32) y_start)
    for i < num_layers do
      let l = num_layers - 1 - i
      let ws = weights_s[l]
      let wt = weights_t[l]
      let per_tok = map2 (\y_row g_row ->
        rsf_forward_adjoint_token y_row g_row ws wt safe_clip_min safe_clip_max ld_shift diffusion radix block stages
      ) y_all g_all
      let ds_columns = transpose (map (\(ds, _, _, _, _, _, _) -> ds) per_tok)
      let h2_columns = transpose (map (\(_, h2, _, _, _, _, _) -> h2) per_tok)
      let x2_columns = transpose (map (\(_, _, x2, _, _, _, _) -> x2) per_tok)
      let u1_columns = transpose (map (\(_, _, _, u1, _, _, _) -> u1) per_tok)
      let gs_l_total = map2 (\ds_column x2_column ->
        [f32.sum (map2 (*) ds_column x2_column), f32.sum ds_column] :> [2]f32) ds_columns x2_columns
      let gt_l_total = map2 (\h2_column u1_column ->
        [f32.sum (map2 (*) h2_column u1_column), f32.sum h2_column] :> [2]f32) h2_columns u1_columns
      let y_next_raw = map (\(_, _, _, _, y_next, _, _) -> y_next) per_tok
      let y_next_all = if i < num_layers - 1
                       then map (map clamp_f16_value) y_next_raw
                       else y_next_raw
      let g_next_all = map (\(_, _, _, _, _, g_next, _) -> g_next) per_tok
      let ld_next = map2 (+) ld_all (map (\(_, _, _, _, _, _, ld_tok) -> ld_tok) per_tok)
      in (gs_acc with [l] = gs_l_total,
          gt_acc with [l] = gt_l_total,
          y_next_all,
          g_next_all,
          ld_next)
  let gs_normalized = map (map (map (\value -> sanitize_f32 (value * gradient_scale)))) gs_stack
  let gt_normalized = map (map (map (\value -> sanitize_f32 (value * gradient_scale)))) gt_stack
  let loss_total = reduce (+) 0f32 (map2 (\y t ->
    f32.sum (map2 (\yv tv ->
      let diff = f32.f16 yv - f32.f16 tv
      let safe = f32.max (-100f32) (f32.min 100f32 (sanitize_f32 diff))
      in safe * safe) y t)) active_final active_targets)
  let loss = loss_total / count_elements_f32
  let recon_total = reduce (+) 0f32 (map2 (\x_row o_row ->
    f32.sum (map2 (\xv ov ->
      let diff = xv - f32.f16 ov
      let safe = f32.max (-100f32) (f32.min 100f32 (sanitize_f32 diff))
      in safe * safe) x_row o_row)) x_stack active_orig)
  let recon_loss = recon_total / count_elements_f32
  let logdet_total = reduce (+) 0f32 ld_stack
  let logdet_mean = sanitize_f32 (logdet_total / count_tokens_f32)
  let active_input_delta = map3 (\g_row x_row o_row ->
    map3 (\gv xv ov ->
      let base = forward_scale * gradient_scale * sanitize_f32 gv
      let diff = xv - f32.f16 ov
      let safe_diff = f32.max (-100f32) (f32.min 100f32 (sanitize_f32 diff))
      let combined = sanitize_f32 (base + reconstruction_alpha * 2f32 * safe_diff / gradient_element_divisor)
      in f16.f32 (f32.max (-65504f32) (f32.min 65504f32 combined))) g_row x_row o_row
    ) g_stack x_stack active_orig
  let zero_delta = replicate (batch_size * seq_len) (replicate (half * 2) 0f16)
  let input_delta = scatter zero_delta active_indices active_input_delta
  let input_delta_3d = unflatten input_delta :> [batch_size][seq_len][half*2]f16
  in (copy gs_normalized, copy gt_normalized, input_delta_3d,
      f32.max 0f32 loss, f32.max 0f32 recon_loss, logdet_mean)

entry rsf_stack_midpoint_fused [batch_size][seq_len][half][num_layers]
  (inputs: [batch_size][seq_len][half*2]f16)
  (targets: [batch_size][seq_len][half*2]f16)
  (lengths: [batch_size]i64)
  (weights_s: [num_layers][half][2]f16)
  (weights_t: [num_layers][half][2]f16)
  (clip_min: f32) (clip_max: f32)
  (logdet_weight: f32)
  (diffusion: bool)
  (grad_mean: bool)
  (gradient_scale: f32)
  (radix: i64) (block: i64) (stages: i64)
  : (*[num_layers][half][2]f32, *[num_layers][half][2]f32,
     *[batch_size][seq_len][half*2]f16,
     f32, f32, f32) =
  let d2 = half * 2
  let m_layers = num_layers / 2
  let back_layers = num_layers - m_layers
  let safe_clip_min = f32.min clip_min clip_max
  let safe_clip_max = f32.max clip_min clip_max
  let flat_in = flatten inputs
  let flat_tg = flatten targets
  let limits = map (\length -> i64.max 0 (i64.min seq_len length)) lengths
  let valid_tokens = i64.sum limits
  let count_elements = if valid_tokens > 0 then valid_tokens * d2 else 1
  let count_elements_f32 = f32.i64 count_elements
  let count_tokens_f32 = f32.max 1f32 (f32.i64 valid_tokens)
  let gradient_element_divisor = if grad_mean then count_elements_f32 else 1f32
  let gradient_token_divisor = if grad_mean then count_tokens_f32 else 1f32
  let ld_shift = sanitize_f32 (logdet_weight / gradient_token_divisor)
  let active_indices = filter (\t ->
    let b = t / seq_len
    let j = t % seq_len
    in j < limits[b]) (iota (batch_size * seq_len))
  let active_in = map (\t -> map (\v -> sanitize_f32 (f32.f16 v)) flat_in[t]) active_indices
  let active_tg = map (\t -> map (\v -> sanitize_f32 (f32.f16 v)) flat_tg[t]) active_indices
  let fwd_pairs = map (\row ->
    loop (cur, acc) = (row, 0f32) for l < m_layers do
      let (nxt, layer_ld) = rsf_stack_coupling_row_ld cur weights_s[l] weights_t[l] safe_clip_min safe_clip_max diffusion radix block stages
      in (nxt, acc + layer_ld)
  ) active_in
  let z_rows = map (\(cur, _) -> cur) fwd_pairs
  let ld_fwd = map (\(_, ld) -> ld) fwd_pairs
  let bwd_pairs = map (\row ->
    loop (cur, acc) = (row, 0f32) for i < back_layers do
      let l = num_layers - 1 - i
      let (nxt, layer_ld) = rsf_stack_invert_row_ld cur weights_s[l] weights_t[l] safe_clip_min safe_clip_max diffusion radix block stages
      in (nxt, acc + layer_ld)
  ) active_tg
  let w_rows = map (\(cur, _) -> cur) bwd_pairs
  let ld_bwd = map (\(_, ld) -> ld) bwd_pairs
  let gz = map2 (\z w ->
    map2 (\zv wv ->
      let diff = zv - wv
      let safe = f32.max (-100f32) (f32.min 100f32 (sanitize_f32 diff))
      in 2f32 * safe / gradient_element_divisor) z w) z_rows w_rows
  let gw = map (map (\v -> -v)) gz
  let collision_total = reduce (+) 0f32 (map2 (\z w ->
    f32.sum (map2 (\zv wv ->
      let diff = zv - wv
      let safe = f32.max (-100f32) (f32.min 100f32 (sanitize_f32 diff))
      in safe * safe) z w)) z_rows w_rows)
  let collision_loss = collision_total / count_elements_f32
  let logdet_forward_mean = sanitize_f32 ((reduce (+) 0f32 ld_fwd) / count_tokens_f32)
  let logdet_backward_mean = sanitize_f32 ((reduce (+) 0f32 ld_bwd) / count_tokens_f32)
  let gs_zero = replicate half (replicate 2 0f32)
  let (gs_fwd, gt_fwd, x_final, g_final) =
    loop (gs_acc, gt_acc, y_all, g_all) =
      (replicate num_layers (copy gs_zero),
       replicate num_layers (copy gs_zero),
       z_rows,
       gz)
    for i < m_layers do
      let l = m_layers - 1 - i
      let ws = weights_s[l]
      let wt = weights_t[l]
      let per_tok = map2 (\y_row g_row ->
        rsf_forward_adjoint_token y_row g_row ws wt safe_clip_min safe_clip_max ld_shift diffusion radix block stages
      ) y_all g_all
      let ds_columns = transpose (map (\(ds, _, _, _, _, _, _) -> ds) per_tok)
      let h2_columns = transpose (map (\(_, h2, _, _, _, _, _) -> h2) per_tok)
      let x2_columns = transpose (map (\(_, _, x2, _, _, _, _) -> x2) per_tok)
      let u1_columns = transpose (map (\(_, _, _, u1, _, _, _) -> u1) per_tok)
      let gs_l_total = map2 (\ds_column x2_column ->
        [f32.sum (map2 (*) ds_column x2_column), f32.sum ds_column] :> [2]f32) ds_columns x2_columns
      let gt_l_total = map2 (\h2_column u1_column ->
        [f32.sum (map2 (*) h2_column u1_column), f32.sum h2_column] :> [2]f32) h2_columns u1_columns
      let y_next_raw = map (\(_, _, _, _, y_next, _, _) -> y_next) per_tok
      let y_next_all = if i < m_layers - 1 then map (map clamp_f16_value) y_next_raw else y_next_raw
      let g_next_all = map (\(_, _, _, _, _, g_next, _) -> g_next) per_tok
      in (gs_acc with [l] = gs_l_total,
          gt_acc with [l] = gt_l_total,
          y_next_all,
          g_next_all)
  let (gs_bwd, gt_bwd) =
    loop (gs_acc, gt_acc, y_all, g_all) =
      (replicate num_layers (copy gs_zero),
       replicate num_layers (copy gs_zero),
       w_rows,
       gw)
    for i < back_layers do
      let l = m_layers + i
      let ws = weights_s[l]
      let wt = weights_t[l]
      let per_tok = map2 (\x_row g_row ->
        rsf_inverted_adjoint_token x_row g_row ws wt safe_clip_min safe_clip_max ld_shift diffusion radix block stages
      ) y_all g_all
      let ds_columns = transpose (map (\(ds, _, _, _, _, _, _) -> ds) per_tok)
      let dtw_columns = transpose (map (\(_, dtw, _, _, _, _, _) -> dtw) per_tok)
      let dtb_columns = transpose (map (\(_, _, dtb, _, _, _, _) -> dtb) per_tok)
      let x2_from_x = transpose (map (\x_row -> x_row[half:d2] :> [half]f32) y_all)
      let gs_l_total = map2 (\ds_column x2_column ->
        [f32.sum (map2 (*) ds_column x2_column), f32.sum ds_column] :> [2]f32) ds_columns x2_from_x
      let gt_l_total = map2 (\dtw_column dtb_column ->
        [f32.sum dtw_column, f32.sum dtb_column] :> [2]f32) dtw_columns dtb_columns
      let y_next_all = map (\(_, _, _, _, y_next, _, _) -> y_next) per_tok
      let g_next_all = map (\(_, _, _, _, _, g_next, _) -> g_next) per_tok
      in (gs_acc with [l] = gs_l_total,
          gt_acc with [l] = gt_l_total,
          y_next_all,
          g_next_all)
  let gs_stack = tabulate num_layers (\l -> if l < m_layers then gs_fwd[l] else gs_bwd[l])
  let gt_stack = tabulate num_layers (\l -> if l < m_layers then gt_fwd[l] else gt_bwd[l])
  let gs_normalized = map (map (map (\value -> sanitize_f32 (value * gradient_scale)))) gs_stack
  let gt_normalized = map (map (map (\value -> sanitize_f32 (value * gradient_scale)))) gt_stack
  let active_input_delta = map (\g_row ->
    map (\gv ->
      let combined = sanitize_f32 (gradient_scale * sanitize_f32 gv)
      in f16.f32 (f32.max (-65504f32) (f32.min 65504f32 combined))) g_row
  ) g_final
  let zero_delta = replicate (batch_size * seq_len) (replicate (half * 2) 0f16)
  let input_delta = scatter zero_delta active_indices active_input_delta
  let input_delta_3d = unflatten input_delta :> [batch_size][seq_len][half*2]f16
  in (copy gs_normalized, copy gt_normalized, input_delta_3d,
      f32.max 0f32 collision_loss, logdet_forward_mean, logdet_backward_mean)

let causal_key_row [seq_len][half]
  (x2: [seq_len][half]f32)
  (bitmask: [seq_len][seq_len]u8)
  (t: i64)
  : [half]f32 =
  map (\d ->
    let acc = loop s = 0f32 for t' < seq_len do
      s + f32.bool (bitmask[t][t'] != 0u8) * x2[t'][d]
    in acc + x2[t][d]) (iota half)

let causal_oftb_forward_row [half]
  (y1: [half]f32) (y2: [half]f32)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : [half*2]f32 =
  let o1 = map2 (\a b -> (a - b) * oftb_scale_f32) y1 y2
  let o2 = map2 (\a b -> (a + b) * oftb_scale_f32) y1 y2
  let joined = (o1 ++ o2) :> [half*2]f32
  in map clamp_f16_value (apply_diffuse joined diffusion radix block stages)

let causal_oftb_undo_row [half]
  (row: [half*2]f32)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : ([half]f32, [half]f32) =
  let y_d = apply_diffuse row diffusion radix block stages
  let y1p = y_d[0:half] :> [half]f32
  let y2p = y_d[half:half*2] :> [half]f32
  let u1 = map2 (\a b -> (a + b) * oftb_scale_f32) y1p y2p
  let u2 = map2 (\a b -> (b - a) * oftb_scale_f32) y1p y2p
  in (u1, u2)

let rsf_causal_bitmask_forward_seq [seq_len][half]
  (x: [seq_len][half*2]f32)
  (bitmask: [seq_len][seq_len]u8)
  (weights_s: [half][2]f16)
  (weights_t: [half][2]f16)
  (clip_min: f32) (clip_max: f32)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : [seq_len][half*2]f32 =
  let x1 = map (\row -> row[0:half] :> [half]f32) x
  let x2 = map (\row -> row[half:half*2] :> [half]f32) x
  let k = map (\t -> causal_key_row x2 bitmask t) (iota seq_len)
  in map (\t ->
    let pre = map (\d -> f32.f16 weights_s[d][0] * k[t][d] + f32.f16 weights_s[d][1]) (iota half)
    let clipped = map (\p -> f32.max clip_min (f32.min clip_max p)) pre
    let y1 = map2 (*) x1[t] (map f32.exp clipped)
    let y2 = map (\d ->
      let trans = f32.f16 weights_t[d][0] * y1[d] + f32.f16 weights_t[d][1]
      in x2[t][d] + sanitize_f32 trans) (iota half)
    in causal_oftb_forward_row y1 y2 diffusion radix block stages) (iota seq_len)

let rsf_causal_bitmask_inverse_seq [seq_len][half]
  (y: [seq_len][half*2]f32)
  (bitmask: [seq_len][seq_len]u8)
  (weights_s: [half][2]f16)
  (weights_t: [half][2]f16)
  (clip_min: f32) (clip_max: f32)
  (diffusion: bool) (radix: i64) (block: i64) (stages: i64)
  : [seq_len][half*2]f32 =
  let undone = map (\row -> causal_oftb_undo_row row diffusion radix block stages) y
  let u1 = map (\(a, _) -> a) undone
  let u2 = map (\(_, b) -> b) undone
  let x2 = map (\t ->
    map (\d ->
      let trans = f32.f16 weights_t[d][0] * u1[t][d] + f32.f16 weights_t[d][1]
      in u2[t][d] - sanitize_f32 trans) (iota half)) (iota seq_len)
  let k = map (\t -> causal_key_row x2 bitmask t) (iota seq_len)
  in map (\t ->
    let pre = map (\d -> f32.f16 weights_s[d][0] * k[t][d] + f32.f16 weights_s[d][1]) (iota half)
    let clipped = map (\p -> f32.max clip_min (f32.min clip_max p)) pre
    let x1 = map2 (\u s -> sanitize_f32 (u / f32.exp s)) u1[t] clipped
    in map clamp_f16_value ((x1 ++ x2[t]) :> [half*2]f32)) (iota seq_len)

entry rsf_causal_bitmask_forward [batch][seq_len][half]
  (x: [batch][seq_len][half*2]f16)
  (bitmask: [seq_len][seq_len]u8)
  (weights_s: [half][2]f16)
  (weights_t: [half][2]f16)
  (clip_min: f32) (clip_max: f32)
  (diffusion: bool)
  (radix: i64) (block: i64) (stages: i64)
  : *[batch][seq_len][half*2]f16 =
  let safe_min = f32.min clip_min clip_max
  let safe_max = f32.max clip_min clip_max
  in copy (map (\seq ->
    let xf = map (map (\v -> sanitize_f32 (f32.f16 v))) seq
    let y = rsf_causal_bitmask_forward_seq xf bitmask weights_s weights_t safe_min safe_max diffusion radix block stages
    in map (map (\v -> f16.f32 (clamp_f16_value v))) y) x)

entry rsf_causal_bitmask_inverse [batch][seq_len][half]
  (y: [batch][seq_len][half*2]f16)
  (bitmask: [seq_len][seq_len]u8)
  (weights_s: [half][2]f16)
  (weights_t: [half][2]f16)
  (clip_min: f32) (clip_max: f32)
  (diffusion: bool)
  (radix: i64) (block: i64) (stages: i64)
  : *[batch][seq_len][half*2]f16 =
  let safe_min = f32.min clip_min clip_max
  let safe_max = f32.max clip_min clip_max
  in copy (map (\seq ->
    let yf = map (map (\v -> sanitize_f32 (f32.f16 v))) seq
    let xrec = rsf_causal_bitmask_inverse_seq yf bitmask weights_s weights_t safe_min safe_max diffusion radix block stages
    in map (map (\v -> f16.f32 (clamp_f16_value v))) xrec) y)

entry rsf_causal_bitmask_forward_stack [batch][seq_len][half][num_layers]
  (x: [batch][seq_len][half*2]f16)
  (bitmask: [seq_len][seq_len]u8)
  (weights_s: [num_layers][half][2]f16)
  (weights_t: [num_layers][half][2]f16)
  (clip_min: f32) (clip_max: f32)
  (diffusion: bool)
  (radix: i64) (block: i64) (stages: i64)
  : *[batch][seq_len][half*2]f16 =
  let safe_min = f32.min clip_min clip_max
  let safe_max = f32.max clip_min clip_max
  in copy (map (\seq ->
    let xf = map (map (\v -> sanitize_f32 (f32.f16 v))) seq
    let y = loop cur = xf for l < num_layers do
      rsf_causal_bitmask_forward_seq cur bitmask weights_s[l] weights_t[l] safe_min safe_max diffusion radix block stages
    in map (map (\v -> f16.f32 (clamp_f16_value v))) y) x)

entry rsf_causal_bitmask_inverse_stack [batch][seq_len][half][num_layers]
  (y: [batch][seq_len][half*2]f16)
  (bitmask: [seq_len][seq_len]u8)
  (weights_s: [num_layers][half][2]f16)
  (weights_t: [num_layers][half][2]f16)
  (clip_min: f32) (clip_max: f32)
  (diffusion: bool)
  (radix: i64) (block: i64) (stages: i64)
  : *[batch][seq_len][half*2]f16 =
  let safe_min = f32.min clip_min clip_max
  let safe_max = f32.max clip_min clip_max
  in copy (map (\seq ->
    let yf = map (map (\v -> sanitize_f32 (f32.f16 v))) seq
    let xrec = loop cur = yf for i < num_layers do
      let l = num_layers - 1 - i
      in rsf_causal_bitmask_inverse_seq cur bitmask weights_s[l] weights_t[l] safe_min safe_max diffusion radix block stages
    in map (map (\v -> f16.f32 (clamp_f16_value v))) xrec) y)
