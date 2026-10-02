pragma circom 2.1.8;

include "circomlib/circuits/poseidon.circom";
include "circomlib/circuits/comparators.circom";
include "circomlib/circuits/bitify.circom";
include "circomlib/circuits/mux1.circom";

function TWO_POW(n) {
    assert(n >= 0);
    assert(n < 254);
    var result = 1;
    for (var i = 0; i < n; i++) {
        result = result * 2;
    }
    return result;
}

function CEIL_LOG2(n) {
    assert(n > 0);
    var value = 1;
    var result = 0;
    while (value < n) {
        value = value * 2;
        result = result + 1;
    }
    return result;
}

function MAX_OF(a, b) {
    if (a > b) {
        return a;
    }
    return b;
}

function IS_POWER_OF_TWO(n) {
    if (n < 1) {
        return 0;
    }
    var value = 1;
    while (value < n) {
        value = value * 2;
    }
    if (value == n) {
        return 1;
    }
    return 0;
}

function FIXED_POINT_BITS() {
    return 20;
}

function FIXED_POINT_SCALE() {
    return 1048576;
}

function EXP_SCALE_BITS() {
    return 40;
}

function EXP_GROUP() {
    return 3;
}

function VALUE_BITS() {
    return 36;
}

function VALUE_LIMIT() {
    return 62914560000;
}

function WEIGHT_LIMIT() {
    return 68685922304;
}

function OFTB_SCALE_FIXED() {
    return 741455;
}

function EXP_MAGNITUDE_BITS(n) {
    assert(n >= 0);
    return (n * 1442696) \ 1000000 + 1;
}

function EXP_ACC_BITS(clip_max) {
    return EXP_MAGNITUDE_BITS(clip_max) + EXP_SCALE_BITS() + 1;
}

function EXP_OUT_BITS(clip_max) {
    return EXP_MAGNITUDE_BITS(clip_max) + FIXED_POINT_BITS() + 1;
}

function EXP_BIT_FACTOR(k) {
    assert(k >= 0);
    assert(k < 26);
    var table[26] = [
        1099512676353,
        1099513724930,
        1099515822088,
        1099520016416,
        1099528405120,
        1099545182720,
        1099578738688,
        1099645853696,
        1099780096003,
        1100048629781,
        1100585894059,
        1101661209942,
        1103814994613,
        1108135204352,
        1116826416478,
        1134413873426,
        1170424035283,
        1245909900143,
        1411800876008,
        1812788208096,
        2988782477963,
        8124353099063,
        60031300816501,
        3277597968664119,
        9770381843001049673,
        86820692884470725459698192
    ];
    return table[k];
}

function EXP_NEG_INT(n) {
    assert(n >= 0);
    assert(n < 21);
    var table[21] = [
        1099511627776,
        404487723188,
        148802717567,
        54741460583,
        20138257928,
        7408451073,
        2725416841,
        1002624824,
        368845060,
        135690515,
        49917751,
        18363714,
        6755633,
        2485258,
        914275,
        336343,
        123734,
        45519,
        16746,
        6160,
        2266
    ];
    return table[n];
}

template UnsignedShift(bits, shift) {
    assert(shift > 0);
    assert(bits > shift);
    assert(bits < 254);

    signal input in;
    signal output out;

    component decomp = Num2Bits(bits);
    decomp.in <== in;

    var accumulated = 0;
    var weight = 1;

    for (var i = shift; i < bits; i++) {
        accumulated += decomp.out[i] * weight;
        weight = weight * 2;
    }

    out <== accumulated;
}

template SignedShift(bits, shift) {
    assert(shift > 0);
    assert(bits > shift);
    assert(bits + 1 < 254);

    signal input in;
    signal output out;

    component decomp = Num2Bits(bits + 1);
    decomp.in <== in + TWO_POW(bits);

    var accumulated = 0;
    var weight = 1;

    for (var i = shift; i <= bits; i++) {
        accumulated += decomp.out[i] * weight;
        weight = weight * 2;
    }

    out <== accumulated - TWO_POW(bits - shift);
}

template SignedRange(bits) {
    assert(bits > 0);
    assert(bits + 1 < 254);

    signal input in;
    signal output abs;
    signal output is_negative;

    component decomp = Num2Bits(bits + 1);
    decomp.in <== in + TWO_POW(bits);

    is_negative <== 1 - decomp.out[bits];

    signal negated;
    negated <== 0 - in;

    component selector = Mux1();
    selector.c[0] <== in;
    selector.c[1] <== negated;
    selector.s <== is_negative;

    abs <== selector.out;
}

template SignedLess(bits) {
    assert(bits > 0);
    assert(bits + 1 < 254);

    signal input a;
    signal input b;
    signal output out;

    component decomp = Num2Bits(bits + 1);
    decomp.in <== a - b + TWO_POW(bits);

    out <== 1 - decomp.out[bits];
}

template ClampToRange(bits, lower_magnitude, upper) {
    assert(lower_magnitude >= 0);
    assert(upper >= 0);
    assert(lower_magnitude + upper > 0);

    signal input in;
    signal output out;

    component below = SignedLess(bits);
    below.a <== in;
    below.b <== 0 - lower_magnitude;

    component lower_mux = Mux1();
    lower_mux.c[0] <== in;
    lower_mux.c[1] <== 0 - lower_magnitude;
    lower_mux.s <== below.out;

    signal lower_bounded;
    lower_bounded <== lower_mux.out;

    component above = SignedLess(bits);
    above.a <== upper;
    above.b <== lower_bounded;

    component upper_mux = Mux1();
    upper_mux.c[0] <== lower_bounded;
    upper_mux.c[1] <== upper;
    upper_mux.s <== above.out;

    out <== upper_mux.out;
}

template FixedExp(clip_min_magnitude, clip_max) {
    assert(clip_min_magnitude >= 0);
    assert(clip_min_magnitude <= 20);
    assert(clip_max >= 0);
    assert(clip_max <= 20);
    assert(clip_min_magnitude + clip_max > 0);

    signal input in;
    signal output out;

    var scale = FIXED_POINT_SCALE();
    var span = (clip_min_magnitude + clip_max) * scale;
    var u_bits = CEIL_LOG2(span + 1);

    assert(u_bits <= 26);

    signal shifted_input;
    shifted_input <== in + clip_min_magnitude * scale;

    component input_decomp = Num2Bits(u_bits);
    input_decomp.in <== shifted_input;

    component span_check = LessThan(u_bits + 1);
    span_check.in[0] <== shifted_input;
    span_check.in[1] <== span + 1;
    span_check.out === 1;

    component factor_mux[u_bits];
    signal factor[u_bits];

    for (var k = 0; k < u_bits; k++) {
        factor_mux[k] = Mux1();
        factor_mux[k].c[0] <== TWO_POW(EXP_SCALE_BITS());
        factor_mux[k].c[1] <== EXP_BIT_FACTOR(k);
        factor_mux[k].s <== input_decomp.out[k];
        factor[k] <== factor_mux[k].out;
    }

    var group = EXP_GROUP();
    var num_groups = (u_bits + group - 1) \ group;
    var acc_bits = EXP_ACC_BITS(clip_max);

    signal accumulator[num_groups + 1];
    signal partial[num_groups][group + 1];
    component normalize[num_groups];

    accumulator[0] <== EXP_NEG_INT(clip_min_magnitude);

    for (var g = 0; g < num_groups; g++) {
        var used = u_bits - g * group;

        if (used > group) {
            used = group;
        }

        partial[g][0] <== accumulator[g];

        for (var j = 0; j < group; j++) {
            if (j < used) {
                partial[g][j + 1] <== partial[g][j] * factor[g * group + j];
            } else {
                partial[g][j + 1] <== partial[g][j];
            }
        }

        normalize[g] = UnsignedShift(acc_bits + used * EXP_SCALE_BITS(), used * EXP_SCALE_BITS());
        normalize[g].in <== partial[g][used];

        accumulator[g + 1] <== normalize[g].out;
    }

    component rescale = UnsignedShift(acc_bits, EXP_SCALE_BITS() - FIXED_POINT_BITS());
    rescale.in <== accumulator[num_groups];

    out <== rescale.out;
}

template PoseidonCommit() {
    signal input value;
    signal input blinding;
    signal output commitment;

    component hasher = Poseidon(2);
    hasher.inputs[0] <== value;
    hasher.inputs[1] <== blinding;

    commitment <== hasher.out;
}

template PoseidonChain(n) {
    assert(n > 0);

    signal input in[n];
    signal output out;

    var num_chunks = (n + 5) \ 6;

    signal intermediate[num_chunks];
    component hashers[num_chunks];

    for (var chunk = 0; chunk < num_chunks; chunk++) {
        hashers[chunk] = Poseidon(8);

        for (var j = 0; j < 6; j++) {
            if (chunk * 6 + j < n) {
                hashers[chunk].inputs[j] <== in[chunk * 6 + j];
            } else {
                hashers[chunk].inputs[j] <== 0;
            }
        }

        if (chunk > 0) {
            hashers[chunk].inputs[6] <== intermediate[chunk - 1];
        } else {
            hashers[chunk].inputs[6] <== 0;
        }

        hashers[chunk].inputs[7] <== n;

        intermediate[chunk] <== hashers[chunk].out;
    }

    out <== intermediate[num_chunks - 1];
}

template VerifyMerkleProof(depth) {
    assert(depth > 0);

    signal input leaf;
    signal input path_elements[depth];
    signal input path_indices[depth];
    signal output root;

    signal hashes[depth + 1];
    hashes[0] <== leaf;

    component hashers[depth];
    component muxers_left[depth];
    component muxers_right[depth];

    for (var i = 0; i < depth; i++) {
        path_indices[i] * (path_indices[i] - 1) === 0;

        muxers_left[i] = Mux1();
        muxers_left[i].c[0] <== hashes[i];
        muxers_left[i].c[1] <== path_elements[i];
        muxers_left[i].s <== path_indices[i];

        muxers_right[i] = Mux1();
        muxers_right[i].c[0] <== path_elements[i];
        muxers_right[i].c[1] <== hashes[i];
        muxers_right[i].s <== path_indices[i];

        hashers[i] = Poseidon(2);
        hashers[i].inputs[0] <== muxers_left[i].out;
        hashers[i].inputs[1] <== muxers_right[i].out;

        hashes[i + 1] <== hashers[i].out;
    }

    root <== hashes[depth];
}

template VerifyBatchInference(batch_size) {
    assert(batch_size >= 2);
    assert(IS_POWER_OF_TWO(batch_size) == 1);

    signal input leaves[batch_size];
    signal input expected_root;

    signal nodes[2 * batch_size];
    component hashers[batch_size - 1];

    nodes[0] <== 0;

    for (var i = 0; i < batch_size; i++) {
        nodes[batch_size + i] <== leaves[i];
    }

    for (var i = batch_size - 1; i >= 1; i--) {
        hashers[i - 1] = Poseidon(2);
        hashers[i - 1].inputs[0] <== nodes[2 * i];
        hashers[i - 1].inputs[1] <== nodes[2 * i + 1];
        nodes[i] <== hashers[i - 1].out;
    }

    nodes[1] === expected_root;
}

template VerifyNoiseBound(dim, precision_bits) {
    assert(dim > 0);
    assert(precision_bits > 0);
    assert(precision_bits + 2 < 254);

    signal input original[dim];
    signal input noisy[dim];
    signal input max_noise;

    signal noise[dim];

    component abs_components[dim];
    component bound_checks[dim];

    for (var i = 0; i < dim; i++) {
        noise[i] <== noisy[i] - original[i];

        abs_components[i] = SignedRange(precision_bits);
        abs_components[i].in <== noise[i];

        bound_checks[i] = LessThan(precision_bits + 1);
        bound_checks[i].in[0] <== abs_components[i].abs;
        bound_checks[i].in[1] <== max_noise + 1;
        bound_checks[i].out === 1;
    }
}

template RSFCouplingLayer(dim, clip_min_magnitude, clip_max, value_bits) {
    assert(dim > 1);
    assert(dim % 2 == 0);
    assert(value_bits > FIXED_POINT_BITS());

    signal input x[dim];
    signal input weights_s[dim \ 2][2];
    signal input weights_t[dim \ 2][2];
    signal output y[dim];

    var half = dim \ 2;
    var fixed_bits = FIXED_POINT_BITS();
    var scale_bits = EXP_OUT_BITS(clip_max);
    var pre_product_bits = 2 * value_bits;
    var pre_bits = pre_product_bits - fixed_bits + 1;
    var y1_product_bits = value_bits + scale_bits;
    var y1_bits = y1_product_bits - fixed_bits + 1;
    var translation_product_bits = value_bits + y1_bits;
    var translation_bits = translation_product_bits - fixed_bits + 2;
    var y2_bits = MAX_OF(value_bits, translation_bits) + 1;
    var sum_bits = MAX_OF(y1_bits, y2_bits) + 1;
    var oftb_product_bits = sum_bits + fixed_bits;
    var out_bits = sum_bits + 1;

    signal x1[half];
    signal x2[half];

    for (var i = 0; i < half; i++) {
        x1[i] <== x[i];
        x2[i] <== x[half + i];
    }

    signal pre_product[half];
    signal pre_activation[half];
    signal scale[half];
    signal y1_product[half];
    signal y1[half];
    signal translation_product[half];
    signal translation[half];
    signal y2[half];
    signal butterfly_difference[half];
    signal butterfly_sum[half];
    signal butterfly_low_product[half];
    signal butterfly_high_product[half];

    component pre_shift[half];
    component pre_clamp[half];
    component exp_scale[half];
    component y1_shift[half];
    component translation_shift[half];
    component butterfly_low_shift[half];
    component butterfly_high_shift[half];
    component butterfly_low_clamp[half];
    component butterfly_high_clamp[half];

    for (var i = 0; i < half; i++) {
        pre_product[i] <== weights_s[i][0] * x2[i];

        pre_shift[i] = SignedShift(pre_product_bits, fixed_bits);
        pre_shift[i].in <== pre_product[i];

        pre_activation[i] <== pre_shift[i].out + weights_s[i][1];

        pre_clamp[i] = ClampToRange(pre_bits + 1, clip_min_magnitude * FIXED_POINT_SCALE(), clip_max * FIXED_POINT_SCALE());
        pre_clamp[i].in <== pre_activation[i];

        exp_scale[i] = FixedExp(clip_min_magnitude, clip_max);
        exp_scale[i].in <== pre_clamp[i].out;

        scale[i] <== exp_scale[i].out;

        y1_product[i] <== x1[i] * scale[i];

        y1_shift[i] = SignedShift(y1_product_bits, fixed_bits);
        y1_shift[i].in <== y1_product[i];

        y1[i] <== y1_shift[i].out;

        translation_product[i] <== weights_t[i][0] * y1[i];

        translation_shift[i] = SignedShift(translation_product_bits, fixed_bits);
        translation_shift[i].in <== translation_product[i];

        translation[i] <== translation_shift[i].out + weights_t[i][1];

        y2[i] <== x2[i] + translation[i];

        butterfly_difference[i] <== y1[i] - y2[i];
        butterfly_sum[i] <== y1[i] + y2[i];

        butterfly_low_product[i] <== butterfly_difference[i] * OFTB_SCALE_FIXED();
        butterfly_high_product[i] <== butterfly_sum[i] * OFTB_SCALE_FIXED();

        butterfly_low_shift[i] = SignedShift(oftb_product_bits, fixed_bits);
        butterfly_low_shift[i].in <== butterfly_low_product[i];

        butterfly_high_shift[i] = SignedShift(oftb_product_bits, fixed_bits);
        butterfly_high_shift[i].in <== butterfly_high_product[i];

        butterfly_low_clamp[i] = ClampToRange(out_bits + 1, VALUE_LIMIT(), VALUE_LIMIT());
        butterfly_low_clamp[i].in <== butterfly_low_shift[i].out;

        butterfly_high_clamp[i] = ClampToRange(out_bits + 1, VALUE_LIMIT(), VALUE_LIMIT());
        butterfly_high_clamp[i].in <== butterfly_high_shift[i].out;

        y[i] <== butterfly_low_clamp[i].out;
        y[half + i] <== butterfly_high_clamp[i].out;
    }
}

template FullInferenceProof(num_layers, dim, clip_min_magnitude, clip_max) {
    assert(num_layers > 0);
    assert(dim > 1);
    assert(dim % 2 == 0);

    var value_bits = VALUE_BITS();
    var half = dim \ 2;
    var error_bits = 2 * (value_bits + 1) + CEIL_LOG2(dim);

    signal input tokens[dim];
    signal input token_blinding;
    signal input output_blinding;
    signal input reference_blinding;
    signal input weights_s[num_layers][half][2];
    signal input weights_t[num_layers][half][2];
    signal input expected_output[dim];
    signal input max_error_squared;
    signal output y[dim];
    signal output input_commitment;
    signal output output_commitment;
    signal output reference_commitment;

    component token_range[dim];
    component token_limit[dim];
    component expected_range[dim];
    component expected_limit[dim];

    for (var i = 0; i < dim; i++) {
        token_range[i] = SignedRange(value_bits);
        token_range[i].in <== tokens[i];

        token_limit[i] = LessThan(value_bits + 1);
        token_limit[i].in[0] <== token_range[i].abs;
        token_limit[i].in[1] <== VALUE_LIMIT() + 1;
        token_limit[i].out === 1;

        expected_range[i] = SignedRange(value_bits);
        expected_range[i].in <== expected_output[i];

        expected_limit[i] = LessThan(value_bits + 1);
        expected_limit[i].in[0] <== expected_range[i].abs;
        expected_limit[i].in[1] <== VALUE_LIMIT() + 1;
        expected_limit[i].out === 1;
    }

    signal states[num_layers + 1][dim];

    for (var i = 0; i < dim; i++) {
        states[0][i] <== tokens[i];
    }

    component layers[num_layers];

    for (var layer = 0; layer < num_layers; layer++) {
        layers[layer] = RSFCouplingLayer(dim, clip_min_magnitude, clip_max, value_bits);

        for (var i = 0; i < dim; i++) {
            layers[layer].x[i] <== states[layer][i];
        }

        for (var i = 0; i < half; i++) {
            layers[layer].weights_s[i][0] <== weights_s[layer][i][0];
            layers[layer].weights_s[i][1] <== weights_s[layer][i][1];
            layers[layer].weights_t[i][0] <== weights_t[layer][i][0];
            layers[layer].weights_t[i][1] <== weights_t[layer][i][1];
        }

        for (var i = 0; i < dim; i++) {
            states[layer + 1][i] <== layers[layer].y[i];
        }
    }

    for (var i = 0; i < dim; i++) {
        y[i] <== states[num_layers][i];
    }

    component input_chain = PoseidonChain(dim);
    component output_chain = PoseidonChain(dim);
    component reference_chain = PoseidonChain(dim);

    for (var i = 0; i < dim; i++) {
        input_chain.in[i] <== tokens[i];
        output_chain.in[i] <== states[num_layers][i];
        reference_chain.in[i] <== expected_output[i];
    }

    component input_commit = PoseidonCommit();
    input_commit.value <== input_chain.out;
    input_commit.blinding <== token_blinding;

    component output_commit = PoseidonCommit();
    output_commit.value <== output_chain.out;
    output_commit.blinding <== output_blinding;

    component reference_commit = PoseidonCommit();
    reference_commit.value <== reference_chain.out;
    reference_commit.blinding <== reference_blinding;

    input_commitment <== input_commit.commitment;
    output_commitment <== output_commit.commitment;
    reference_commitment <== reference_commit.commitment;

    signal difference[dim];
    signal squared_difference[dim];
    signal error_accumulator[dim + 1];

    component difference_range[dim];

    error_accumulator[0] <== 0;

    for (var i = 0; i < dim; i++) {
        difference[i] <== states[num_layers][i] - expected_output[i];

        difference_range[i] = SignedRange(value_bits + 1);
        difference_range[i].in <== difference[i];

        squared_difference[i] <== difference_range[i].abs * difference_range[i].abs;

        error_accumulator[i + 1] <== error_accumulator[i] + squared_difference[i];
    }

    component error_check = LessThan(error_bits + 1);
    error_check.in[0] <== error_accumulator[dim];
    error_check.in[1] <== max_error_squared + 1;
    error_check.out === 1;
}

template InferenceTraceWithBatch(batch_size, num_layers, dim, clip_min_magnitude, clip_max, model_depth) {
    assert(batch_size >= 2);
    assert(IS_POWER_OF_TWO(batch_size) == 1);
    assert(num_layers > 0);
    assert(dim > 1);
    assert(dim % 2 == 0);
    assert(model_depth > 0);

    var value_bits = VALUE_BITS();
    var half = dim \ 2;
    var weight_count = num_layers * half * 4;
    var error_bits = 2 * (value_bits + 1) + CEIL_LOG2(dim);

    signal input tokens[batch_size][dim];
    signal input token_blindings[batch_size];
    signal input output_blindings[batch_size];
    signal input reference_blindings[batch_size];
    signal input expected_outputs[batch_size][dim];
    signal input weights_s[num_layers][half][2];
    signal input weights_t[num_layers][half][2];
    signal input weights_blinding;
    signal input model_path_elements[model_depth];
    signal input model_path_indices[model_depth];
    signal input input_root;
    signal input output_root;
    signal input reference_root;
    signal input model_root;
    signal input max_error_squared;
    signal input max_abs_error;

    component error_bound_bits = Num2Bits(error_bits);
    error_bound_bits.in <== max_error_squared;

    component abs_bound_bits = Num2Bits(value_bits + 1);
    abs_bound_bits.in <== max_abs_error;

    component weight_range[num_layers][half][4];
    component weight_limit[num_layers][half][4];

    for (var layer = 0; layer < num_layers; layer++) {
        for (var i = 0; i < half; i++) {
            weight_range[layer][i][0] = SignedRange(value_bits);
            weight_range[layer][i][0].in <== weights_s[layer][i][0];

            weight_range[layer][i][1] = SignedRange(value_bits);
            weight_range[layer][i][1].in <== weights_s[layer][i][1];

            weight_range[layer][i][2] = SignedRange(value_bits);
            weight_range[layer][i][2].in <== weights_t[layer][i][0];

            weight_range[layer][i][3] = SignedRange(value_bits);
            weight_range[layer][i][3].in <== weights_t[layer][i][1];

            for (var column = 0; column < 4; column++) {
                weight_limit[layer][i][column] = LessThan(value_bits + 1);
                weight_limit[layer][i][column].in[0] <== weight_range[layer][i][column].abs;
                weight_limit[layer][i][column].in[1] <== WEIGHT_LIMIT() + 1;
                weight_limit[layer][i][column].out === 1;
            }
        }
    }

    component weight_chain = PoseidonChain(weight_count);
    var weight_index = 0;

    for (var layer = 0; layer < num_layers; layer++) {
        for (var i = 0; i < half; i++) {
            weight_chain.in[weight_index] <== weights_s[layer][i][0];
            weight_chain.in[weight_index + 1] <== weights_s[layer][i][1];
            weight_chain.in[weight_index + 2] <== weights_t[layer][i][0];
            weight_chain.in[weight_index + 3] <== weights_t[layer][i][1];
            weight_index = weight_index + 4;
        }
    }

    component model_commit = PoseidonCommit();
    model_commit.value <== weight_chain.out;
    model_commit.blinding <== weights_blinding;

    component model_proof = VerifyMerkleProof(model_depth);
    model_proof.leaf <== model_commit.commitment;

    for (var level = 0; level < model_depth; level++) {
        model_proof.path_elements[level] <== model_path_elements[level];
        model_proof.path_indices[level] <== model_path_indices[level];
    }

    model_proof.root === model_root;

    component inference[batch_size];
    component noise_bounds[batch_size];

    signal input_commitments[batch_size];
    signal output_commitments[batch_size];
    signal reference_commitments[batch_size];

    for (var b = 0; b < batch_size; b++) {
        inference[b] = FullInferenceProof(num_layers, dim, clip_min_magnitude, clip_max);

        for (var i = 0; i < dim; i++) {
            inference[b].tokens[i] <== tokens[b][i];
            inference[b].expected_output[i] <== expected_outputs[b][i];
        }

        for (var layer = 0; layer < num_layers; layer++) {
            for (var i = 0; i < half; i++) {
                inference[b].weights_s[layer][i][0] <== weights_s[layer][i][0];
                inference[b].weights_s[layer][i][1] <== weights_s[layer][i][1];
                inference[b].weights_t[layer][i][0] <== weights_t[layer][i][0];
                inference[b].weights_t[layer][i][1] <== weights_t[layer][i][1];
            }
        }

        inference[b].token_blinding <== token_blindings[b];
        inference[b].output_blinding <== output_blindings[b];
        inference[b].reference_blinding <== reference_blindings[b];
        inference[b].max_error_squared <== max_error_squared;

        input_commitments[b] <== inference[b].input_commitment;
        output_commitments[b] <== inference[b].output_commitment;
        reference_commitments[b] <== inference[b].reference_commitment;

        noise_bounds[b] = VerifyNoiseBound(dim, value_bits + 1);

        for (var i = 0; i < dim; i++) {
            noise_bounds[b].original[i] <== expected_outputs[b][i];
            noise_bounds[b].noisy[i] <== inference[b].y[i];
        }

        noise_bounds[b].max_noise <== max_abs_error;
    }

    component input_batch = VerifyBatchInference(batch_size);
    component output_batch = VerifyBatchInference(batch_size);
    component reference_batch = VerifyBatchInference(batch_size);

    for (var b = 0; b < batch_size; b++) {
        input_batch.leaves[b] <== input_commitments[b];
        output_batch.leaves[b] <== output_commitments[b];
        reference_batch.leaves[b] <== reference_commitments[b];
    }

    input_batch.expected_root <== input_root;
    output_batch.expected_root <== output_root;
    reference_batch.expected_root <== reference_root;
}

component main {public [input_root, output_root, reference_root, model_root, max_error_squared, max_abs_error]} = InferenceTraceWithBatch(2, 2, 16, 5, 5, 4);
