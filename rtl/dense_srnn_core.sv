`timescale 1ns/1ps

module dense_srnn_core #(
    parameter integer N_INPUTS = 8,
    parameter integer N_NEURONS = 16,
    parameter integer WEIGHT_BITS = 8,
    parameter integer STATE_BITS = 16,
    parameter integer FRAC_BITS = 8,
    parameter integer SYNAPTIC_DELAY = 0,
    parameter integer OUTPUT_DELAY = 0,
    parameter integer INTEGRATION_MODE = 0,
    parameter integer THRESHOLD_TIMING = 0,
    parameter integer RESET_MODE = 0,
    parameter integer SATURATE = 1,
    parameter integer TAU_STEPS = 2
) (
    input  logic clk,
    input  logic rst,
    input  logic valid_in,
    input  logic [N_INPUTS-1:0] input_spikes,
    input  logic signed [N_INPUTS*N_NEURONS*WEIGHT_BITS-1:0] input_weights,
    input  logic signed [N_NEURONS*N_NEURONS*WEIGHT_BITS-1:0] recurrent_weights,
    input  logic signed [N_NEURONS*STATE_BITS-1:0] thresholds,
    input  logic signed [N_NEURONS*STATE_BITS-1:0] reset_values,
    output logic valid_out,
    output logic [N_NEURONS-1:0] output_spikes
);
    localparam integer ACC_BITS = STATE_BITS + $clog2(N_INPUTS + N_NEURONS + 1) + WEIGHT_BITS;
    logic [N_NEURONS-1:0] recurrent_spikes;
    logic [N_NEURONS-1:0] raw_spikes;
    logic [N_NEURONS-1:0] neuron_valid;
    logic signed [STATE_BITS-1:0] raw_current [0:N_NEURONS-1];
    logic signed [STATE_BITS-1:0] delayed_current [0:N_NEURONS-1];
    logic delayed_valid [0:N_NEURONS-1];
    integer neuron;
    integer source;
    logic signed [ACC_BITS-1:0] accumulator;
    logic signed [WEIGHT_BITS-1:0] selected_weight;

    always_comb begin
        for (neuron = 0; neuron < N_NEURONS; neuron = neuron + 1) begin
            accumulator = '0;
            for (source = 0; source < N_INPUTS; source = source + 1) begin
                selected_weight = input_weights[
                    (neuron*N_INPUTS + source)*WEIGHT_BITS +: WEIGHT_BITS
                ];
                if (input_spikes[source]) accumulator = accumulator + selected_weight;
            end
            for (source = 0; source < N_NEURONS; source = source + 1) begin
                selected_weight = recurrent_weights[
                    (neuron*N_NEURONS + source)*WEIGHT_BITS +: WEIGHT_BITS
                ];
                if (recurrent_spikes[source]) accumulator = accumulator + selected_weight;
            end
            // The generated deployment wrapper must prove the accumulator range.
            // This cast is exact only when that range fits STATE_BITS.
            raw_current[neuron] = accumulator[STATE_BITS-1:0];
        end
    end

    genvar generated_neuron;
    generate
        for (generated_neuron = 0; generated_neuron < N_NEURONS; generated_neuron = generated_neuron + 1) begin : g_neurons
            synaptic_delay_line #(
                .WIDTH(STATE_BITS), .DELAY_STEPS(SYNAPTIC_DELAY)
            ) current_delay (
                .clk(clk), .rst(rst), .valid_in(valid_in),
                .value_in(raw_current[generated_neuron]),
                .valid_out(delayed_valid[generated_neuron]),
                .value_out(delayed_current[generated_neuron])
            );
            lif_neuron #(
                .STATE_BITS(STATE_BITS), .FRAC_BITS(FRAC_BITS),
                .INTEGRATION_MODE(INTEGRATION_MODE), .TAU_STEPS(TAU_STEPS),
                .THRESHOLD_TIMING(THRESHOLD_TIMING), .RESET_MODE(RESET_MODE),
                .SATURATE(SATURATE)
            ) neuron_instance (
                .clk(clk), .rst(rst), .valid_in(delayed_valid[generated_neuron]),
                .synaptic_current(delayed_current[generated_neuron]),
                .threshold(thresholds[generated_neuron*STATE_BITS +: STATE_BITS]),
                .reset_value(reset_values[generated_neuron*STATE_BITS +: STATE_BITS]),
                .valid_out(neuron_valid[generated_neuron]),
                .spike(raw_spikes[generated_neuron]), .membrane()
            );
        end
    endgenerate

    assign recurrent_spikes = raw_spikes;
    spike_delay_line #(.WIDTH(N_NEURONS), .DELAY_STEPS(OUTPUT_DELAY)) output_delay (
        .clk(clk), .rst(rst), .valid_in(neuron_valid[0]), .spikes_in(raw_spikes),
        .valid_out(valid_out), .spikes_out(output_spikes)
    );
endmodule

