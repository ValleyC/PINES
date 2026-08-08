`timescale 1ns/1ps

module lif_neuron #(
    parameter integer STATE_BITS = 16,
    parameter integer FRAC_BITS = 8,
    // 0: forward Euler, 1: exponential Euler
    parameter integer INTEGRATION_MODE = 0,
    parameter integer TAU_STEPS = 2,
    parameter signed [STATE_BITS-1:0] ALPHA_Q = 16'sd155,
    // 0: post-integration, 1: pre-integration
    parameter integer THRESHOLD_TIMING = 0,
    // 0: subtract threshold, 1: set reset_value
    parameter integer RESET_MODE = 0,
    // 0: two's-complement wrap, 1: signed saturation
    parameter integer SATURATE = 1
) (
    input  logic clk,
    input  logic rst,
    input  logic valid_in,
    input  logic signed [STATE_BITS-1:0] synaptic_current,
    input  logic signed [STATE_BITS-1:0] threshold,
    input  logic signed [STATE_BITS-1:0] reset_value,
    output logic valid_out,
    output logic spike,
    output logic signed [STATE_BITS-1:0] membrane
);
    localparam integer WIDE_BITS = 2 * STATE_BITS + 2;
    localparam signed [STATE_BITS:0] ONE_Q = (1 <<< FRAC_BITS);

    logic signed [WIDE_BITS-1:0] integrated_wide;
    logic signed [WIDE_BITS-1:0] reset_wide;
    logic signed [STATE_BITS-1:0] integrated;
    logic signed [STATE_BITS-1:0] membrane_next;
    logic spike_next;

    function automatic signed [STATE_BITS-1:0] narrow(
        input signed [WIDE_BITS-1:0] value
    );
        logic signed [WIDE_BITS-1:0] maximum;
        logic signed [WIDE_BITS-1:0] minimum;
        begin
            maximum = (1 <<< (STATE_BITS-1)) - 1;
            minimum = -(1 <<< (STATE_BITS-1));
            if (SATURATE && value > maximum)
                narrow = maximum[STATE_BITS-1:0];
            else if (SATURATE && value < minimum)
                narrow = minimum[STATE_BITS-1:0];
            else
                narrow = value[STATE_BITS-1:0];
        end
    endfunction

    function automatic signed [WIDE_BITS-1:0] integrate(
        input signed [STATE_BITS-1:0] voltage,
        input signed [STATE_BITS-1:0] current
    );
        logic signed [WIDE_BITS-1:0] difference;
        logic signed [WIDE_BITS-1:0] alpha_product;
        logic signed [WIDE_BITS-1:0] current_product;
        begin
            if (INTEGRATION_MODE == 0) begin
                difference = current - voltage;
                integrate = voltage + difference / TAU_STEPS;
            end else begin
                alpha_product = ALPHA_Q * voltage;
                current_product = (ONE_Q - ALPHA_Q) * current;
                integrate = (alpha_product + current_product) >>> FRAC_BITS;
            end
        end
    endfunction

    function automatic signed [WIDE_BITS-1:0] apply_reset(
        input signed [STATE_BITS-1:0] voltage
    );
        begin
            if (RESET_MODE == 0)
                apply_reset = voltage - threshold;
            else
                apply_reset = reset_value;
        end
    endfunction

    always_comb begin
        spike_next = 1'b0;
        membrane_next = membrane;
        if (THRESHOLD_TIMING == 1) begin
            spike_next = (membrane >= threshold);
            if (spike_next)
                reset_wide = apply_reset(membrane);
            else
                reset_wide = membrane;
            integrated_wide = integrate(narrow(reset_wide), synaptic_current);
            membrane_next = narrow(integrated_wide);
        end else begin
            integrated_wide = integrate(membrane, synaptic_current);
            integrated = narrow(integrated_wide);
            spike_next = (integrated >= threshold);
            if (spike_next)
                membrane_next = narrow(apply_reset(integrated));
            else
                membrane_next = integrated;
            reset_wide = '0;
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            membrane <= '0;
            spike <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            valid_out <= valid_in;
            if (valid_in) begin
                membrane <= membrane_next;
                spike <= spike_next;
            end else begin
                spike <= 1'b0;
            end
        end
    end

    initial begin
        if (TAU_STEPS <= 0) $fatal(1, "TAU_STEPS must be positive");
        if (FRAC_BITS < 0 || FRAC_BITS >= STATE_BITS)
            $fatal(1, "FRAC_BITS outside state format");
    end
endmodule

