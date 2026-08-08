`timescale 1ns/1ps

module spike_delay_line #(
    parameter integer WIDTH = 8,
    parameter integer DELAY_STEPS = 0
) (
    input  logic clk,
    input  logic rst,
    input  logic valid_in,
    input  logic [WIDTH-1:0] spikes_in,
    output logic valid_out,
    output logic [WIDTH-1:0] spikes_out
);
    generate
        if (DELAY_STEPS == 0) begin : g_no_delay
            always_comb begin
                valid_out = valid_in;
                spikes_out = spikes_in;
            end
        end else begin : g_delay
            logic [WIDTH-1:0] values [0:DELAY_STEPS-1];
            logic valids [0:DELAY_STEPS-1];
            integer index;
            always_ff @(posedge clk) begin
                if (rst) begin
                    for (index = 0; index < DELAY_STEPS; index = index + 1) begin
                        values[index] <= '0;
                        valids[index] <= 1'b0;
                    end
                end else begin
                    values[0] <= spikes_in;
                    valids[0] <= valid_in;
                    for (index = 1; index < DELAY_STEPS; index = index + 1) begin
                        values[index] <= values[index-1];
                        valids[index] <= valids[index-1];
                    end
                end
            end
            always_comb begin
                spikes_out = values[DELAY_STEPS-1];
                valid_out = valids[DELAY_STEPS-1];
            end
        end
    endgenerate
endmodule

