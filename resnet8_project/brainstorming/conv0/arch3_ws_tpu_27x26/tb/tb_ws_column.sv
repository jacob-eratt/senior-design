`timescale 1ns/1ps
// Self-checking testbench for ws_column. Vectors come from
// model/gen_column_vectors.py --rows ROWS (already skewed per row). Stimulus is
// applied on the falling edge; outputs are compared 1 ns after the rising edge
// that consumed it. Pass +VEC_DIR=<dir> to override tb/vectors.
module tb_ws_column #(
  parameter int ROWS = ws_pkg::ROWS
);
  import ws_pkg::*;

  localparam int WORD_W  = 10 * ROWS + 41;
  localparam int MAX_VEC = 1 << 15;
  localparam int VLD_LO  = 8 * ROWS;
  localparam int SWP_LO  = 9 * ROWS;
  localparam int TAIL_LO = 10 * ROWS;

  logic                     clk = 1'b0;
  logic                     rst;
  logic signed [ACT_W-1:0]  pixel_in           [ROWS];
  logic                     pixel_valid_in     [ROWS];
  logic                     new_pass_start_in  [ROWS];
  logic                     weight_load_enable;
  logic signed [W_W-1:0]    weight_load_in;
  logic signed [PSUM_W-1:0] partial_sum_in;
  logic signed [ACT_W-1:0]  pixel_out          [ROWS];
  logic                     pixel_valid_out    [ROWS];
  logic                     new_pass_start_out [ROWS];
  logic signed [W_W-1:0]    weight_load_out;
  logic signed [PSUM_W-1:0] partial_sum_out;
  logic                     partial_sum_valid_out;

  logic [WORD_W-1:0] stim_mem  [MAX_VEC];
  logic [WORD_W-1:0] exp_mem   [MAX_VEC];
  logic [31:0]       count_mem [2];
  logic [WORD_W-1:0] got;
  string             vec_dir;
  string             prefix;
  int                n_vec;
  int                n_dot;
  int                dots_seen;
  int                errors;

  ws_column #(.ROWS(ROWS)) dut (
    .clk                   (clk),
    .rst                   (rst),
    .pixel_in              (pixel_in),
    .pixel_valid_in        (pixel_valid_in),
    .new_pass_start_in     (new_pass_start_in),
    .weight_load_enable    (weight_load_enable),
    .weight_load_in        (weight_load_in),
    .partial_sum_in        (partial_sum_in),
    .pixel_out             (pixel_out),
    .pixel_valid_out       (pixel_valid_out),
    .new_pass_start_out    (new_pass_start_out),
    .weight_load_out       (weight_load_out),
    .partial_sum_out       (partial_sum_out),
    .partial_sum_valid_out (partial_sum_valid_out)
  );

  always #2.5 clk = ~clk;

  function automatic logic [WORD_W-1:0] pack_outputs();
    logic [WORD_W-1:0] w;
    w = '0;
    for (int r = 0; r < ROWS; r++) begin
      w[8*r +: 8]    = $unsigned(pixel_out[r]);
      w[VLD_LO + r]  = pixel_valid_out[r];
      w[SWP_LO + r]  = new_pass_start_out[r];
    end
    w[TAIL_LO +: 8]      = $unsigned(weight_load_out);
    w[TAIL_LO + 8 +: 32] = $unsigned(partial_sum_out);
    w[TAIL_LO + 40]      = partial_sum_valid_out;
    return w;
  endfunction

  task automatic apply(input logic [WORD_W-1:0] s);
    for (int r = 0; r < ROWS; r++) begin
      pixel_in[r]          = $signed(s[8*r +: 8]);
      pixel_valid_in[r]    = s[VLD_LO + r];
      new_pass_start_in[r] = s[SWP_LO + r];
    end
    weight_load_enable = s[TAIL_LO];
    weight_load_in     = $signed(s[TAIL_LO + 1 +: 8]);
    partial_sum_in     = $signed(s[TAIL_LO + 9 +: 32]);
  endtask

  initial begin
    if (!$value$plusargs("VEC_DIR=%s", vec_dir)) vec_dir = "tb/vectors";
    prefix = $sformatf("%s/column_r%0d", vec_dir, ROWS);
    $readmemh({prefix, "_count.mem"}, count_mem);
    $readmemh({prefix, "_stim.mem"}, stim_mem);
    $readmemh({prefix, "_exp.mem"}, exp_mem);
    n_vec = int'(count_mem[0]);
    n_dot = int'(count_mem[1]);
    if (n_vec <= 0 || n_vec > MAX_VEC)
      $fatal(1, "tb_ws_column: bad vector count %0d in %s (generate with --rows %0d)",
             n_vec, prefix, ROWS);
    errors    = 0;
    dots_seen = 0;
    apply('0);
    rst = 1'b1;
    repeat (3) @(posedge clk);
    #1;
    if (pack_outputs() !== '0) begin
      errors++;
      $display("ERROR reset: outputs not all zero");
    end
    @(negedge clk);
    rst = 1'b0;
    for (int i = 0; i < n_vec; i++) begin
      apply(stim_mem[i]);
      @(posedge clk);
      #1;
      got = pack_outputs();
      if (partial_sum_valid_out === 1'b1) dots_seen++;
      if (got !== exp_mem[i]) begin
        errors++;
        if (errors <= 10)
          $display("ERROR cycle %0d: partial_sum_out got %0d exp %0d, valid got %b exp %b%s",
                   i, partial_sum_out, $signed(exp_mem[i][TAIL_LO + 8 +: 32]),
                   partial_sum_valid_out, exp_mem[i][TAIL_LO + 40],
                   (got[TAIL_LO + 8 +: 33] === exp_mem[i][TAIL_LO + 8 +: 33]) ?
                   " (mismatch is in the pass-through/weight-chain fields)" : "");
      end
      @(negedge clk);
    end
    if (dots_seen != n_dot) begin
      errors++;
      $display("ERROR saw %0d valid column outputs, expected %0d", dots_seen, n_dot);
    end
    if (errors == 0)
      $display("PASS tb_ws_column: ROWS=%0d, %0d cycles compared, %0d dot products, 0 mismatches",
               ROWS, n_vec, dots_seen);
    else
      $fatal(1, "FAIL tb_ws_column: ROWS=%0d, %0d mismatches in %0d cycles", ROWS, errors, n_vec);
    $finish;
  end
endmodule
