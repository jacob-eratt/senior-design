`timescale 1ns/1ps
// Self-checking testbench for ws_mxu (input skews + ws_array). Vectors come
// from model/gen_mxu_vectors.py --rows ROWS --cols COLS. Stimulus is applied on
// the falling edge; every output is compared 1 ns after the rising edge that
// consumed it. Pass +VEC_DIR=<dir> to override tb/vectors.
module tb_ws_mxu #(
  parameter int ROWS = ws_pkg::ROWS,
  parameter int COLS = ws_pkg::COLS
);
  import ws_pkg::*;

  localparam int STIM_W  = 8 * ROWS + 3 + 8 * COLS;
  localparam int EXP_W   = 33 * COLS;
  localparam int CTL_LO  = 8 * ROWS;
  localparam int WLD_LO  = 8 * ROWS + 3;
  localparam int VLD_LO  = 32 * COLS;
  localparam int MAX_VEC = 1 << 14;

  logic                     clk = 1'b0;
  logic                     rst;
  logic signed [ACT_W-1:0]  pixel_in              [ROWS];
  logic                     pixel_valid_in;
  logic                     new_pass_start_in;
  logic                     weight_load_enable;
  logic signed [W_W-1:0]    weight_load_in        [COLS];
  logic signed [PSUM_W-1:0] partial_sum_out       [COLS];
  logic                     partial_sum_valid_out [COLS];

  logic [STIM_W-1:0] stim_mem  [MAX_VEC];
  logic [EXP_W-1:0]  exp_mem   [MAX_VEC];
  logic [31:0]       count_mem [2];
  logic [EXP_W-1:0]  got;
  string             vec_dir;
  string             prefix;
  int                n_vec;
  int                n_out;
  int                outs_seen;
  int                errors;

  ws_mxu #(.ROWS(ROWS), .COLS(COLS)) dut (
    .clk                   (clk),
    .rst                   (rst),
    .pixel_in              (pixel_in),
    .pixel_valid_in        (pixel_valid_in),
    .new_pass_start_in     (new_pass_start_in),
    .weight_load_enable    (weight_load_enable),
    .weight_load_in        (weight_load_in),
    .partial_sum_out       (partial_sum_out),
    .partial_sum_valid_out (partial_sum_valid_out)
  );

  always #2.5 clk = ~clk;

  function automatic logic [EXP_W-1:0] pack_outputs();
    logic [EXP_W-1:0] w;
    w = '0;
    for (int c = 0; c < COLS; c++) begin
      w[32*c +: 32]  = $unsigned(partial_sum_out[c]);
      w[VLD_LO + c]  = partial_sum_valid_out[c];
    end
    return w;
  endfunction

  task automatic apply(input logic [STIM_W-1:0] s);
    for (int r = 0; r < ROWS; r++) pixel_in[r] = $signed(s[8*r +: 8]);
    pixel_valid_in     = s[CTL_LO];
    new_pass_start_in  = s[CTL_LO + 1];
    weight_load_enable = s[CTL_LO + 2];
    for (int c = 0; c < COLS; c++) weight_load_in[c] = $signed(s[WLD_LO + 8*c +: 8]);
  endtask

  initial begin
    if (!$value$plusargs("VEC_DIR=%s", vec_dir)) vec_dir = "tb/vectors";
    prefix = $sformatf("%s/mxu_r%0dc%0d", vec_dir, ROWS, COLS);
    $readmemh({prefix, "_count.mem"}, count_mem);
    $readmemh({prefix, "_stim.mem"}, stim_mem);
    $readmemh({prefix, "_exp.mem"}, exp_mem);
    n_vec = int'(count_mem[0]);
    n_out = int'(count_mem[1]);
    if (n_vec <= 0 || n_vec > MAX_VEC)
      $fatal(1, "tb_ws_mxu: bad vector count %0d in %s (generate with --rows %0d --cols %0d)",
             n_vec, prefix, ROWS, COLS);
    errors    = 0;
    outs_seen = 0;
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
      for (int c = 0; c < COLS; c++) begin
        if (partial_sum_valid_out[c] === 1'b1) outs_seen++;
        if (got[32*c +: 32] !== exp_mem[i][32*c +: 32] ||
            got[VLD_LO + c] !== exp_mem[i][VLD_LO + c]) begin
          errors++;
          if (errors <= 10)
            $display("ERROR cycle %0d col %0d: got %0d (valid %b) exp %0d (valid %b)",
                     i, c, partial_sum_out[c], partial_sum_valid_out[c],
                     $signed(exp_mem[i][32*c +: 32]), exp_mem[i][VLD_LO + c]);
        end
      end
      @(negedge clk);
    end
    if (outs_seen != n_out) begin
      errors++;
      $display("ERROR saw %0d valid outputs, expected %0d", outs_seen, n_out);
    end
    if (errors == 0)
      $display("PASS tb_ws_mxu: %0dx%0d, %0d cycles compared, %0d outputs (= X*W), 0 mismatches",
               ROWS, COLS, n_vec, outs_seen);
    else
      $fatal(1, "FAIL tb_ws_mxu: %0dx%0d, %0d mismatches in %0d cycles", ROWS, COLS, errors, n_vec);
    $finish;
  end
endmodule
