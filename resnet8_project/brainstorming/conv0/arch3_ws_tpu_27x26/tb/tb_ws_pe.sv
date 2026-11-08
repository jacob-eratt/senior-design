`timescale 1ns/1ps
// Self-checking testbench for ws_pe. Vectors come from model/gen_pe_vectors.py.
// Stimulus is applied on the falling edge; outputs are compared 1 ns after the
// rising edge that consumed it. Pass +VEC_DIR=<dir> to override tb/vectors.
module tb_ws_pe;
  import ws_pkg::*;

  localparam int MAX_VEC = 1 << 16;

  logic                     clk = 1'b0;
  logic                     rst;
  logic signed [ACT_W-1:0]  act_in, act_out;
  logic                     act_vld_in, act_vld_out;
  logic                     swap_in, swap_out;
  logic                     wld_en;
  logic signed [W_W-1:0]    wld_in, wld_out;
  logic signed [PSUM_W-1:0] psum_in, psum_out;

  logic [63:0] stim_mem  [MAX_VEC];
  logic [63:0] exp_mem   [MAX_VEC];
  logic [31:0] count_mem [1];
  logic [63:0] got;
  string       vec_dir;
  int          n_vec;
  int          errors;

  ws_pe dut (
    .clk                (clk),
    .rst                (rst),
    .pixel_in           (act_in),
    .pixel_valid_in     (act_vld_in),
    .new_pass_start_in  (swap_in),
    .weight_load_enable (wld_en),
    .weight_load_in     (wld_in),
    .partial_sum_in     (psum_in),
    .pixel_out          (act_out),
    .pixel_valid_out    (act_vld_out),
    .new_pass_start_out (swap_out),
    .weight_load_out    (wld_out),
    .partial_sum_out    (psum_out)
  );

  always #2.5 clk = ~clk;

  function automatic logic [63:0] pack_outputs();
    logic [63:0] w;
    w        = '0;
    w[7:0]   = $unsigned(act_out);
    w[8]     = act_vld_out;
    w[9]     = swap_out;
    w[23:16] = $unsigned(wld_out);
    w[63:32] = $unsigned(psum_out);
    return w;
  endfunction

  task automatic apply(input logic [63:0] s);
    act_in     = $signed(s[7:0]);
    act_vld_in = s[8];
    swap_in    = s[9];
    wld_en     = s[10];
    wld_in     = $signed(s[23:16]);
    psum_in    = $signed(s[63:32]);
  endtask

  initial begin
    if (!$value$plusargs("VEC_DIR=%s", vec_dir)) vec_dir = "tb/vectors";
    $readmemh({vec_dir, "/pe_count.mem"}, count_mem);
    $readmemh({vec_dir, "/pe_stim.mem"}, stim_mem);
    $readmemh({vec_dir, "/pe_exp.mem"}, exp_mem);
    n_vec = int'(count_mem[0]);
    if (n_vec <= 0 || n_vec > MAX_VEC)
      $fatal(1, "tb_ws_pe: bad vector count %0d in %s", n_vec, vec_dir);
    errors = 0;
    apply('0);
    rst = 1'b1;
    repeat (3) @(posedge clk);
    #1;
    if (pack_outputs() !== 64'd0) begin
      errors++;
      $display("ERROR reset: outputs %016h, expected all zero", pack_outputs());
    end
    @(negedge clk);
    rst = 1'b0;
    for (int i = 0; i < n_vec; i++) begin
      apply(stim_mem[i]);
      @(posedge clk);
      #1;
      got = pack_outputs();
      if (got !== exp_mem[i]) begin
        errors++;
        if (errors <= 10)
          $display("ERROR cycle %0d: stim %016h got %016h exp %016h (psum got %0d exp %0d)",
                   i, stim_mem[i], got, exp_mem[i],
                   $signed(got[63:32]), $signed(exp_mem[i][63:32]));
      end
      @(negedge clk);
    end
    if (errors == 0)
      $display("PASS tb_ws_pe: %0d cycles compared, 0 mismatches", n_vec);
    else
      $fatal(1, "FAIL tb_ws_pe: %0d mismatches in %0d cycles", errors, n_vec);
    $finish;
  end
endmodule
