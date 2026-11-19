`timescale 1ns/1ps
// Mitigation 1 (streamed weights): ping-pong weight buffer.
// Two banks of ROWS x COLS weight bytes with the same read interface as c0_weight_store
// (rd_row -> rd_data, combinational). Writes always go to the standby bank and reads
// always come from the active bank, so the next layer's weights can stream in while the
// current layer computes from the active bank.
// Protocol: write ROWS rows with wr_en (wr_last on the final row) while wr_ready is high.
// staged goes high when a complete standby bank is written; it blocks further writes.
// swap (taken only when staged is high) flips active/standby and clears staged. Assert
// swap at a layer boundary, before the controller's WLOAD phase reads rd_row.
// Writes are ignored while staged = 1, so a completed bank cannot be overwritten.
// Memory contents are not reset (REQ-IMP-04); only the control flags are.
// Assumption: wr_row < ROWS and rd_row < ROWS.
module ws_wbuf_pingpong #(
  parameter int ROWS = ws_pkg::ROWS,
  parameter int COLS = ws_pkg::COLS,
  parameter int W_W  = ws_pkg::W_W
) (
  // Inputs
  input  logic                    clk,
  input  logic                    rst,
  input  logic                    wr_en,
  input  logic                    wr_last,
  input  logic [$clog2(ROWS)-1:0] wr_row,
  input  logic signed [W_W-1:0]   wr_data [COLS],
  input  logic                    swap,
  input  logic [$clog2(ROWS)-1:0] rd_row,

  // Outputs
  output logic                    wr_ready,
  output logic                    staged,
  output logic signed [W_W-1:0]   rd_data [COLS]
);

  logic signed [W_W-1:0] bank_mem [2][ROWS][COLS];
  logic                  active;

  assign wr_ready = ~staged;

  for (genvar c = 0; c < COLS; c++) begin : g_rd
    assign rd_data[c] = bank_mem[active][rd_row][c];
  end

  always_ff @(posedge clk) begin
    if (wr_en && !staged) bank_mem[~active][wr_row] <= wr_data;
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      active <= 1'b0;
      staged <= 1'b0;
    end else if (swap && staged) begin
      active <= ~active;
      staged <= 1'b0;
    end else if (wr_en && !staged && wr_last) begin
      staged <= 1'b1;
    end
  end

endmodule
