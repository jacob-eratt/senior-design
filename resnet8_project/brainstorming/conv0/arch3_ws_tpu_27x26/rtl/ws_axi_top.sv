`timescale 1ns/1ps
// Wiring-only AXI boundary; see docs/05_axi_submodules.md for a guided tour.
// Flat external ports support IP Integrator; internal buses use structs.
// Fixed 27x26 INT8 array, 1..64 vectors, 256-bit AXI streams.
module ws_axi_top (
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:S_AXIS:M_AXIS, ASSOCIATED_RESET aresetn" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    input  logic         aclk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    input  logic         aresetn,
    input  logic [7:0]   s_axi_awaddr,
    input  logic [2:0]   s_axi_awprot,
    input  logic         s_axi_awvalid,
    output logic         s_axi_awready,
    input  logic [31:0]  s_axi_wdata,
    input  logic [3:0]   s_axi_wstrb,
    input  logic         s_axi_wvalid,
    output logic         s_axi_wready,
    output logic [1:0]   s_axi_bresp,
    output logic         s_axi_bvalid,
    input  logic         s_axi_bready,
    input  logic [7:0]   s_axi_araddr,
    input  logic [2:0]   s_axi_arprot,
    input  logic         s_axi_arvalid,
    output logic         s_axi_arready,
    output logic [31:0]  s_axi_rdata,
    output logic [1:0]   s_axi_rresp,
    output logic         s_axi_rvalid,
    input  logic         s_axi_rready,
    input  logic [255:0] s_axis_tdata,
    input  logic [31:0]  s_axis_tkeep,
    input  logic         s_axis_tlast,
    input  logic         s_axis_tvalid,
    output logic         s_axis_tready,
    output logic [255:0] m_axis_tdata,
    output logic [31:0]  m_axis_tkeep,
    output logic         m_axis_tlast,
    output logic         m_axis_tvalid,
    input  logic         m_axis_tready
);
    import ws_axi_pkg::*;
    axil_req_t cpu_req;
    axil_rsp_t cpu_rsp;
    axis_fwd_t rx_stream, tx_stream;
    axis_back_t rx_back, tx_back;
    control_t control;
    loader_status_t loader_status;
    engine_status_t engine_status;
    tile_read_t tile_read;
    tile_data_t tile_data;
    result_read_t result_read;
    logic datapath_reset, send_enable, output_finished;
    logic [31:0] run_vectors, result_word, output_bytes;

    // 1. Bundle the CPU-facing AXI-Lite channels by signal direction.
    assign cpu_req.aw = '{addr:s_axi_awaddr, prot:s_axi_awprot, valid:s_axi_awvalid};
    assign cpu_req.w = '{data:s_axi_wdata, strb:s_axi_wstrb, valid:s_axi_wvalid};
    assign cpu_req.bready = s_axi_bready;
    assign cpu_req.ar = '{addr:s_axi_araddr, prot:s_axi_arprot, valid:s_axi_arvalid};
    assign cpu_req.rready = s_axi_rready;
    assign s_axi_awready = cpu_rsp.awready;
    assign s_axi_wready = cpu_rsp.wready;
    assign s_axi_bresp = cpu_rsp.b.resp;
    assign s_axi_bvalid = cpu_rsp.b.valid;
    assign s_axi_arready = cpu_rsp.arready;
    assign s_axi_rdata = cpu_rsp.r.data;
    assign s_axi_rresp = cpu_rsp.r.resp;
    assign s_axi_rvalid = cpu_rsp.r.valid;

    // 2. Streams have a forward payload/valid bundle and reverse ready bundle.
    assign rx_stream = '{data:s_axis_tdata, keep:s_axis_tkeep,
                         last:s_axis_tlast, valid:s_axis_tvalid};
    assign s_axis_tready = rx_back.ready;
    assign m_axis_tdata = tx_stream.data;
    assign m_axis_tkeep = tx_stream.keep;
    assign m_axis_tlast = tx_stream.last;
    assign m_axis_tvalid = tx_stream.valid;
    assign tx_back.ready = m_axis_tready;
    assign datapath_reset = !aresetn || control.soft_reset;

    // 3. Commands flow out of the register block; status flows back into it.
    ws_axi_regs u_regs (
        .clk(aclk), .aresetn(aresetn), .bus_req(cpu_req), .bus_rsp(cpu_rsp),
        .loader(loader_status), .engine(engine_status),
        .output_bytes(output_bytes), .control(control)
    );
    ws_axis_loader u_loader (
        .clk(aclk), .reset(datapath_reset), .control(control),
        .engine_idle(engine_status.idle), .engine_fault(engine_status.errors != 0),
        .stream_in(rx_stream), .stream_back(rx_back), .status(loader_status),
        .tile_read(tile_read), .tile_data(tile_data)
    );
    ws_tile_engine u_engine (
        .clk(aclk), .reset(datapath_reset), .control(control), .loader(loader_status),
        .status(engine_status), .tile_read(tile_read), .tile_data(tile_data),
        .send_enable(send_enable), .run_vectors(run_vectors),
        .output_finished(output_finished), .result_read(result_read), .result_word(result_word)
    );
    ws_axis_output u_output (
        .clk(aclk), .reset(datapath_reset), .start(control.start),
        .send_enable(send_enable), .run_vectors(run_vectors),
        .result_read(result_read), .result_word(result_word),
        .stream_out(tx_stream), .stream_back(tx_back),
        .finished(output_finished), .output_bytes(output_bytes)
    );
endmodule
