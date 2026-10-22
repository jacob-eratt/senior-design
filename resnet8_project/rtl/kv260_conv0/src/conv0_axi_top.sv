`timescale 1ns/1ps
module conv0_axi_top(
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:S_AXIS:M_AXIS, ASSOCIATED_RESET aresetn, FREQ_HZ 100000000" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *) input wire aclk,
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *) input wire aresetn,
    (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 7, FREQ_HZ 100000000" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    input wire [6:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input wire [2:0] s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input wire  s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire  s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input wire [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input wire [3:0] s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input wire  s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire  s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output reg [1:0] s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output reg  s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input wire  s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input wire [6:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *)
    input wire [2:0] s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input wire  s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output wire  s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output reg [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output reg [1:0] s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output reg  s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input wire  s_axi_rready,
    (* X_INTERFACE_PARAMETER = "TDATA_NUM_BYTES 32, HAS_TKEEP 1, HAS_TLAST 1, HAS_TREADY 1, FREQ_HZ 100000000" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TDATA" *)
    input wire [255:0] s_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *)
    input wire [31:0] s_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TLAST" *)
    input wire  s_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TVALID" *)
    input wire  s_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS TREADY" *)
    output wire  s_axis_tready,
    (* X_INTERFACE_PARAMETER = "TDATA_NUM_BYTES 32, HAS_TKEEP 1, HAS_TLAST 1, HAS_TREADY 1, FREQ_HZ 100000000" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output wire [255:0] m_axis_tdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *)
    output wire [31:0] m_axis_tkeep,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output wire  m_axis_tlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output wire  m_axis_tvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input wire  m_axis_tready);
    import conv0_quant_pkg::*;
    reg aw_pending, w_pending;
    reg [6:0] awaddr;
    reg [31:0] wdata;
    reg [3:0] wstrb;
    reg [1:0] load_kind;
    reg start_pulse, clear_pulse;
    reg running, done_sticky, raw_sticky;
    reg [3:0] loaded;
    reg [31:0] errors, run_cycles, raw_cycles, input_stalls, output_stalls;
    reg [31:0] out_bytes, first_output_cycle, load_cycles;
    reg first_output_seen;
    wire reset_datapath=!aresetn || clear_pulse;

    // Four AXIS beat slots, with independent 16-byte/clock consumption.
    reg [255:0] rx_data [0:3];
    reg [31:0] rx_keep [0:3];
    reg rx_last [0:3];
    reg [1:0] rx_kind [0:3];
    reg [1:0] rx_wr, rx_rd;
    reg [2:0] rx_count;
    reg upper;
    reg [12:0] packet_bytes;
    wire [15:0] chunk_keep=upper ? rx_keep[rx_rd][31:16] : rx_keep[rx_rd][15:0];
    wire [127:0] chunk_data=upper ? rx_data[rx_rd][255:128] : rx_data[rx_rd][127:0];
    wire chunk_valid=rx_count!=0 && !running && !start_pulse && !clear_pulse;
    wire beat_end=upper || rx_keep[rx_rd][31:16]==0;
    wire packet_end=beat_end && rx_last[rx_rd];
    wire [1:0] chunk_kind=rx_kind[rx_rd];
    integer chunk_bytes, expected_bytes;
    reg bad_quant, bad_keep;
    always @* begin
        chunk_bytes=0;
        for (integer i=0; i<16; i=i+1) chunk_bytes=chunk_bytes+chunk_keep[i];
        case (chunk_kind)
            0: expected_bytes=3072;
            1: expected_bytes=702;
            2: expected_bytes=104;
            default: expected_bytes=208;
        endcase
        bad_keep=(rx_keep[rx_rd]==0) || ((rx_keep[rx_rd] & (rx_keep[rx_rd]+32'd1))!=0)
            || (!rx_last[rx_rd] && rx_keep[rx_rd]!=32'hffffffff);
        bad_quant=0;
        for (integer i=0; i<16; i=i+1)
            if (chunk_kind==3 && chunk_keep[i] && chunk_data[i*8 +: 8]!=quant_byte(packet_bytes+i)) bad_quant=1;
    end
    assign s_axis_tready=aresetn && !running && !start_pulse && !clear_pulse && rx_count<4;
    wire rx_push=s_axis_tvalid && s_axis_tready;
    wire rx_pop=chunk_valid && beat_end;
    wire loader_busy=rx_count!=0 || packet_bytes!=0 || s_axis_tvalid;
    wire load_to_core=chunk_valid && !bad_keep && !bad_quant && packet_bytes+chunk_bytes<=expected_bytes;
    always @(posedge aclk) begin
        if (reset_datapath) begin rx_wr<=0; rx_rd<=0; rx_count<=0; upper<=0; packet_bytes<=0; end
        else begin
            if (rx_push) begin
                rx_data[rx_wr]<=s_axis_tdata; rx_keep[rx_wr]<=s_axis_tkeep;
                rx_last[rx_wr]<=s_axis_tlast; rx_kind[rx_wr]<=load_kind; rx_wr<=rx_wr+1'b1;
            end
            if (chunk_valid) begin
                if (beat_end) begin rx_rd<=rx_rd+1'b1; upper<=0; end else upper<=1;
                if (packet_end) packet_bytes<=0; else packet_bytes<=packet_bytes+chunk_bytes;
            end
            case ({rx_push,rx_pop})
                2'b10: rx_count<=rx_count+1'b1;
                2'b01: rx_count<=rx_count-1'b1;
                default: rx_count<=rx_count;
            endcase
        end
    end
    wire core_valid, core_ready, core_done, core_busy;
    wire [3327:0] core_data;
    wire q_valid, q_ready, fifo_valid;
    wire [831:0] q_data;
    conv0_832_core core(aclk,reset_datapath,start_pulse,load_to_core,chunk_kind,
        packet_bytes[11:0],chunk_data,chunk_keep,core_valid,core_ready,core_data,core_done,core_busy);
    conv0_requant quant(aclk,reset_datapath,core_valid,core_ready,core_data,q_valid,q_ready,q_data);
    conv0_result_fifo fifo(aclk,reset_datapath,q_valid,q_ready,q_data,
        fifo_valid,m_axis_tready && running,m_axis_tdata);
    assign m_axis_tvalid=fifo_valid && running;
    assign m_axis_tkeep=32'hffffffff;
    assign m_axis_tlast=out_bytes==26592;
    wire tx_fire=m_axis_tvalid && m_axis_tready;

    assign s_axi_awready=aresetn && !aw_pending && !s_axi_bvalid;
    assign s_axi_wready=aresetn && !w_pending && !s_axi_bvalid;
    assign s_axi_arready=aresetn && !s_axi_rvalid;
    always @(posedge aclk) begin
        if (!aresetn) begin
            aw_pending<=0; w_pending<=0; awaddr<=0; wdata<=0; wstrb<=0;
            s_axi_bvalid<=0; s_axi_bresp<=0; s_axi_rvalid<=0; s_axi_rresp<=0; s_axi_rdata<=0;
            load_kind<=0; start_pulse<=0; clear_pulse<=0; running<=0;
            done_sticky<=0; raw_sticky<=0; loaded<=0; errors<=0;
            run_cycles<=0; raw_cycles<=0; input_stalls<=0; output_stalls<=0;
            out_bytes<=0; first_output_cycle<=0; first_output_seen<=0; load_cycles<=0;
        end else begin
            start_pulse<=0; clear_pulse<=0;
            if (s_axi_awvalid && s_axi_awready) begin aw_pending<=1; awaddr<=s_axi_awaddr; end
            if (s_axi_wvalid && s_axi_wready) begin w_pending<=1; wdata<=s_axi_wdata; wstrb<=s_axi_wstrb; end
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid<=0;
            if (s_axi_rvalid && s_axi_rready) s_axi_rvalid<=0;
            if (s_axis_tvalid && !s_axis_tready) input_stalls<=input_stalls+1;
            if (chunk_valid) begin
                load_cycles<=load_cycles+1;
                if (packet_bytes==0) loaded[chunk_kind]<=0;
                if (bad_keep) errors[0]<=1;
                if (bad_quant) errors[1]<=1;
                if (packet_bytes+chunk_bytes>expected_bytes || (packet_end && packet_bytes+chunk_bytes!=expected_bytes)) errors[2]<=1;
                if (packet_end && load_to_core && packet_bytes+chunk_bytes==expected_bytes && errors==0) loaded[chunk_kind]<=1;
            end
            if (running) run_cycles<=run_cycles+1;
            if (core_done) begin raw_sticky<=1; raw_cycles<=run_cycles; end
            if (m_axis_tvalid && !m_axis_tready) output_stalls<=output_stalls+1;
            if (tx_fire) begin
                out_bytes<=out_bytes+32;
                if (!first_output_seen) begin first_output_seen<=1; first_output_cycle<=run_cycles; end
                if (m_axis_tlast) begin running<=0; done_sticky<=1; end
            end
            if (aw_pending && w_pending && !s_axi_bvalid) begin
                aw_pending<=0; w_pending<=0; s_axi_bvalid<=1; s_axi_bresp<=0;
                if (awaddr==7'h00 && wstrb[0]) begin
                    if (wdata[7:0]==1 && !running && !loader_busy && !clear_pulse && loaded==15 && errors==0 && !done_sticky) begin
                        start_pulse<=1; running<=1; run_cycles<=0;
                    end else if (wdata[7:0]==4 || (wdata[7:0]==2 && !running && !loader_busy)) begin
                        clear_pulse<=1; running<=0; done_sticky<=0; raw_sticky<=0; errors<=0;
                        loaded<=wdata[2] ? 4'd0 : (loaded & 4'b1110);
                        run_cycles<=0; raw_cycles<=0; input_stalls<=0; output_stalls<=0;
                        out_bytes<=0; first_output_cycle<=0; first_output_seen<=0; load_cycles<=0;
                    end else s_axi_bresp<=2'b10;
                end else if (awaddr==7'h0c && wstrb[0] && !running && !loader_busy && wdata[7:0]<4) load_kind<=wdata[1:0];
                else s_axi_bresp<=2'b10;
            end
            if (s_axi_arvalid && s_axi_arready) begin
                s_axi_rvalid<=1; s_axi_rresp<=0;
                case (s_axi_araddr)
                    7'h00: s_axi_rdata<=0;
                    7'h04: s_axi_rdata<={20'd0,loaded,3'd0,loader_busy,(errors!=0),raw_sticky,done_sticky,running};
                    7'h08: s_axi_rdata<=out_bytes;
                    7'h0c: s_axi_rdata<={30'd0,load_kind};
                    7'h10: s_axi_rdata<=errors;
                    7'h14: s_axi_rdata<=run_cycles;
                    7'h18: s_axi_rdata<=raw_cycles;
                    7'h1c: s_axi_rdata<=input_stalls;
                    7'h20: s_axi_rdata<=output_stalls;
                    7'h24: s_axi_rdata<=first_output_cycle;
                    7'h28: s_axi_rdata<=load_cycles;
                    7'h2c: s_axi_rdata<=32'h00010000;
                    7'h30: s_axi_rdata<=832;
                    7'h34: s_axi_rdata<=104;
                    default: begin s_axi_rdata<=0; s_axi_rresp<=2'b10; end
                endcase
            end
        end
    end
endmodule
