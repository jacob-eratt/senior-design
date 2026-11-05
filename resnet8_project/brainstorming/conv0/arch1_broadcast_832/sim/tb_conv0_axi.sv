`timescale 1ns/1ps
module tb_conv0_axi;
    reg clk=0; always #5 clk=~clk;
    reg resetn=0;
    reg [6:0] awaddr=0, araddr=0;
    reg awvalid=0,wvalid=0,bready=0,arvalid=0,rready=0;
    wire awready,wready,bvalid,arready,rvalid;
    reg [31:0] wdata=0; reg [3:0] wstrb=15;
    wire [31:0] rdata; wire [1:0] bresp,rresp;
    reg [255:0] sdata=0; reg [31:0] skeep=0;
    reg slast=0,svalid=0; wire sready;
    wire [255:0] mdata; wire [31:0] mkeep;
    wire mlast,mvalid; reg mready=0;
    conv0_axi_bd dut(.aclk(clk),.aresetn(resetn),
        .s_axi_awaddr(awaddr),.s_axi_awprot(3'd0),.s_axi_awvalid(awvalid),.s_axi_awready(awready),
        .s_axi_wdata(wdata),.s_axi_wstrb(wstrb),.s_axi_wvalid(wvalid),.s_axi_wready(wready),
        .s_axi_bresp(bresp),.s_axi_bvalid(bvalid),.s_axi_bready(bready),
        .s_axi_araddr(araddr),.s_axi_arprot(3'd0),.s_axi_arvalid(arvalid),.s_axi_arready(arready),
        .s_axi_rdata(rdata),.s_axi_rresp(rresp),.s_axi_rvalid(rvalid),.s_axi_rready(rready),
        .s_axis_tdata(sdata),.s_axis_tkeep(skeep),.s_axis_tlast(slast),.s_axis_tvalid(svalid),.s_axis_tready(sready),
        .m_axis_tdata(mdata),.m_axis_tkeep(mkeep),.m_axis_tlast(mlast),.m_axis_tvalid(mvalid),.m_axis_tready(mready));
    reg [7:0] image [0:3071], weights [0:701], expected [0:26623];
    reg [31:0] biases [0:25], quant [0:51];
    reg [31:0] expected_acc [0:26623];
    string fixture;
    integer output_count=0, frame_count=0;
    reg checking=0, stalls=0;
    integer raw_count=0, stall_remaining=0;
    reg long_stall_seen=0;
    reg [31:0] rng=32'h832acafe;
    reg held=0; reg [255:0] held_data; reg held_last;
    always @(negedge clk) begin
        rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
        if (stall_remaining>0) begin mready=0; stall_remaining=stall_remaining-1; end
        else mready=checking && (!stalls || rng[2:0]>2);
    end
    always @(posedge clk) begin
        if (!resetn) held<=0;
        else begin
            // Compare before ReLU/saturation can hide incorrect raw sums.
            if (dut.impl.core_valid && dut.impl.core_ready) begin
                if (!checking || raw_count>=26624) $fatal(1,"Unexpected raw output");
                for (integer lane=0;lane<104;lane=lane+1)
                    if (dut.impl.core_data[lane*32 +: 32]!==expected_acc[raw_count+lane])
                        $fatal(1,"Raw mismatch frame=%0d index=%0d expected=%08x actual=%08x",
                            frame_count,raw_count+lane,expected_acc[raw_count+lane],dut.impl.core_data[lane*32 +: 32]);
                raw_count=raw_count+104;
            end
            if (checking && stall_remaining>0 && !dut.impl.q_ready) long_stall_seen=1;
            if (held && (!mvalid || mdata!==held_data || mlast!==held_last || mkeep!==32'hffffffff))
                $fatal(1,"Output changed while backpressured");
            held<=mvalid && !mready; held_data<=mdata; held_last<=mlast;
            if (mvalid && mready) begin
                if (!checking || output_count>=26624) $fatal(1,"Unexpected output");
                if (mkeep!==32'hffffffff || mlast!==(output_count==26592)) $fatal(1,"KEEP/LAST mismatch");
                for (integer i=0;i<32;i=i+1)
                    if (mdata[i*8 +: 8]!==expected[output_count+i])
                        $fatal(1,"Frame %0d byte %0d expected=%02x actual=%02x",frame_count,output_count+i,expected[output_count+i],mdata[i*8 +: 8]);
                output_count=output_count+32;
            end
        end
    end
    task automatic write_reg(input [6:0] address,input [31:0] value,input bit reverse_order,input [1:0] response);
        fork
            begin
                if (reverse_order) repeat(4) @(negedge clk);
                @(negedge clk); awaddr=address; awvalid=1;
                do @(posedge clk); while (!awready);
                @(negedge clk); awvalid=0;
            end
            begin
                if (!reverse_order) repeat(4) @(negedge clk);
                @(negedge clk); wdata=value; wvalid=1;
                do @(posedge clk); while (!wready);
                @(negedge clk); wvalid=0;
            end
        join
        wait(bvalid);
        repeat(3) begin @(negedge clk); if (!bvalid || bresp!==response) $fatal(1,"Write response not held/correct"); end
        bready=1; @(posedge clk); @(negedge clk); bready=0;
    endtask
    task automatic read_reg(input [6:0] address,output [31:0] value);
        @(negedge clk); araddr=address; arvalid=1;
        do @(posedge clk); while (!arready);
        @(negedge clk); arvalid=0;
        wait(rvalid); value=rdata;
        repeat(2) begin @(negedge clk); if (!rvalid || rdata!==value || rresp!=0) $fatal(1,"Read response unstable"); end
        rready=1; @(posedge clk); @(negedge clk); rready=0;
    endtask
    function automatic [7:0] payload(input integer kind,input integer index);
        case(kind)
            0: payload=image[index];
            1: payload=weights[index];
            2: payload=biases[index/4] >> (8*(index%4));
            3: payload=quant[index/4] >> (8*(index%4));
        endcase
    endfunction
    task automatic load_packet(input integer kind,input integer length,input bit corrupt);
        reg [31:0] status;
        write_reg(7'h0c,kind,kind%2,0);
        for (integer base=0;base<length;base=base+32) begin
            repeat(base%3) @(negedge clk);
            @(negedge clk); sdata=0; skeep=0;
            for (integer i=0;i<32;i=i+1) if (base+i<length) begin
                skeep[i]=1; sdata[i*8 +: 8]=payload(kind,base+i);
            end
            if (corrupt && base==0) sdata[0]=~sdata[0];
            slast=base+32>=length; svalid=1;
            do @(posedge clk); while (!sready);
            @(negedge clk); svalid=0; slast=0;
        end
        do read_reg(4,status); while(status[4]);
        if (!corrupt && (!status[8+kind] || status[3])) $fatal(1,"Load failed kind=%0d status=%x",kind,status);
    endtask
    task automatic run_frame(input bit warm,input bit output_stalls);
        reg [31:0] status, cycles, raw_cycles, first_cycle, output_stall_cycles, loader_chunks;
        write_reg(0,warm?2:4,warm,0);
        load_packet(0,3072,0);
        if (!warm) begin load_packet(1,702,0); load_packet(2,104,0); load_packet(3,208,0); end
        @(negedge clk); output_count=0; raw_count=0; checking=1; stalls=output_stalls;
        stall_remaining=output_stalls ? 400 : 0; long_stall_seen=0; frame_count=frame_count+1;
        write_reg(0,1,1,0);
        do read_reg(4,status); while(!status[1]);
        if (output_count!=26624 || raw_count!=26624 || status[3]) $fatal(1,"Frame incomplete/error");
        if (output_stalls && !long_stall_seen) $fatal(1,"Long stall did not backpressure requantizer");
        read_reg(7'h14,cycles);
        read_reg(7'h18,raw_cycles);
        read_reg(7'h24,first_cycle);
        read_reg(7'h20,output_stall_cycles);
        read_reg(7'h28,loader_chunks);
        $display("PASS frame %0d warm=%0d stalls=%0d bytes=%0d run_cycles=%0d",frame_count,warm,output_stalls,output_count,cycles);
        $display("METRICS frame=%0d raw_cycles=%0d first_output_cycle=%0d output_stalls=%0d loader_chunks=%0d",frame_count,raw_cycles,first_cycle,output_stall_cycles,loader_chunks);
        @(negedge clk); checking=0;
    endtask
    // A single malformed image beat must set the expected error and block START.
    task automatic reject_packet(input [31:0] keep_mask,input bit last_bit,input integer error_bit);
        reg [31:0] err;
        write_reg(0,4,0,0);
        write_reg(7'h0c,0,0,0);
        @(negedge clk); sdata=0; skeep=keep_mask; slast=last_bit; svalid=1;
        do @(posedge clk); while (!sready);
        @(negedge clk); svalid=0; slast=0;
        repeat(8) @(negedge clk);
        read_reg(7'h10,err);
        if (!err[error_bit]) $fatal(1,"Malformed packet not rejected keep=%x last=%b errors=%x",keep_mask,last_bit,err);
        write_reg(0,1,0,2);
        write_reg(0,4,1,0); // also tests recovery from an unfinished packet
        read_reg(4,err);
        if (err[11:0]!=0) $fatal(1,"Reset did not clear readiness/status: %x",err);
    endtask
    reg [31:0] status;
    initial begin
        if (!$value$plusargs("FIXTURE_DIR=%s",fixture)) $fatal(1,"Set +FIXTURE_DIR=<resnet8_project/fixtures>");
        $readmemh({fixture,"/layer_vectors/conv0/input.mem"},image);
        $readmemh({fixture,"/export/conv0_weights.mem"},weights);
        $readmemh({fixture,"/export/conv0_bias.mem"},biases);
        $readmemh({fixture,"/export/conv0_quant.mem"},quant);
        $readmemh({fixture,"/layer_vectors/conv0/expected_output.mem"},expected);
        $readmemh({fixture,"/layer_vectors/conv0/expected_acc.mem"},expected_acc);
        if ($isunknown({image[0],image[3071],weights[0],weights[701],biases[0],biases[25],quant[0],quant[51],expected[0],expected[26623]}))
            $fatal(1,"Missing/incomplete fixture files");
        repeat(8) @(negedge clk); resetn=1;
        for (integer i=0;i<26624;i=i+1)
            if ($isunknown(expected[i]) || $isunknown(expected_acc[i])) $fatal(1,"Incomplete golden index=%0d",i);
        write_reg(0,1,0,2); // reject START before preload
        reject_packet(32'h00000005,1,0); // noncontiguous KEEP
        reject_packet(32'h00000000,1,0); // empty KEEP
        reject_packet(32'h0000ffff,0,0); // partial nonfinal beat / unfinished packet
        reject_packet(32'hffffffff,1,2); // premature TLAST
        load_packet(3,208,1);
        read_reg(7'h10,status); if (!status[1]) $fatal(1,"Corrupt quantization accepted");
        write_reg(0,1,1,2);
        run_frame(0,0);
        run_frame(1,1);
        $display("PASS tb_conv0_axi: 53248 raw INT32 + 53248 INT8 comparisons; cold/warm, AXI channel skew, response/output stalls, FIFO backpressure, malformed packets, reset recovery, bad quantization");
        $finish;
    end
    initial begin repeat(2000000) @(posedge clk); $fatal(1,"Watchdog timeout"); end
endmodule
