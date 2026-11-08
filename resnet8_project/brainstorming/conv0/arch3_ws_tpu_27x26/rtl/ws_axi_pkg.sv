`timescale 1ns/1ps
// Internal interface types. Structs group wires; they do not add storage.
// Top-level ports stay flat so IP Integrator can recognize standard AXI buses.
package ws_axi_pkg;
    localparam int ROWS = 27, COLS = 26, MAX_VECTORS = 64;
    localparam int STREAM_BYTES = 32, WORDS_PER_BEAT = 8;

    typedef struct packed {
        logic [7:0] addr;
        logic [2:0] prot;
        logic valid;
    } axil_addr_t;
    typedef struct packed {
        logic [31:0] data;
        logic [3:0] strb;
        logic valid;
    } axil_write_t;
    typedef struct packed {
        logic [1:0] resp;
        logic valid;
    } axil_b_t;
    typedef struct packed {
        logic [31:0] data;
        logic [1:0] resp;
        logic valid;
    } axil_r_t;
    typedef struct packed {
        axil_addr_t aw;
        axil_write_t w;
        logic bready;
        axil_addr_t ar;
        logic rready;
    } axil_req_t; // CPU/master -> register block.
    typedef struct packed {
        logic awready, wready;
        axil_b_t b;
        logic arready;
        axil_r_t r;
    } axil_rsp_t; // Register block -> CPU/master.

    typedef struct packed {
        logic [255:0] data;
        logic [31:0] keep;
        logic last, valid;
    } axis_fwd_t; // Stream producer -> consumer.
    typedef struct packed {
        logic ready;
    } axis_back_t; // Consumer -> producer (opposite direction).

    typedef struct packed {
        logic [31:0] load_kind, vector_count;
        logic start, soft_reset, write_pending;
    } control_t;
    typedef struct packed {
        logic weights_ready, inputs_ready;
        logic [31:0] loaded_vectors;
        logic busy;
        logic packet_start, packet_error; // Events on this clock's handshake.
        logic [1:0] errors;               // Sticky KEEP/LAST errors.
    } loader_status_t;
    typedef struct packed {
        logic idle, compute_done, output_done;
        logic [31:0] errors, elapsed_cycles, compute_cycles;
    } engine_status_t;
    typedef struct packed {
        logic weight_en;
        logic [31:0] weight_addr;
        logic input_en;
        logic [31:0] input_addr;
    } tile_read_t;
    typedef struct packed {
        logic [COLS*8-1:0] weights;
        logic [ROWS*8-1:0] inputs;
    } tile_data_t;
    typedef struct packed {
        logic enable;
        logic [$clog2(MAX_VECTORS)-1:0] vector;
        logic [$clog2(COLS)-1:0] column;
    } result_read_t; // Separate fields: no divide/modulo on the read path.
endpackage
