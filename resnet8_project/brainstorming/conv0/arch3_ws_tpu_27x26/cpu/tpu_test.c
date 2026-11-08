/* Standalone Cortex-A53 bring-up test for the 27x26 weight-stationary TPU tile.
 * Polling AXI DMA (simple mode), no OS. Build in Vitis 2024.2 on the tpu_kv260.xsa platform.
 * Register map: docs/04_axi_wrapper.md. Golden results are computed on the CPU. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "xparameters.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_io.h"
#ifdef SDT
#include "xiltimer.h"
#else
#include "xtime_l.h"
#endif

#define ACC_BASE ((UINTPTR)0xA0000000U)
#define DMA_BASE ((UINTPTR)0xA0010000U)
#define PL_HZ 100000000U
#define ROWS 27
#define COLS 26
#define MAX_VECTORS 64
#define WEIGHT_BYTES (ROWS * COLS)
#define RESULT_BYTES(n) ((unsigned)(n) * COLS * 4U)
#define GUARD 64U
#define TIMEOUT_TICKS ((uint64_t)COUNTS_PER_SECOND)
#define RANDOM_RUNS 200

#define REG_CONTROL 0x00U
#define REG_STATUS 0x04U
#define REG_LOAD_KIND 0x08U
#define REG_VECTOR_COUNT 0x0CU
#define REG_ERROR 0x10U
#define REG_ELAPSED 0x14U
#define REG_COMPUTE 0x18U
#define REG_OUTPUT_BYTES 0x1CU
#define REG_ID 0x20U
#define REG_MAX_VECTORS 0x24U
#define ST_BUSY (1U << 0)
#define ST_WEIGHTS (1U << 1)
#define ST_INPUTS (1U << 2)
#define ST_COMPUTE_DONE (1U << 3)
#define ST_OUTPUT_DONE (1U << 4)
#define ST_ERROR (1U << 5)
#define ST_LOADER_BUSY (1U << 6)

static XAxiDma dma;
static int8_t weights[ROWS][COLS];
static int8_t inputs[MAX_VECTORS][ROWS];
static uint8_t tx[MAX_VECTORS * ROWS] __attribute__((aligned(64)));
static uint8_t rx[RESULT_BYTES(MAX_VECTORS) + 2 * GUARD] __attribute__((aligned(64)));
static uint32_t rng = 0x12345678U;

static uint32_t rd(unsigned off) { return Xil_In32(ACC_BASE + off); }
static void wr(unsigned off, uint32_t value) { Xil_Out32(ACC_BASE + off, value); }
static uint64_t ticks(void) { XTime t; XTime_GetTime(&t); return (uint64_t)t; }
static uint64_t ns(uint64_t dt) {
    return (dt / COUNTS_PER_SECOND) * 1000000000ULL +
           (dt % COUNTS_PER_SECOND) * 1000000000ULL / COUNTS_PER_SECOND;
}
static int8_t rand8(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (int8_t)(rng >> 24);
}
static int fail(const char *where) {
    printf("FAIL,%s,status=%08lx,error=%08lx,mm2s_sr=%08lx,s2mm_sr=%08lx\r\n", where,
           (unsigned long)rd(REG_STATUS), (unsigned long)rd(REG_ERROR),
           (unsigned long)Xil_In32(DMA_BASE + 0x04), (unsigned long)Xil_In32(DMA_BASE + 0x34));
    return -1;
}
static int healthy(void) {
    return !rd(REG_ERROR) && !(Xil_In32(DMA_BASE + 0x04) & 0x770U) &&
           !(Xil_In32(DMA_BASE + 0x34) & 0x770U);
}
/* Wait until all bits in 'set' are 1 and all bits in 'clear' are 0. */
static int wait_status(uint32_t set, uint32_t clear, const char *where) {
    uint64_t begin = ticks();
    for (;;) {
        uint32_t s = rd(REG_STATUS);
        if ((s & set) == set && !(s & clear)) return 0;
        if (!healthy() || ticks() - begin > TIMEOUT_TICKS) return fail(where);
    }
}
static int wait_dma(int direction) {
    uint64_t begin = ticks();
    while (XAxiDma_Busy(&dma, direction))
        if (!healthy() || ticks() - begin > TIMEOUT_TICKS) return fail("DMA wait");
    return healthy() ? 0 : fail("DMA completion");
}
/* Rejected AXI-Lite writes return SLVERR, which aborts the A53: only write when idle. */
static int send(unsigned kind, unsigned bytes) {
    if (wait_status(0, ST_BUSY, "idle before load")) return -1;
    wr(REG_LOAD_KIND, kind);
    Xil_DCacheFlushRange((UINTPTR)tx, bytes);
    if (XAxiDma_SimpleTransfer(&dma, (UINTPTR)tx, bytes, XAXIDMA_DMA_TO_DEVICE) != XST_SUCCESS)
        return fail("MM2S submit");
    if (wait_dma(XAXIDMA_DMA_TO_DEVICE)) return -1;
    return wait_status(kind ? ST_INPUTS : ST_WEIGHTS, ST_LOADER_BUSY, "loader ready");
}
static int load_weights(void) {
    for (unsigned r = 0; r < ROWS; r++)
        for (unsigned c = 0; c < COLS; c++) tx[r * COLS + c] = (uint8_t)weights[r][c];
    return send(0, WEIGHT_BYTES);
}
static int load_inputs(unsigned n) {
    if (wait_status(0, ST_BUSY, "idle before count")) return -1;
    wr(REG_VECTOR_COUNT, n);
    for (unsigned p = 0; p < n; p++)
        for (unsigned r = 0; r < ROWS; r++) tx[p * ROWS + r] = (uint8_t)inputs[p][r];
    return send(1, n * ROWS);
}
static int32_t golden(unsigned p, unsigned c) {
    int32_t acc = 0;
    for (unsigned r = 0; r < ROWS; r++) acc += (int32_t)inputs[p][r] * (int32_t)weights[r][c];
    return acc;
}
/* Inputs and weights must already be loaded. Times START -> results in DDR. */
static int run(const char *name, unsigned n, int print) {
    unsigned bytes = RESULT_BYTES(n);
    memset(rx, 0xa5, sizeof(rx));
    Xil_DCacheFlushRange((UINTPTR)rx, sizeof(rx));
    if (wait_status(ST_WEIGHTS | ST_INPUTS, ST_BUSY | ST_ERROR, "ready before start")) return -1;
    uint64_t begin = ticks();
    if (XAxiDma_SimpleTransfer(&dma, (UINTPTR)(rx + GUARD), bytes, XAXIDMA_DEVICE_TO_DMA) != XST_SUCCESS)
        return fail("S2MM submit");
    wr(REG_CONTROL, 1);
    if (wait_dma(XAXIDMA_DEVICE_TO_DMA)) return -1;
    uint64_t elapsed = ticks() - begin;
    if (wait_status(ST_COMPUTE_DONE | ST_OUTPUT_DONE, ST_BUSY, "output done")) return -1;
    Xil_DCacheInvalidateRange((UINTPTR)rx, sizeof(rx));
    if (rd(REG_OUTPUT_BYTES) != bytes) return fail("output byte count");
    for (unsigned p = 0; p < n; p++)
        for (unsigned c = 0; c < COLS; c++) {
            int32_t got;
            memcpy(&got, rx + GUARD + 4U * (p * COLS + c), 4);
            if (got != golden(p, c)) {
                printf("Mismatch,%s,vector=%u,column=%u,expected=%ld,actual=%ld\r\n", name, p, c,
                       (long)golden(p, c), (long)got);
                return fail("golden comparison");
            }
        }
    for (unsigned i = 0; i < GUARD; i++)
        if (rx[i] != 0xa5 || rx[GUARD + bytes + i] != 0xa5) return fail("DMA guard overwrite");
    if (print)
        printf("RUN,%s,%u,%llu,%lu,%lu\r\n", name, n, (unsigned long long)ns(elapsed),
               (unsigned long)rd(REG_ELAPSED), (unsigned long)rd(REG_COMPUTE));
    return 0;
}
static void pattern_weights(void) {
    for (unsigned r = 0; r < ROWS; r++)
        for (unsigned c = 0; c < COLS; c++) weights[r][c] = (int8_t)(((r * 13 + c * 7) % 255) - 127);
}
static void pattern_inputs(unsigned n) {
    for (unsigned p = 0; p < n; p++)
        for (unsigned r = 0; r < ROWS; r++) inputs[p][r] = (int8_t)(((p * 11 + r * 3) % 255) - 127);
}
static void random_inputs(unsigned n) {
    for (unsigned p = 0; p < n; p++)
        for (unsigned r = 0; r < ROWS; r++) inputs[p][r] = rand8();
}
static int init(void) {
    XAxiDma_Config *cfg;
#ifdef SDT
    cfg = XAxiDma_LookupConfig(DMA_BASE);
#else
    cfg = XAxiDma_LookupConfig(XPAR_AXIDMA_0_DEVICE_ID);
#endif
    if (!cfg || XAxiDma_CfgInitialize(&dma, cfg) != XST_SUCCESS || XAxiDma_HasSg(&dma))
        return fail("DMA configuration");
    XAxiDma_Reset(&dma);
    uint64_t begin = ticks();
    while (!XAxiDma_ResetIsDone(&dma))
        if (ticks() - begin > TIMEOUT_TICKS) return fail("DMA reset");
    XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DMA_TO_DEVICE);
    XAxiDma_IntrDisable(&dma, XAXIDMA_IRQ_ALL_MASK, XAXIDMA_DEVICE_TO_DMA);
    wr(REG_CONTROL, 2);
    if (rd(REG_ID) != 0x57530001U || rd(REG_MAX_VECTORS) != MAX_VECTORS) return fail("hardware identity");
    return wait_status(0, ST_BUSY | ST_ERROR, "idle after reset");
}
int main(void) {
    static const unsigned sizes[] = {1, 64, 3, 2};
    unsigned checked = 0;
    printf("\r\nTPU 27x26 KV260 test: PL_HZ=%u, timer_HZ=%llu\r\n", PL_HZ,
           (unsigned long long)COUNTS_PER_SECOND);
    if (init()) return 1;
    printf("ID_VERSION=%08lx OK\r\n", (unsigned long)rd(REG_ID));
    printf("RUN,name,vectors,total_ns,elapsed_cycles,compute_cycles\r\n");
    pattern_weights();
    if (load_weights()) return 1;
    for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        pattern_inputs(sizes[i]);
        if (load_inputs(sizes[i]) || run("pattern", sizes[i], 1)) return 1;
        checked += sizes[i] * COLS;
    }
    for (unsigned r = 0; r < ROWS; r++)
        for (unsigned c = 0; c < COLS; c++) weights[r][c] = -128;
    for (unsigned r = 0; r < ROWS; r++) { inputs[0][r] = -128; inputs[1][r] = 127; inputs[2][r] = 0; }
    if (load_weights() || load_inputs(3) || run("extremes", 3, 1)) return 1;
    checked += 3 * COLS;
    for (unsigned r = 0; r < ROWS; r++)
        for (unsigned c = 0; c < COLS; c++) weights[r][c] = rand8();
    if (load_weights()) return 1;
    for (unsigned i = 0; i < RANDOM_RUNS; i++) {
        unsigned n = (i % 8 == 7) ? 1U + (rng % MAX_VECTORS) : MAX_VECTORS;
        random_inputs(n);
        if (load_inputs(n) || run("random", n, i < 3 || i == RANDOM_RUNS - 1)) return 1;
        checked += n * COLS;
    }
    pattern_weights();
    pattern_inputs(MAX_VECTORS);
    if (load_weights() || load_inputs(MAX_VECTORS)) return 1;
    for (unsigned i = 0; i < 10; i++) {
        if (run("warm_repeat", MAX_VECTORS, i == 9)) return 1;
        checked += MAX_VECTORS * COLS;
        if (load_inputs(MAX_VECTORS)) return 1;
    }
    printf("PASS: %u INT32 results checked\r\n", checked);
    return 0;
}
