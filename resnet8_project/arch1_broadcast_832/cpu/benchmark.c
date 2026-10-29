/* Standalone Cortex-A53, polling AXI DMA, no OS. Build with Vitis BSP. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "xparameters.h"
#include "xaxidma.h"
#include "xil_cache.h"
#include "xil_io.h"
#include "xtime_l.h"
#include "fixture_data.h"

#define ACC_BASE ((UINTPTR)0xA0000000U)
#define DMA_BASE ((UINTPTR)0xA0010000U)
#define PL_HZ 100000000U
#define OUTPUT_BYTES 26624U
#define TIMEOUT_TICKS ((uint64_t)COUNTS_PER_SECOND * 2U)
#define COLD_FRAMES 20
#define WARM_FRAMES 200
static XAxiDma dma;
static uint8_t rx[OUTPUT_BYTES+128] __attribute__((aligned(64)));
static uint8_t tx[3072] __attribute__((aligned(64)));
static uint32_t rd(unsigned off) { return Xil_In32(ACC_BASE+off); }
static void wr(unsigned off, uint32_t value) { Xil_Out32(ACC_BASE+off,value); }
static uint64_t ticks(void) { XTime t; XTime_GetTime(&t); return (uint64_t)t; }
static uint64_t ns(uint64_t dt) {
    return (dt/COUNTS_PER_SECOND)*1000000000ULL + (dt%COUNTS_PER_SECOND)*1000000000ULL/COUNTS_PER_SECOND;
}
static int fail(const char *where) {
    printf("FAIL,%s,status=%08lx,error=%08lx,mm2s=%08lx,s2mm=%08lx\n",where,
        (unsigned long)rd(4),(unsigned long)rd(0x10),
        (unsigned long)Xil_In32(DMA_BASE+4),(unsigned long)Xil_In32(DMA_BASE+0x34));
    return -1;
}
static int healthy(void) {
    return !rd(0x10) && !(Xil_In32(DMA_BASE+4)&0x770U) && !(Xil_In32(DMA_BASE+0x34)&0x770U);
}
static int wait_dma(int direction) {
    uint64_t begin=ticks();
    while (XAxiDma_Busy(&dma,direction)) {
        if (!healthy() || ticks()-begin>TIMEOUT_TICKS) return fail("DMA wait");
    }
    return healthy()?0:fail("DMA completion");
}
static int load(unsigned kind, const uint8_t *source, unsigned bytes) {
    memcpy(tx,source,bytes);
    Xil_DCacheFlushRange((UINTPTR)tx,bytes);
    wr(0x0c,kind);
    if (XAxiDma_SimpleTransfer(&dma,(UINTPTR)tx,bytes,XAXIDMA_DMA_TO_DEVICE)!=XST_SUCCESS) return fail("MM2S submit");
    if (wait_dma(XAXIDMA_DMA_TO_DEVICE)) return -1;
    uint64_t begin=ticks();
    while (!(rd(4)&(1U<<(8+kind))) || (rd(4)&16)) {
        if (!healthy() || ticks()-begin>TIMEOUT_TICKS) return fail("loader wait");
    }
    return 0;
}
static int frame(int cold, unsigned index) {
    uint64_t begin=ticks();
    wr(0,cold?4:2); /* cold invalidates all memories; warm retains parameters */
    if (load(0,fixture_image,sizeof(fixture_image))) return -1;
    if (cold && (load(1,fixture_weights,sizeof(fixture_weights)) ||
                 load(2,fixture_bias,sizeof(fixture_bias)) || load(3,fixture_quant,sizeof(fixture_quant)))) return -1;
    memset(rx,0xa5,sizeof(rx));
    Xil_DCacheFlushRange((UINTPTR)rx,sizeof(rx));
    /* Arm receive before START: otherwise the output FIFO eventually stalls. */
    if (XAxiDma_SimpleTransfer(&dma,(UINTPTR)(rx+64),OUTPUT_BYTES,XAXIDMA_DEVICE_TO_DMA)!=XST_SUCCESS) return fail("S2MM submit");
    wr(0,1);
    if (wait_dma(XAXIDMA_DEVICE_TO_DMA)) return -1;
    uint64_t wait_begin=ticks();
    while (!(rd(4)&2)) if (!healthy() || ticks()-wait_begin>TIMEOUT_TICKS) return fail("core done wait");
    Xil_DCacheInvalidateRange((UINTPTR)rx,sizeof(rx));
    uint64_t elapsed=ticks()-begin;
    if (rd(8)!=OUTPUT_BYTES) return fail("output byte count");
    /* Checking and UART printing are deliberately outside timed transaction. */
    for (unsigned i=0;i<OUTPUT_BYTES;i++) if (rx[64+i]!=fixture_expected[i]) {
        printf("Mismatch,index=%u,expected=%u,actual=%u\n",i,fixture_expected[i],rx[64+i]);
        return fail("golden comparison");
    }
    for (unsigned i=0;i<64;i++) if (rx[i]!=0xa5 || rx[64+OUTPUT_BYTES+i]!=0xa5) return fail("DMA guard overwrite");
    printf("FRAME,%s,%u,%llu,%lu,%lu,%lu,%lu,%lu,%lu\n",cold?"cold":"warm",index,
        (unsigned long long)ns(elapsed),(unsigned long)rd(0x14),(unsigned long)rd(0x18),
        (unsigned long)rd(0x24),(unsigned long)rd(0x1c),(unsigned long)rd(0x20),(unsigned long)rd(0x28));
    return 0;
}
int main(void) {
    printf("Conv0 KV260 benchmark: PL_HZ=%u, counter_HZ=%llu\n",PL_HZ,(unsigned long long)COUNTS_PER_SECOND);
    XAxiDma_Config *cfg;
#ifdef SDT
    cfg=XAxiDma_LookupConfig(DMA_BASE);
#else
    cfg=XAxiDma_LookupConfig(XPAR_AXIDMA_0_DEVICE_ID);
#endif
    if (!cfg || XAxiDma_CfgInitialize(&dma,cfg)!=XST_SUCCESS || XAxiDma_HasSg(&dma)) return fail("DMA configuration");
    XAxiDma_IntrDisable(&dma,XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DMA_TO_DEVICE);
    XAxiDma_IntrDisable(&dma,XAXIDMA_IRQ_ALL_MASK,XAXIDMA_DEVICE_TO_DMA);
    if (rd(0x2c)!=0x00010000 || rd(0x30)!=832 || rd(0x34)!=104) return fail("hardware identity");
    printf("kind,mode,index,total_ns,run_cycles,raw_cycles,first_output_cycle,input_stalls,output_stalls,load_cycles\n");
    for (unsigned i=0;i<COLD_FRAMES;i++) if (frame(1,i)) return 1;
    for (unsigned i=0;i<WARM_FRAMES;i++) if (frame(0,i)) return 1;
    printf("PASS: %u frames, %u INT8 comparisons per frame\n",COLD_FRAMES+WARM_FRAMES,OUTPUT_BYTES);
    return 0;
}
