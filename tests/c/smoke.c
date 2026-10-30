/* C ABI smoke test: run a kernel through GO and through driven mode, check results. */
#include <stdio.h>
#include <string.h>

#include "mgpu_core.h"

#define CODE 0x1000u
#define OUT  0x8000u

static const uint32_t prog[] = {
    0x1C320001u, /* V_GETSR v1, GTID */
    0x0841100Eu, /* V_SHLI v2, v1, 2 */
    0x08420C00u, /* V_ADD v2, v2, s1 */
    0x4C220000u, /* V_SW v1, 0(v2) */
    0x00000001u, /* EXIT */
};

static int check_output(mgpu_gpu *g, uint32_t n, const char *what) {
    uint32_t out[256];
    if (mgpu_mem_read(g, OUT, (uint8_t *)out, 4 * n) != MGPU_OK) return 1;
    for (uint32_t i = 0; i < n; i++)
        if (out[i] != i) { printf("%s: out[%u] = %u\n", what, i, out[i]); return 1; }
    return 0;
}

static void setup(mgpu_gpu *g, uint32_t grid, uint32_t block) {
    mgpu_mem_write(g, CODE, (const uint8_t *)prog, sizeof prog);
    mgpu_reg_write(g, MGPU_REG_ENTRY, CODE);
    mgpu_reg_write(g, MGPU_REG_GRID_DIM, grid);
    mgpu_reg_write(g, MGPU_REG_BLOCK_DIM, block);
    mgpu_reg_write(g, MGPU_REG_ARG(0), OUT);
}

int main(void) {
    mgpu_config cfg;
    mgpu_config_default(&cfg);
    cfg.mem_size = 1u << 20;
    mgpu_gpu *g = mgpu_create(&cfg);
    if (!g || mgpu_reg_read(g, MGPU_REG_ID) != 0x4D475055u) { puts("create failed"); return 1; }

    /* 1. launch through GO */
    setup(g, 3, 64);
    if (mgpu_reg_write(g, MGPU_REG_GO, 1) != MGPU_OK || mgpu_reg_read(g, MGPU_REG_STATUS) != 0)
        { puts("GO launch failed"); return 1; }
    if (check_output(g, 192, "GO")) return 1;
    if (mgpu_steps(g) != 3 * 2 * 5) { printf("steps %llu\n", (unsigned long long)mgpu_steps(g)); return 1; }

    /* 2. driven mode: always step the last runnable warp */
    uint8_t zero[1024];
    memset(zero, 0, sizeof zero);
    mgpu_mem_write(g, OUT, zero, sizeof zero);
    setup(g, 2, 40);
    if (!mgpu_begin(g)) { puts("begin failed"); return 1; }
    mgpu_record rec;
    uint32_t pairs[64], stores = 0;
    while (mgpu_is_running(g)) {
        uint32_t n = mgpu_runnable(g, pairs, 32);
        if (n == 0 || n > 32) { printf("runnable %u\n", n); return 1; }
        if (mgpu_step(g, pairs[2 * (n - 1)], pairs[2 * (n - 1) + 1], &rec) != MGPU_OK)
            { puts("step failed"); return 1; }
        if (rec.has_fault) { printf("fault %u\n", rec.fault_code); return 1; }
        stores += rec.n_mwrites;
    }
    if (check_output(g, 80, "driven") || stores != 80) { printf("stores %u\n", stores); return 1; }
    if (mgpu_step(g, 0, 0, &rec) != MGPU_ERR_NOT_RUNNABLE) { puts("step after end"); return 1; }

    /* 3. a fault is reported through STATUS */
    setup(g, 1, 32);
    mgpu_reg_write(g, MGPU_REG_ARG(0), 2);     /* misaligned output base */
    mgpu_reg_write(g, MGPU_REG_GO, 1);
    if (mgpu_reg_read(g, MGPU_REG_STATUS) != 2 || mgpu_reg_read(g, MGPU_REG_ERR_PC) != CODE + 12)
        { puts("fault not reported"); return 1; }

    mgpu_destroy(g);
    puts("OK");
    return 0;
}
