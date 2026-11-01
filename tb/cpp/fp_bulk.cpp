// Bulk bit-exact test of the Verilated FPU and SFU against the Rust reference
// (libmgpu_core, mgpu_fp_op). Streams one pass per cycle into each unit and checks every
// lane of every result.
//
//   fp_bulk random N [seed]     N random operand sets per FPU op and per SFU op
//   fp_bulk sfu-sweep [stride]  every stride-th mantissa of every SFU path-selecting binade
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "Vmgpu_fp_bulk_top.h"
#include "mgpu_core.h"
#include "ops.h"
#include "verilated.h"

namespace {

struct Pending {
  uint32_t fp_op, a[8], b[8], c[8];
  int lanes;
};

uint64_t rng_state = 0x9E3779B97F4A7C15ull;
uint32_t rnd() {
  rng_state ^= rng_state >> 12;
  rng_state ^= rng_state << 25;
  rng_state ^= rng_state >> 27;
  return uint32_t((rng_state * 0x2545F4914F6CDD1Dull) >> 32);
}

const uint32_t SPECIALS[] = {
    0x00000000, 0x80000000, 0x00000001, 0x807FFFFF, 0x00400000, 0x00800000, 0x80800000,
    0x00800001, 0x7F7FFFFF, 0xFF7FFFFF, 0x7F7FFFFE, 0x7F800000, 0xFF800000, 0x7FC00000,
    0xFFC00000, 0x7F800001, 0x7FBFFFFF, 0x3F800000, 0xBF800000, 0x3F800001, 0x3F7FFFFF,
    0x40000000, 0x3F000000, 0x4B000000, 0x4B7FFFFF, 0x4F000000, 0xCF000000, 0x4EFFFFFF,
    0xCF000001, 0x34000000, 0x33800000, 0x0C800000, 0x1F800000};
const int NSPECIAL = sizeof(SPECIALS) / sizeof(SPECIALS[0]);

uint32_t random_f32() {  // the mix of mgpu.fp.testvals.random_f32
  uint32_t r = rnd() % 100, s = rnd() & 1u;
  if (r < 15) return SPECIALS[rnd() % NSPECIAL];
  if (r < 30) return (s << 31) | ((rnd() % 4) << 23) | (rnd() & 0x7FFFFF);
  if (r < 40) return (s << 31) | ((0xFC + rnd() % 3) << 23) | (rnd() & 0x7FFFFF);
  if (r < 75) return (s << 31) | ((100 + rnd() % 55) << 23) | (rnd() & 0x7FFFFF);
  return rnd();
}

struct Harness {
  Vmgpu_fp_bulk_top* top;
  std::deque<Pending> fpu_q, sfu_q;
  uint64_t checked = 0, failures = 0;

  explicit Harness(VerilatedContext* ctx) : top(new Vmgpu_fp_bulk_top(ctx)) {
    top->rst_n = 0;
    top->fpu_valid = 0;
    top->sfu_valid = 0;
    for (int i = 0; i < 3; i++) tick();
    top->rst_n = 1;
  }
  ~Harness() { top->final(); delete top; }

  void tick() {
    top->clk = 0;
    top->eval();
    top->clk = 1;
    top->eval();
    if (top->rst_n) check();
  }

  void check() {
    if (top->fpu_out_valid) collect(fpu_q, 8, [&](int i) { return uint32_t(top->fpu_result[i]); });
    if (top->sfu_out_valid) collect(sfu_q, 2, [&](int i) { return uint32_t(top->sfu_result >> (32 * i)); });
  }

  template <class F> void collect(std::deque<Pending>& q, int width, F get) {
    if (q.empty()) { std::printf("unexpected output\n"); std::exit(2); }
    Pending p = q.front();
    q.pop_front();
    for (int i = 0; i < p.lanes && i < width; i++) {
      uint32_t want = mgpu_fp_op(p.fp_op, p.a[i], p.b[i], p.c[i]), got = get(i);
      checked++;
      if (got != want && ++failures <= 10)
        std::printf("MISMATCH op %u (%08x, %08x, %08x): rtl %08x, reference %08x\n", p.fp_op,
                    p.a[i], p.b[i], p.c[i], got, want);
    }
  }

  // Issue one pass to each unit (either may be idle).
  void issue(const Pending* f, uint32_t fpu_ope, const Pending* s, uint32_t sfu_ope) {
    top->fpu_valid = f != nullptr;
    if (f) {
      top->fpu_op = fpu_ope;
      for (int i = 0; i < 8; i++) {
        top->fpu_a[i] = f->a[i];
        top->fpu_b[i] = f->b[i];
        top->fpu_c[i] = f->c[i];
      }
      fpu_q.push_back(*f);
    }
    top->sfu_valid = s != nullptr;
    if (s) {
      top->sfu_op = sfu_ope;
      top->sfu_a = uint64_t(s->a[0]) | (uint64_t(s->a[1]) << 32);
      sfu_q.push_back(*s);
    }
    tick();
  }

  void drain() {
    top->fpu_valid = 0;
    top->sfu_valid = 0;
    for (int i = 0; i < 16; i++) tick();
    if (!fpu_q.empty() || !sfu_q.empty()) { std::printf("outputs missing\n"); std::exit(2); }
  }
};

const uint32_t FPU_OPE[] = {OPE_V_FADD, OPE_V_FSUB, OPE_V_FMUL, OPE_V_FMIN, OPE_V_FMAX, OPE_V_FFMA, OPE_V_I2F, OPE_V_F2I};
const uint32_t FPU_FP[] = {MGPU_FP_FADD, MGPU_FP_FSUB, MGPU_FP_FMUL, MGPU_FP_FMIN, MGPU_FP_FMAX, MGPU_FP_FFMA, MGPU_FP_I2F, MGPU_FP_F2I};
const uint32_t SFU_OPE[] = {OPE_V_FRCP, OPE_V_FRSQ, OPE_V_FEX2, OPE_V_FLG2};
const uint32_t SFU_FP[] = {MGPU_FP_FRCP, MGPU_FP_FRSQ, MGPU_FP_FEX2, MGPU_FP_FLG2};

void random_mode(Harness& h, uint64_t n) {
  for (int op = 0; op < 8; op++) {
    uint64_t passes = (n + 7) / 8;
    for (uint64_t k = 0; k < passes; k++) {
      Pending f{FPU_FP[op], {}, {}, {}, 8};
      for (int i = 0; i < 8; i++) {
        uint32_t a = random_f32(), b = random_f32(), c = random_f32();
        if (rnd() % 4 == 0) b = (a ^ 0x80000000u) + (rnd() % 7) - 3;      // cancellation
        if (op == 5 && rnd() % 3 == 0) c = mgpu_fp_op(MGPU_FP_FMUL, a, b, 0) ^ 0x80000000u ^ (rnd() % 4);
        if (op == 6) a = rnd();
        f.a[i] = a; f.b[i] = b; f.c[i] = c;
      }
      Pending s{SFU_FP[op % 4], {}, {}, {}, 2};
      s.a[0] = random_f32();
      s.a[1] = random_f32();
      bool sfu_on = k < (n + 1) / 2;
      h.issue(&f, FPU_OPE[op], sfu_on ? &s : nullptr, SFU_OPE[op % 4]);
    }
  }
}

void sfu_sweep(Harness& h, uint32_t stride) {
  struct Range { int op; int exps[10]; int nexp; bool both_signs; };
  const Range ranges[] = {
      {0, {0, 1, -1}, 3, true},                       // FRCP: the polynomial depends on the mantissa only
      {1, {0, 1}, 2, false},                          // FRSQ: even and odd exponents
      {2, {-24, -10, -1, 0, 1, 3, 6}, 7, true},       // FEX2: every integer/fraction split
      {3, {-2, -1, 0, 1, 7, -126, 126}, 7, false},    // FLG2: near-one paths and the far path
  };
  for (const Range& r : ranges)
    for (int e = 0; e < r.nexp; e++)
      for (uint32_t sign = 0; sign <= (r.both_signs ? 1u : 0u); sign++)
        for (uint32_t m = 0; m < (1u << 23); m += 2 * stride) {
          Pending s{SFU_FP[r.op], {}, {}, {}, 2};
          for (int i = 0; i < 2; i++)
            s.a[i] = (sign << 31) | (uint32_t(r.exps[e] + 127) << 23) | ((m + i * stride) & 0x7FFFFF);
          h.issue(nullptr, 0, &s, SFU_OPE[r.op]);
        }
}

}  // namespace

int main(int argc, char** argv) {
  VerilatedContext ctx;
  ctx.commandArgs(argc, argv);
  std::string mode = argc > 1 ? argv[1] : "random";
  Harness h(&ctx);
  if (mode == "random") {
    uint64_t n = argc > 2 ? std::strtoull(argv[2], nullptr, 0) : 100000;
    if (argc > 3) rng_state ^= std::strtoull(argv[3], nullptr, 0) * 0x9E3779B97F4A7C15ull;
    random_mode(h, n);
  } else if (mode == "sfu-sweep") {
    uint32_t stride = argc > 2 ? uint32_t(std::strtoul(argv[2], nullptr, 0)) : 1;
    sfu_sweep(h, stride ? stride : 1);
  } else {
    std::printf("usage: fp_bulk random N [seed] | sfu-sweep [stride]\n");
    return 2;
  }
  h.drain();
  std::printf("%s: %llu lane results checked, %llu mismatches\n", mode.c_str(),
              (unsigned long long)h.checked, (unsigned long long)h.failures);
  return h.failures ? 1 : 0;
}
