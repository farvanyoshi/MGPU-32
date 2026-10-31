# Bulk FPU/SFU harness: Verilated RTL linked against the Rust reference (libmgpu_core).
#   make -C tb/cpp bulk N=10000000          random vectors per op
#   make -C tb/cpp sweep STRIDE=1           every mantissa of every SFU path-selecting binade
ROOT      := $(abspath ../..)
OBJ       := $(ROOT)/build/cpp
VERILATOR ?= $(ROOT)/tools/verilator.sh
LIB       := $(ROOT)/rust/target/release/libmgpu_core.a
N         ?= 100000
STRIDE    ?= 64
SEED      ?= 1
RTL       := $(ROOT)/rtl
SOURCES   := $(RTL)/pkg/mgpu_isa_pkg.sv $(RTL)/pkg/mgpu_fp_pkg.sv $(RTL)/common/mgpu_pipe.sv \
             $(RTL)/units/mgpu_round.sv $(RTL)/units/mgpu_fpu_lane.sv $(RTL)/units/mgpu_fpu.sv \
             $(RTL)/units/mgpu_sfu_rom.sv $(RTL)/units/mgpu_sfu_lane.sv $(RTL)/units/mgpu_sfu.sv \
             $(ROOT)/tb/rtl/mgpu_fp_bulk_top.sv
# The verilator wheel leaves CFG_CXXFLAGS_PCH_I empty (see docs/rtl/README.md); GCC's
# flag is harmless for an installed Verilator too.
VMAKEFLAGS := --MAKEFLAGS CFG_CXXFLAGS_PCH_I=-include

.PHONY: build bulk sweep
build: $(OBJ)/fp_bulk

$(LIB):
	cargo build --release --manifest-path $(ROOT)/rust/Cargo.toml -p mgpu-core

$(OBJ)/ops.h: gen_ops_h.py $(ROOT)/spec/isa.toml
	mkdir -p $(OBJ)
	python3 gen_ops_h.py $@

$(OBJ)/fp_bulk: fp_bulk.cpp $(SOURCES) $(LIB) $(OBJ)/ops.h
	$(VERILATOR) --cc --exe --build -O3 -Wall -Wno-fatal $(VMAKEFLAGS) --top-module mgpu_fp_bulk_top \
	  -Mdir $(OBJ)/obj -o $(OBJ)/fp_bulk -CFLAGS "-O2 -I$(ROOT)/include -I$(OBJ)" \
	  -LDFLAGS "$(LIB) -lpthread -ldl -lm" $(SOURCES) fp_bulk.cpp

bulk: build
	$(OBJ)/fp_bulk random $(N) $(SEED)

sweep: build
	$(OBJ)/fp_bulk sfu-sweep $(STRIDE)
