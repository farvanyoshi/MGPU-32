# MGPU-32 RTL: lint and file lists. Simulation is driven from pytest (tests/rtl).
VERILATOR ?= ../tools/verilator.sh

PKG    := pkg/mgpu_isa_pkg.sv pkg/mgpu_fp_pkg.sv
COMMON := common/mgpu_pipe.sv
UNITS  := units/mgpu_round.sv units/mgpu_int_lane.sv units/mgpu_valu.sv \
          units/mgpu_fpu_lane.sv units/mgpu_fpu.sv \
          units/mgpu_sfu_rom.sv units/mgpu_sfu_lane.sv units/mgpu_sfu.sv
TOPS   := mgpu_pipe mgpu_valu mgpu_fpu mgpu_sfu

SOURCES = $(wildcard $(PKG)) $(COMMON) $(wildcard $(UNITS))

.PHONY: lint files
lint:
	@for top in $(TOPS); do \
	  if [ $$top = mgpu_pipe ] || ls units/$$top.sv >/dev/null 2>&1; then \
	    echo "lint $$top"; \
	    $(VERILATOR) --lint-only -Wall --top-module $$top $(SOURCES) || exit 1; \
	  fi; \
	done

files:
	@echo $(SOURCES)
