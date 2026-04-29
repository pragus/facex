# FaceX — Fast face embedding library
# 3ms inference, 7MB binary, zero dependencies
#
# Build:  make
# Test:   make example && ./facex-example

CC ?= gcc
CXX ?= g++
AR ?= ar
# LTO jobs: keep 1 by default (warning-free).
# Set LTOFLAGS to override explicitly, e.g.:
#   make LTO_JOBS=4      (parallel LTRANS jobs)
#   make LTOFLAGS=-flto   (toolchain-default LTRANS threading)
LTO_JOBS ?= 1
LTOFLAGS ?= -flto=$(LTO_JOBS)
ARCHFLAGS ?= -march=native
IPAFLAGS ?= -fdevirtualize-at-ltrans -fno-semantic-interposition
CFLAGS = -Ofast -funroll-loops -ftree-vectorize $(ARCHFLAGS) $(LTOFLAGS) $(IPAFLAGS)
CXXFLAGS = $(CFLAGS) -std=c++17 -fno-threadsafe-statics -I. -Ithird_party/highway -DHWY_COMPILE_ALL_ATTAINABLE
LDFLAGS = $(LTOFLAGS) $(IPAFLAGS) -lm -lpthread
OLD_LDFLAGS = -lm -lpthread
EMBED_WEIGHTS ?= docs/demo/edgeface_xs_fp32.bin
DETECT_WEIGHTS ?= weights/yunet_fp32.bin
OLD_CFLAGS = -O3 -march=native -mfma -funroll-loops
PROFILE_GEMM ?= 0

ifeq ($(PROFILE_GEMM),1)
  CXXFLAGS += -DFACEX_ENABLE_GEMM_PROFILE=1
endif

ifeq ($(OS),Windows_NT)
  LDFLAGS += -lsynchronization
  OLD_LDFLAGS += -lsynchronization
  EXT = .exe
endif

SRCS = src/facex.cc src/transformer_ops.cc src/gemm_int8_4x8c8.cc src/threadpool.cc src/detect.cc src/align.cc src/weight_crypto.cc

.PHONY: all clean example lib cli encrypt test bench bench-old bench-detect detect-lib

all: lib cli detect-lib

# Static library
lib: libfacex.a

libfacex.a: $(SRCS) src/edgeface_engine.cc Makefile
	$(CXX) $(CXXFLAGS) -DFACEX_LIB -c src/facex.cc -o facex.o
	$(CXX) $(CXXFLAGS) -c src/transformer_ops.cc -o transformer_ops.o
	$(CXX) $(CXXFLAGS) -c src/gemm_int8_4x8c8.cc -o gemm_int8_4x8c8.o
	$(CXX) $(CXXFLAGS) -c third_party/highway/hwy/targets.cc -o highway_targets.o
	$(CXX) $(CXXFLAGS) -c third_party/highway/hwy/abort.cc -o highway_abort.o
	$(CXX) $(CXXFLAGS) -c src/threadpool.cc -o threadpool.o
	$(CXX) $(CXXFLAGS) -Iinclude -c src/detect.cc -o detect.o
	$(CXX) $(CXXFLAGS) -Iinclude -c src/align.cc -o align.o
	$(CXX) $(CXXFLAGS) -Iinclude -c src/weight_crypto.cc -o weight_crypto.o
	$(AR) rcs $@ facex.o transformer_ops.o gemm_int8_4x8c8.o highway_targets.o highway_abort.o threadpool.o detect.o align.o weight_crypto.o
	@rm -f *.o
	@echo "Built libfacex.a"

# Standalone CLI (for Go subprocess / testing)
cli: facex-cli$(EXT)

facex-cli$(EXT): src/edgeface_engine.cc src/transformer_ops.cc src/gemm_int8_4x8c8.cc src/threadpool.cc src/weight_crypto.cc
	$(CXX) $(CXXFLAGS) -Iinclude -c src/edgeface_engine.cc -o edgeface_engine.o
	$(CXX) $(CXXFLAGS) -Iinclude -c src/transformer_ops.cc -o transformer_ops.o
	$(CXX) $(CXXFLAGS) -Iinclude -c src/gemm_int8_4x8c8.cc -o gemm_int8_4x8c8.o
	$(CXX) $(CXXFLAGS) -Iinclude -c third_party/highway/hwy/targets.cc -o highway_targets.o
	$(CXX) $(CXXFLAGS) -Iinclude -c third_party/highway/hwy/abort.cc -o highway_abort.o
	$(CXX) $(CXXFLAGS) -Iinclude -c src/threadpool.cc -o threadpool.o
	$(CXX) $(CXXFLAGS) -Iinclude -c src/weight_crypto.cc -o weight_crypto.o
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ edgeface_engine.o transformer_ops.o gemm_int8_4x8c8.o highway_targets.o highway_abort.o threadpool.o weight_crypto.o $(LDFLAGS)
	@rm -f edgeface_engine.o transformer_ops.o gemm_int8_4x8c8.o highway_targets.o highway_abort.o threadpool.o weight_crypto.o
	@echo "Built facex-cli$(EXT)"

# Example program
example: facex-example$(EXT)

facex-example$(EXT): examples/example.c libfacex.a
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ $< -L. -lfacex $(LDFLAGS)

# Encryption tool
encrypt: facex-encrypt$(EXT)

facex-encrypt$(EXT): src/weight_crypto.cc
	$(CXX) $(CXXFLAGS) -DWEIGHT_CRYPTO_MAIN -o $@ $< $(LDFLAGS)

# Golden test
test: golden-test$(EXT)
	@echo "Running golden test..."
	@./golden-test$(EXT) $(EMBED_WEIGHTS)

golden-test$(EXT): tests/golden_test.c libfacex.a Makefile
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ $< -L. -lfacex $(LDFLAGS)

# Benchmarks
bench: facex-cli$(EXT)
	@echo "Running embedding benchmark..."
	@./facex-cli$(EXT) $(EMBED_WEIGHTS) > /dev/null

bench-old: .bench-old/facex-cli-old$(EXT)
	@echo "Running old embedding benchmark..."
	@.bench-old/facex-cli-old$(EXT) $(EMBED_WEIGHTS) > /dev/null

.bench-old:
	mkdir -p $@

.bench-old/edgeface_engine.c: | .bench-old
	git show HEAD:src/edgeface_engine.c > $@

.bench-old/transformer_ops.c: | .bench-old
	git show HEAD:src/transformer_ops.c > $@

.bench-old/gemm_int8_4x8c8.c: | .bench-old
	git show HEAD:src/gemm_int8_4x8c8.c > $@

.bench-old/threadpool.c: | .bench-old
	git show HEAD:src/threadpool.c > $@

.bench-old/threadpool.h: | .bench-old
	git show HEAD:src/threadpool.h > $@

.bench-old/weight_crypto.c: | .bench-old
	git show HEAD:src/weight_crypto.c > $@

.bench-old/facex-cli-old$(EXT): .bench-old/edgeface_engine.c .bench-old/transformer_ops.c .bench-old/gemm_int8_4x8c8.c .bench-old/threadpool.c .bench-old/threadpool.h .bench-old/weight_crypto.c Makefile
	$(CC) $(OLD_CFLAGS) -Iinclude -I.bench-old -o $@ .bench-old/edgeface_engine.c .bench-old/transformer_ops.c .bench-old/gemm_int8_4x8c8.c .bench-old/threadpool.c .bench-old/weight_crypto.c $(OLD_LDFLAGS)

bench-detect: facex-bench-detect$(EXT)
	@echo "Running detector benchmark..."
	@if [ -f "$(DETECT_WEIGHTS)" ]; then \
		./facex-bench-detect$(EXT) $(DETECT_WEIGHTS); \
	else \
		echo "Skipping detector benchmark: missing $(DETECT_WEIGHTS)"; \
		echo "Generate it with:"; \
		echo "  python3 -m pip install onnx numpy"; \
		echo "  python3 tools/export_yunet_weights.py weights/yunet_2023mar.onnx $(DETECT_WEIGHTS)"; \
	fi

facex-bench-detect$(EXT): tests/bench_detect.c src/detect.cc include/detect.h Makefile
	$(CXX) $(CXXFLAGS) -Iinclude -o $@ tests/bench_detect.c src/detect.cc $(LDFLAGS)

# Detector static library (Sprint 1+: scaffold only, real engine arrives in
# later sprints — see docs/plan/detector_plan.md).
detect-lib: libdetect.a

libdetect.a: src/detect.cc include/detect.h
	$(CXX) $(CXXFLAGS) -Iinclude -c src/detect.cc -o detect.o
	$(AR) rcs $@ detect.o
	@rm -f detect.o
	@echo "Built libdetect.a"

clean:
	rm -rf .bench-old
	rm -rf .pgo
	rm -f libfacex.a libdetect.a facex-cli$(EXT) facex-example$(EXT) facex-encrypt$(EXT) golden-test$(EXT) facex-bench-detect$(EXT) bench-detect$(EXT) *.o
	rm -f perf.data perf.data.old gmon.out
