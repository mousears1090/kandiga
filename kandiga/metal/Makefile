CC = clang

# CPU-only library: NEON-optimized dequant matvec + GCD parallel dispatch
# No Metal framework needed -- runs entirely on CPU cores
CPU_OBJC_FLAGS = -fobjc-arc -framework Foundation -O2 -march=native
DYLIB = libkandiga_cpu_expert.dylib

.PHONY: all clean

all: $(DYLIB)

$(DYLIB): kandiga_cpu_expert.m kandiga_cpu_expert.h
	$(CC) -shared -o $@ kandiga_cpu_expert.m $(CPU_OBJC_FLAGS)

clean:
	rm -f $(DYLIB)
