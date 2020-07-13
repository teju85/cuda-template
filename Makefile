
### Project specific variables
GPU_ARCHS     ?= 70
CUDA_HOME     ?= /usr/local/cuda
### Project specific variables

### Project specific constants
STD_CXX       := c++11
SRC_DIR       := src
EXE           := cuda-template
TEST_DIR      := tests
TEST_EXE      := test-$(EXE)
CATCH2_DIR    := catch2
LIBS          :=
INCLUDES      := -I$(CATCH2_DIR)/single_include \
                 -I$(CUDA_HOME)/include \
                 -I$(SRC_DIR)
### Project specific constants

######## Don't edit anything below this!
NVCC          := nvcc
GENCODE       := $(foreach arch,$(GPU_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
NVCCFLAGS     := $(GENCODE) \
                 -std=$(STD_CXX) \
                 --expt-extended-lambda \
                 $(INCLUDES)
CXX           := g++
CXXFLAGS      := -std=$(STD_CXX) \
                 $(INCLUDES)
LD            := nvcc
LDFLAGS       := $(LIBS)
CU_SRCS       := $(shell find $(SRC_DIR) -name "*.cu")
CXX_SRCS      := $(shell find $(SRC_DIR) -name "*.cpp")
CU_OBJS       := $(patsubst %.cu,%.cu.o,$(CU_SRCS))
CXX_OBJS      := $(patsubst %.cpp,%.cpp.o,$(CXX_SRCS))
OBJS          := $(CU_OBJS) $(CXX_OBJS)
TEST_CU_SRCS  := $(shell find $(TEST_DIR) -name "*.cu")
TEST_CXX_SRCS := $(shell find $(TEST_DIR) -name "*.cpp")
TEST_CU_OBJS  := $(patsubst %.cu,%.cu.o,$(TEST_CU_SRCS))
TEST_CXX_OBJS := $(patsubst %.cpp,%.cpp.o,$(TEST_CXX_SRCS))
TEST_OBJS     := $(TEST_CU_OBJS) $(TEST_CXX_OBJS)

default:
	@echo "make what? Available targets are:"
	@echo "  . clean     - clean up built files"
	@echo "  . clean_all - clean up built files and other downloaded files"
	@echo "  . exe       - build the executable"
	@echo "  . test      - build and run the test exe"
	@echo "Flags to customize behavior:"
	@echo "  . GPU_ARCHS - space-separated list of gpu-architectures to"
	@echo "                compile for [$(GPU_ARCHS)]"

debug:
	@echo "TEST_OBJS=$(TEST_OBJS)"
	@echo "TEST_CU_OBJS=$(TEST_CU_OBJS)"
	@echo "TEST_CXX_OBJS=$(TEST_CXX_OBJS)"
	@echo "TEST_CU_SRCS=$(TEST_CU_SRCS)"
	@echo "TEST_CXX_SRCS=$(TEST_CXX_SRCS)"

.PHONY: exe
exe: $(EXE)

$(EXE): $(OBJS)
	$(LD) $(LDLFAGS) -o $@ $^

.PHONY: test
test: $(CATCH2_DIR) $(TEST_EXE)
	./$(TEST_EXE)

$(TEST_EXE): $(TEST_OBJS) $(OBJS)
	$(LD) $(LDFLAGS) -o $@ $^

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(CATCH2_DIR):
	git clone https://github.com/catchorg/Catch2 $@

.PHONY: clean
clean:
	rm -f $(EXE) $(OBJS) $(TEST_OBJS)

.PHONY: clean_all
clean_all: clean
	rm -rf $(CATCH2_DIR)
