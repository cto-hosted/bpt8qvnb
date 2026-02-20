CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude
LDFLAGS  :=

SRC_DIR  := src
INC_DIR  := include
BUILD_DIR := build

SRCS := $(SRC_DIR)/dsp_engine.cpp \
        $(SRC_DIR)/loop_extractor.cpp \
        $(SRC_DIR)/similarity_detector.cpp \
        $(SRC_DIR)/crossfade_processor.cpp \
        $(SRC_DIR)/main.cpp

OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))
TARGET := $(BUILD_DIR)/dsp_engine

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)

test: $(TARGET)
	@echo "DSP engine built successfully at $(TARGET)"
	@$(TARGET) --help 2>&1 | head -3 || true
