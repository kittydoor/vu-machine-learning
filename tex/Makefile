.PHONY: all clean

PUBLISH_DIR := ../publish
BUILD_DIR := ../.build
TEX_DIR := .
#SRC_DIR := ../src
TEX_FILES := $(wildcard $(TEX_DIR)/sects/*/*.tex)
CC := xelatex
FLAGS := -shell-escape -output-directory=$(BUILD_DIR)

all: $(PUBLISH_DIR)/doc.pdf

$(PUBLISH_DIR)/doc.pdf: $(TEX_DIR)/doc.tex $(TEX_FILES)
	mkdir -p $(BUILD_DIR)
	$(CC) $(FLAGS) $<
	$(CC) $(FLAGS) $<
	mkdir -p $(PUBLISH_DIR)
	mv $(BUILD_DIR)/doc.pdf $(PUBLISH_DIR)/

clean:
	rm -rf $(PUBLISH_DIR) $(BUILD_DIR)
