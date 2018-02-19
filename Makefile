.PHONY: all clean

all:
	$(MAKE) -C tex all
	$(MAKE) -C src all

test:
	$(MAKE) -C src test

clean:
	$(MAKE) -C tex clean
