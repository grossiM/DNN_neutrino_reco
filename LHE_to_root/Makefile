PWD 		:= $(shell pwd)
ROOTCFLAGS      := $(shell $(ROOTSYS)/bin/root-config --cflags)
ROOTGLIBS       := $(shell $(ROOTSYS)/bin/root-config --glibs)
ROOTLDFL       := $(shell $(ROOTSYS)/bin/root-config --ldflags)

write_lhe2root: write_lhe2root.cc
	g++ -o write_lhe2root -I$(LHAPDF_PREFIX)/include $(ROOTCFLAGS) -L$(PWD) $(ROOTGLIBS) write_lhe2root.cc

