FORTRAN := ifort
FFLAGS := -I$(MKLROOT)/include -fPIC -O3
# LDFLAGS :=  -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm
LDFLAGS :=  -L$(MKLROOT)/lib/intel64  -lmkl_rt -lpthread -lm

libhint.so: libhint.pyf system.o todeint.o hint.o
	f2py -c $(LDFLAGS) --fcompiler=intelem $^
libhint.pyf: hint.f90
	f2py --overwrite-signature -m libhint -h libhint.pyf hint.f90
%.o: %.f90
	$(FORTRAN) $(FFLAGS) -c $^

clean:
	rm -vf *.o *.mod *.pyf *.so
