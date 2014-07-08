FORTRAN := ifort
INCMKL := -I$(MKLROOT)/include -fPIC -O3
# LDFLAGS :=  -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm
LDMKL :=  -L$(MKLROOT)/lib/intel64  -lmkl_rt -lpthread -lm


default: hconstruct.so libhint.so timedepsparse.so tools.so
# default: libhint.so

main: system.o todeint.o hint.o main.o
	$(FORTRAN) -o main $^ $(LDMKL)

libhint.so: libhint.pyf system.o todeint.o hint.o
	f2py -c $(LDMKL) --fcompiler=intelem $^
libhint.pyf: hint.f90
	f2py --overwrite-signature -m libhint -h libhint.pyf hint.f90 skip: timedep_csr_matvec :
%.o: %.f90
	$(FORTRAN) $(INCMKL) -c $^


###############################################################################
PYTHON := python
PYVERSION := $(shell $(PYTHON) -c "import sys; print(sys.version[:3])")

PYINC := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_python_inc())")
PLATINCDIR := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_python_inc(plat_specific=True))")
PYLIBDIR1 := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYLIBDIR2 := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBPL'))")
PYLIB := $(shell $(PYTHON) -c "from distutils import sysconfig; print(sysconfig.get_config_var('LIBRARY')[3:-2])")

LINKCC := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LINKCC'))")
LINKFORSHARED := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LINKFORSHARED'))")
LIBS := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('LIBS'))")
SYSLIBS := $(shell $(PYTHON) -c "import distutils.sysconfig; print(distutils.sysconfig.get_config_var('SYSLIBS'))")

PYCFLAGS := -w -O3 -fPIC -fopenmp
LINKFLAGS := -fopenmp



.SECONDARY:
%.so: %.o
	g++ -o $@ $^ -L$(PYLIBDIR1) -L$(PYLIBDIR2) -l$(PYLIB) $(LIBS) $(SYSLIBS) $(LINKFORSHARED) $(LINKFLAGS) -shared
%.cpp: %.pyx
	cython --cplus $^
%.o: %.cpp
	g++ $(PYCFLAGS) -c -o $@ $^ -I$(PYINC) -I$(PLATINCDIR)

###############################################################################
clean:
	rm -vf *.o *.mod *.pyf *.so main *.pyc *.cpp
