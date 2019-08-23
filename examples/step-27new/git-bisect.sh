# !bin/bash

# store related paths for convenience
PATH_BUILD_PETSC="/raid/fehling/petsc"
PATH_BUILD_DEALII="/raid/fehling/build-dealii-petsc-git"
PATH_INSTALL_DEALII="/raid/fehling/build-dealii-petsc-git/install"

PATH_BUILD_TEST="$PATH_INSTALL_DEALII/examples/step-27new/build"

# clean and rebuild PETSc
#   NOTE: we expect that PETSc has already been configured via configure
cd "$PATH_BUILD_PETSC"
export PETSC_DIR=`pwd`
export PETSC_ARCH=x86_64
make clean
make -j80 all test

# rebuild deal.II
#   NOTE: we expect that deal.II has already been configured via cmake
cd "$PATH_BUILD_DEALII"
make -j80 install

# perform failing test
rm -rf "$PATH_BUILD_TEST"
mkdir -p "$PATH_BUILD_TEST"
cd "$PATH_BUILD_TEST"
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
mpirun -np 20 ./step-27

# git bisect does not understand SIGSEGV, so we'll convert it
if [ "$?" -eq "0" ]; then
  exit 0
else
  exit 1
fi
