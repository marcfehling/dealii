# !bin/bash


# GIT BISECT SESSION:
#   find that specific commit in PETSc at which a specific bug in deal.ii
#   appears for the very first time
#
# interval of commits to check
#   bad : 00c67f3
#   good: d49172c
#
# run a git bisect session as follows
#   git bisect start
#   git bisect bad 00c67f3
#   git bisect good d49172c
#   git bisect run ./git-bisect.sh
#   git bisect reset


# access compiler shortcuts for openmpi, i.e. mpicc, mpicxx, mpif90, ...
export PATH="$PATH:/usr/lib64/openmpi/bin"


# store paths for convenience
PATH_BUILD_PETSC="/raid/fehling/petsc"

PATH_SOURCE_DEALII="/raid/fehling/dealii"
PATH_BUILD_DEALII="/raid/fehling/build-dealii-petsc-git"
PATH_INSTALL_DEALII="$PATH_BUILD_DEALII/install"




# build petsc from scratch
cd "$PATH_BUILD_PETSC"
export PETSC_DIR=`pwd`
export PETSC_ARCH=x86_64
make distclean
./config/configure.py --with-shared-libraries=1 --with-x=0 --with-mpi=1 --download-hypre=1 --download-superlu_dist=yes --download-mumps=yes --download-parmetis=yes --download-scalapack=yes --download-metis=yes --download-blacs=yes
make all test
# leave this script informing 'git bisect' to skip this petsc commit
# since petsc cannot be built
if [ "$?" -ne "0" ]; then
  exit 125
fi


# build and install deal.ii
cd "$PATH_BUILD_DEALII"
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX="$PATH_INSTALL_DEALII" -DDEAL_II_WITH_MPI=ON -DDEAL_II_WITH_P4EST=ON -DP4EST_DIR=/raid/fehling/bin/p4est-2.2 -DDEAL_II_WITH_PETSC=ON -DPETSC_ARCH=x86_64 -DPETSC_DIR="$PATH_BUILD_PETSC" "$PATH_SOURCE_DEALII"
make -j80 install
# leave this script with an error if deal.ii cannot be built
#
# we suspect that "include/deal.II/lac/petsc_compatibility.h"
# is responsible for the 'segfault' in the example below,
# so we'll consider this petsc commit as 'bad'
# when we fail to build deal.ii
if [ "$?" -ne "0" ]; then
  exit 1
fi


# perform crucial test
cd "$PATH_INSTALL_DEALII/examples/step-27new"
rm -rf "build"
mkdir -p "build"
cd "build"
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
mpirun -np 20 ./step-27new
# git bisect does not understand SIGSEGV, so we'll convert it
if [ "$?" -ne "0" ]; then
  exit 1
fi


# return success if everything went fine
exit 0
