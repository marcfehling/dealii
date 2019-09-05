# !bin/bash


# GIT BISECT SESSION:
#   find that specific commit in hypre at which a specific bug in deal.ii
#   appears for the very first time
#
# interval of commits to check
#   bad : 92a582b
#   good: f4e70c8
#
# run a git bisect session as follows
#   git bisect start
#   git bisect bad 92a582b
#   git bisect good f4e70c8
#   git bisect run ./git-bisect.sh
#   git bisect reset


# access compiler shortcuts for openmpi, i.e. mpicc, mpicxx, mpif90, ...
export PATH="$PATH:/usr/lib64/openmpi/bin"


# store paths for convenience
export PATH_BUILD_HYPRE="/raid/fehling/hypre"
export PATH_BUILD_PETSC="/raid/fehling/petsc"

export PATH_SOURCE_DEALII="/raid/fehling/dealii"
export PATH_BUILD_DEALII="/raid/fehling/build-dealii-petsc-git"
export PATH_INSTALL_DEALII="$PATH_BUILD_DEALII/install"


# helper function
build_hypre () {
  export PATH_INSTALL_HYPRE="$1/hypre"

  cd $1
  ./configure --enable-shared CXXFLAGS="-std=c++11"
  make -j80
  # leave this script informing 'git bisect' to skip this hypre commit
  # since hypre cannot be built
  if [ "$?" -ne "0" ]; then
    exit 125
  fi
}


# build hypre from scratch
cd "$PATH_BUILD_HYPRE"
git clean -fd
if [ -d "$PATH_BUILD_HYPRE/src" ]; then
  build_hypre "$PATH_BUILD_HYPRE/src"
else
  build_hypre "$PATH_BUILD_HYPRE"
fi


# build petsc from scratch
cd "$PATH_BUILD_PETSC"
export PETSC_DIR=`pwd`
export PETSC_ARCH=x86_64
make distclean
./config/configure.py --with-shared-libraries=1 --with-cxx-flags=C++11 --with-x=0 --with-mpi=1 --with-hypre-dir="$PATH_INSTALL_HYPRE" --download-superlu_dist=yes --download-mumps=yes --download-parmetis=yes --download-scalapack=yes --download-metis=yes --download-blacs=yes
make all test
# leave this script informing 'git bisect' to skip this hypre commit
# since petsc cannot be built
if [ "$?" -ne "0" ]; then
  exit 125
fi


# build and install deal.ii
rm -rf "$PATH_BUILD_DEALII"
mkdir -p "$PATH_BUILD_DEALII"
cd "$PATH_BUILD_DEALII"
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX="$PATH_INSTALL_DEALII" -DDEAL_II_WITH_MPI=ON -DDEAL_II_WITH_P4EST=ON -DP4EST_DIR=/raid/fehling/bin/p4est-2.2 -DDEAL_II_WITH_PETSC=ON -DPETSC_ARCH=x86_64 -DPETSC_DIR="$PATH_BUILD_PETSC" "$PATH_SOURCE_DEALII"
make -j80 install
# leave this script informing 'git bisect' to skip this hypre commit
# since deal.ii cannot be built
if [ "$?" -ne "0" ]; then
  exit 125
fi


# perform crucial test
cd "$PATH_INSTALL_DEALII/examples/step-27new"
cmake -DCMAKE_BUILD_TYPE=Debug .
make
mpirun -np 20 ./step-27new
# git bisect does not understand SIGSEGV, so we'll convert it
if [ "$?" -ne "0" ]; then
  exit 1
fi


# return success if everything went fine
exit 0
