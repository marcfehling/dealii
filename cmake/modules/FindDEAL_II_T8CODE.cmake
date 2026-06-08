## ------------------------------------------------------------------------
##
## SPDX-License-Identifier: LGPL-2.1-or-later
## Copyright (C) 2012 - 2023 by the deal.II authors
##
## This file is part of the deal.II library.
##
## Part of the source code is dual licensed under Apache-2.0 WITH
## LLVM-exception OR LGPL-2.1-or-later. Detailed license information
## governing the source code and code contributions can be found in
## LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
##
## ------------------------------------------------------------------------

#
# Try to find the T8CODE library
#
# This module exports:
#   T8CODE_DIR
#   T8CODE_LIBRARIES
#   T8CODE_INCLUDE_DIRS
#   T8CODE_VERSION
#   T8CODE_VERSION_MAJOR
#   T8CODE_VERSION_MINOR
#   T8CODE_VERSION_PATCH
#   T8CODE_WITH_MPI
#

set(T8CODE_DIR "" CACHE PATH
  "An optional hint to a t8code installation/directory"
  )
set_if_empty(T8CODE_DIR "$ENV{T8CODE_DIR}")



## add additional hints for different ways to install t8code/p4est/sc
find_package(SC CONFIG
             HINTS ${T8CODE_DIR}/cmake ${T8CODE_DIR}/install/cmake)

find_package(P4EST CONFIG
             HINTS ${T8CODE_DIR}/cmake ${T8CODE_DIR}/install/cmake)

find_package(T8CODE CONFIG
             HINTS ${T8CODE_DIR}/cmake ${T8CODE_DIR}/install/cmake)

if(T8CODE_FOUND)
  #
  # Adopt variable name to be consistent within deal.II.
  # - extract Version Numbers
  # -  MPI Status.
  #
  set(T8CODE_VERSION "${T8_VERSION}")
  set(T8CODE_VERSION_MAJOR "${T8_VERSION_MAJOR}")
  set(T8CODE_VERSION_MINOR "${T8_VERSION_MINOR}")
  set(T8CODE_VERSION_PATCH "${T8_VERSION_PATCH}")
  message(STATUS "T8CODE VERSION: ${T8CODE_VERSION} ${T8CODE_VERSION_MAJOR} ${T8CODE_VERSION_MINOR} ${T8CODE_VERSION_PATCH}" )

  set(T8CODE_WITH_MPI "${T8CODE_ENABLE_MPI}")

  #
  # t8code does not export the include directory. We find it ourselves.
  #
  #~ message(STATUS "T8CODE_INCLUDE_DIR: ${T8CODE_INCLUDE_DIR}" )
  #~ message(STATUS "T8CODE_INCLUDE_DIRS: ${T8CODE_INCLUDE_DIRS}")
  deal_ii_find_path(T8CODE_INCLUDE_DIR t8.h
    HINTS ${T8CODE_DIR}/.. ${T8CODE_DIR}
    PATH_SUFFIXES include
    )

  set(_targets T8CODE::T8)
endif()

process_feature(T8CODE
  TARGETS
    REQUIRED _targets
  LIBRARIES
    OPTIONAL LAPACK_LIBRARIES MPI_C_LIBRARIES
  INCLUDE_DIRS
    REQUIRED T8CODE_INCLUDE_DIR
  CLEAR
    T8CODE_INCLUDE_DIR P4EST_DIR SC_DIR
  )
