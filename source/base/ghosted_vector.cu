// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2019 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

#include <deal.II/base/parallel_vector.h>
#include <deal.II/base/parallel_vector.templates.h>

DEAL_II_NAMESPACE_OPEN


template class GhostedVector<float, MemorySpace::CUDA>;
template class GhostedVector<double, MemorySpace::CUDA>;
template void
GhostedVector<float, MemorySpace::Host>::import<MemorySpace::CUDA>(
  const GhostedVector<float, MemorySpace::CUDA> &,
  VectorOperation::values);
template void
GhostedVector<double, MemorySpace::Host>::import<MemorySpace::CUDA>(
  const GhostedVector<double, MemorySpace::CUDA> &,
  VectorOperation::values);

template void
GhostedVector<float, MemorySpace::CUDA>::import<MemorySpace::Host>(
  const GhostedVector<float, MemorySpace::Host> &,
  VectorOperation::values);
template void
GhostedVector<double, MemorySpace::CUDA>::import<MemorySpace::Host>(
  const GhostedVector<double, MemorySpace::Host> &,
  VectorOperation::values);

template void
GhostedVector<float, MemorySpace::CUDA>::import<MemorySpace::CUDA>(
  const GhostedVector<float, MemorySpace::CUDA> &,
  VectorOperation::values);
template void
GhostedVector<double, MemorySpace::CUDA>::import<MemorySpace::CUDA>(
  const GhostedVector<double, MemorySpace::CUDA> &,
  VectorOperation::values);


DEAL_II_NAMESPACE_CLOSE
