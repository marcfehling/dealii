// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
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

// #include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/grid/grid_tools_mapping.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/dof_handler.h>


DEAL_II_NAMESPACE_OPEN


namespace GridTools
{
  template <template <int, int> class MeshType, int volumedim, int spacedim>
  MappingVolumeSurface<MeshType, volumedim, spacedim>::MappingVolumeSurface(
    const MeshType<volumedim, spacedim> & volume,
    const MeshType<surfacedim, spacedim> &surface)
    : volume(&volume)
    , surface(&surface)
  {}



  template <template <int, int> class MeshType, int volumedim, int spacedim>
  MappingVolumeSurface<MeshType, volumedim, spacedim>::~MappingVolumeSurface()
  {}



  template <template <int, int> class MeshType, int volumedim, int spacedim>
  CellId &
  MappingVolumeSurface<MeshType, volumedim, spacedim>::get_surface_cell(
    const std::pair<CellId, unsigned int> &volume_face)
  {
    return volume_to_surface[volume_face];
  }



  template <template <int, int> class MeshType, int volumedim, int spacedim>
  std::pair<CellId, unsigned int> &
  MappingVolumeSurface<MeshType, volumedim, spacedim>::get_volume_face(
    const CellId &surface_cell)
  {
    return surface_to_volume[surface_cell];
  }



  template <template <int, int> class MeshType, int volumedim, int spacedim>
  void
  MappingVolumeSurface<MeshType, volumedim, spacedim>::update_mapping()
  {
    // update volume_to_surface map
    // ...

    // update surface_to_volume map
    // ...
  }
} // namespace GridTools


// explicit instantiations
#include "grid_tools_mapping.inst"

DEAL_II_NAMESPACE_CLOSE
