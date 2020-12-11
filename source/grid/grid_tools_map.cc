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

#include <deal.II/grid/grid_tools_map.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/dof_handler.h>


DEAL_II_NAMESPACE_OPEN


namespace GridTools
{
  template <class VolumeMeshType, class SurfaceMeshType>
  VolumeToSurfaceCellMap<VolumeMeshType, SurfaceMeshType>::
    VolumeToSurfaceCellMap(const VolumeMeshType & volume,
                           const SurfaceMeshType &surface)
    : volume(&volume)
    , surface(&surface)
  {}



  template <class VolumeMeshType, class SurfaceMeshType>
  VolumeToSurfaceCellMap<VolumeMeshType,
                         SurfaceMeshType>::~VolumeToSurfaceCellMap()
  {}



  template <class VolumeMeshType, class SurfaceMeshType>
  CellId &
  VolumeToSurfaceCellMap<VolumeMeshType, SurfaceMeshType>::get_surface_cell(
    const std::pair<CellId, unsigned int> &volume_face)
  {
    return volume_to_surface[volume_face];
  }



  template <class VolumeMeshType, class SurfaceMeshType>
  CellId &
  VolumeToSurfaceCellMap<VolumeMeshType, SurfaceMeshType>::get_surface_cell(
    const CellId &     volume_cell,
    const unsigned int face_number)
  {
    return get_surface_cell({volume_cell, face_number});
  }



  template <class VolumeMeshType, class SurfaceMeshType>
  std::pair<CellId, unsigned int> &
  VolumeToSurfaceCellMap<VolumeMeshType, SurfaceMeshType>::get_volume_face(
    const CellId &surface_cell)
  {
    return surface_to_volume[surface_cell];
  }



  template <class VolumeMeshType, class SurfaceMeshType>
  void
  VolumeToSurfaceCellMap<VolumeMeshType, SurfaceMeshType>::update_maps()
  {
    // update volume_to_surface map
    // ...

    // update surface_to_volume map
    // ...
  }
} // namespace GridTools


// explicit instantiations
#include "grid_tools_map.inst"

DEAL_II_NAMESPACE_CLOSE
