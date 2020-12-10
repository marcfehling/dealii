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

#include <deal.II/grid/grid_tools_mapping.h>


DEAL_II_NAMESPACE_OPEN


namespace GridTools
{
  template <int volumedim, int spacedim>
  MappingVolumeSurface<volumedim, spacedim>::MappingVolumeSurface(
    const Triangulation<volumedim, spacedim> & volume,
    const Triangulation<surfacedim, spacedim> &surface)
    : volume(&volume)
    , surface(&surface)
  {
    // register the update of cells and faces after adaptation
    // TODO: Check if those are the right signals to connect to
    connection_to_volume = volume.signals.post_refinement.connect(
      [this] { this->update_volume_faces(); });
    connection_to_surface = surface.signals.post_refinement.connect(
      [this] { this->update_surface_cells(); });

    // create initial mapping
    // ...
  }



  template <int volumedim, int spacedim>
  MappingVolumeSurface<volumedim, spacedim>::~MappingVolumeSurface()
  {
    connection_to_volume.disconnect();
    connection_to_surface.disconnect();
  }



  template <int volumedim, int spacedim>
  CellId &
  MappingVolumeSurface<volumedim, spacedim>::get_surface_cell(
    const std::pair<CellId, unsigned int> &volume_face)
  {
    return volume_to_surface[volume_face];
  }



  template <int volumedim, int spacedim>
  std::pair<CellId, unsigned int> &
  MappingVolumeSurface<volumedim, spacedim>::get_volume_face(
    const CellId &surface_cell)
  {
    return surface_to_volume[surface_cell];
  }



  template <int volumedim, int spacedim>
  void
  MappingVolumeSurface<volumedim, spacedim>::update_volume_faces()
  {
    // update volume_face in volume_to_surface_map
    // ...

    // update volume_face in surface_to_volume_map
    // ...
  }



  template <int volumedim, int spacedim>
  void
  MappingVolumeSurface<volumedim, spacedim>::update_surface_cells()
  {
    // update surface_cell in volume_to_surface_map
    // ...

    // update surface_cell in surface_to_volume_map
    // ...
  }
} // namespace GridTools


// explicit instantiations
#include "grid_tools_mapping.inst"

DEAL_II_NAMESPACE_CLOSE
