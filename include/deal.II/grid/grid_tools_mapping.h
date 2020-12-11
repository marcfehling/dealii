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

#ifndef dealii_grid_tools_mapping_h
#define dealii_grid_tools_mapping_h


#include <deal.II/base/config.h>

#include <deal.II/base/smartpointer.h>

#include <deal.II/grid/cell_id.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <boost/signals2.hpp>

#include <map>
#include <utility>


DEAL_II_NAMESPACE_OPEN


namespace GridTools
{
  /**
   * TODO: Doc
   */
  template <int volumedim, int spacedim = volumedim>
  class MappingVolumeSurface : public Subscriptor
  {
  public:
    /**
     * Dimension of the surface triangulation has to be one lower than the
     * surface triangulation.
     */
    static constexpr unsigned int surfacedim = volumedim - 1;
    static_assert(surfacedim > 0, "Surfaces must be at least one dimensional");

    /**
     * Constructor.
     */
    MappingVolumeSurface(const Triangulation<volumedim, spacedim> & volume,
                         const Triangulation<surfacedim, spacedim> &surface);

    /**
     * Destructor.
     */
    ~MappingVolumeSurface();

    /**
     * Get surface cell.
     */
    CellId &
    get_surface_cell(const std::pair<CellId, unsigned int> &volume_face);


    /**
     * Get volume face.
     */
    std::pair<CellId, unsigned int> &
    get_volume_face(const CellId &surface_cell);


  private:
    /**
     * The volume triangulation.
     */
    SmartPointer<const Triangulation<volumedim, spacedim>> volume;

    /**
     * The surface triangulation.
     */
    SmartPointer<const Triangulation<surfacedim, spacedim>> surface;

    /**
     * Volume to surface map.
     */
    std::map<std::pair<CellId, unsigned int>, CellId> volume_to_surface;

    /**
     * Surface to volume map.
     */
    std::map<CellId, std::pair<CellId, unsigned int>> surface_to_volume;

    /**
     * Connection to signal of volume triangulation.
     */
    boost::signals2::connection connection_to_volume;

    /**
     * Update volume.
     */
    void
    update_volume_faces();

    /**
     * Connection to signal of surface triangulation.
     */
    boost::signals2::connection connection_to_surface;

    /**
     * Update surface.
     */
    void
    update_surface_cells();

    /**
     * Allow the extracting function to access the internal mapping.
     */
    friend MappingVolumeSurface<volumedim, spacedim>
    GridGenerator::extract_surface_mesh<volumedim, spacedim>(
      const Triangulation<volumedim, spacedim> &,
      Triangulation<surfacedim, spacedim> &,
      const std::set<types::boundary_id> &);
  };
} // namespace GridTools


DEAL_II_NAMESPACE_CLOSE

#endif
