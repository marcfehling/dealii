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

// #include <boost/signals2.hpp>

#include <map>
#include <utility>


DEAL_II_NAMESPACE_OPEN


namespace GridTools
{
  /**
   * TODO: Doc
   */
  template <class VolumeMeshType, class SurfaceMeshType>
  class VolumeToSurfaceCellMap : public Subscriptor
  {
  public:
    /**
     * Dimension of the surface triangulation has to be one lower than the
     * surface triangulation.
     */
    static_assert(
      VolumeMeshType::dimension == SurfaceMeshType::dimension + 1,
      "Dimension of volume and surface meshes do not fulfil the codim 1 criterion.");
    static_assert(
      VolumeMeshType::space_dimension == SurfaceMeshType::space_dimension,
      "Space dimensions of volume and surface meshes do not match.");

    /**
     * Extract dimension parameters.
     */
    static constexpr unsigned int volumedim  = VolumeMeshType::dimension;
    static constexpr unsigned int surfacedim = SurfaceMeshType::dimension;
    static constexpr unsigned int spacedim   = VolumeMeshType::space_dimension;

    /**
     * Destructor.
     */
    ~VolumeToSurfaceCellMap();

    /**
     * Update mapping after adaptation.
     */
    void
    update_maps();

    /**
     * Get surface cell.
     */
    CellId &
    get_surface_cell(const std::pair<CellId, unsigned int> &volume_face);

    /**
     * Same as above, but for two separate parameters.
     */
    CellId &
    get_surface_cell(const CellId &volume_cell, const unsigned int face_number);

    /**
     * Get volume face.
     */
    std::pair<CellId, unsigned int> &
    get_volume_face(const CellId &surface_cell);


  private:
    /**
     * Private constructor.
     *
     * Objects of this class will only be constructed during
     * GridGenerator::extract_surface_mesh().
     */
    VolumeToSurfaceCellMap(const VolumeMeshType & volume,
                           const SurfaceMeshType &surface);

    /**
     * The volume triangulation.
     */
    SmartPointer<const VolumeMeshType> volume;

    /**
     * The surface triangulation.
     */
    SmartPointer<const SurfaceMeshType> surface;

    /**
     * Volume to surface map.
     */
    std::map<std::pair<CellId, unsigned int>, CellId> volume_to_surface;

    /**
     * Surface to volume map.
     */
    std::map<CellId, std::pair<CellId, unsigned int>> surface_to_volume;

    /**
     * Allow the extracting function to access the internal mapping.
     */
    friend VolumeToSurfaceCellMap<VolumeMeshType, SurfaceMeshType>
    GridGenerator::extract_surface_mesh<VolumeMeshType, SurfaceMeshType>(
      const VolumeMeshType &,
      SurfaceMeshType &,
      const std::set<types::boundary_id> &);
  };
} // namespace GridTools


DEAL_II_NAMESPACE_CLOSE

#endif
