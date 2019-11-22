// ---------------------------------------------------------------------
//
// Copyright (C) 2019 by the deal.II authors
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

#ifndef dealii_distributed_error_predictor_h
#define dealii_distributed_error_predictor_h

#include <deal.II/base/config.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/hp/dof_handler.h>

#include <vector>


// forward declarations?


DEAL_II_NAMESPACE_OPEN

namespace parallel
{
  namespace distributed
  {
    /**
     * TODO: Doc.
     *
     * @ingroup distributed
     * @author Marc Fehling, 2019
     */
    template <int dim, int spacedim = dim>
    class ErrorPredictor
    {
    public:
      /**
       * Constructor.
       *
       * @param[in] dof The DoFHandler or hp::DoFHandler on which all
       *   operations will happen. At the time when this constructor
       *   is called, the DoFHandler still points to the triangulation
       *   before the refinement in question happens.
       */
      ErrorPredictor(const hp::DoFHandler<dim, spacedim> &dof);

      /**
       * Destructor.
       */
      ~ErrorPredictor() = default;

      /**
       * Prepare the current object for coarsening and refinement. It
       * stores the dof indices of each cell and stores the dof values of the
       * vectors in @p all_in in each cell that'll be coarsened. @p all_in
       * includes all vectors that are to be interpolated onto the new
       * (refined and/or coarsened) grid.
       */
      void
      prepare_for_coarsening_and_refinement(
        const std::vector<const Vector<float> *> &all_in,
        const double                              gamma_p = std::sqrt(0.1),
        const double                              gamma_h = 1.,
        const double                              gamma_n = 1.);

      /**
       * Same as the previous function but for only one discrete function to be
       * interpolated.
       */
      void
      prepare_for_coarsening_and_refinement(
        const Vector<float> &in,
        const double         gamma_p = std::sqrt(0.1),
        const double         gamma_h = 1.,
        const double         gamma_n = 1.);

      /**
       * Interpolate the data previously stored in this object before the mesh
       * was refined or coarsened onto the current set of cells. Do so for
       * each of the vectors provided to
       * prepare_for_coarsening_and_refinement() and write the result into the
       * given set of vectors.
       */
      void
      unpack(std::vector<Vector<float> *> &all_out);

      /**
       * Same as the previous function. It interpolates only one function. It
       * assumes the vectors having the right sizes (i.e.
       * <tt>in.size()==n_dofs_old</tt>, <tt>out.size()==n_dofs_refined</tt>)
       *
       * Multiple calling of this function is NOT allowed. Interpolating
       * several functions can be performed in one step by using
       * <tt>interpolate (all_in, all_out)</tt>
       */
      void
      unpack(Vector<float> &out);

      /**
       * Prepare the serialization of the given vector. The serialization is
       * done by Triangulation::save(). The given vector needs all information
       * on the locally active DoFs (it must be ghosted). See documentation of
       * this class for more information.
       */
      void
      prepare_for_serialization(const Vector<float> &in,
                                const double         gamma_p = std::sqrt(0.1),
                                const double         gamma_h = 1.,
                                const double         gamma_n = 1.);

      /**
       * Same as the function above, only for a list of vectors.
       */
      void
      prepare_for_serialization(
        const std::vector<const Vector<float> *> &all_in,
        const double                              gamma_p = std::sqrt(0.1),
        const double                              gamma_h = 1.,
        const double                              gamma_n = 1.);

      /**
       * Execute the deserialization of the given vector. This needs to be
       * done after calling Triangulation::load(). The given vector must be a
       * fully distributed vector without ghost elements. See documentation of
       * this class for more information.
       */
      void
      deserialize(Vector<float> &in);


      /**
       * Same as the function above, only for a list of vectors.
       */
      void
      deserialize(std::vector<Vector<float> *> &all_in);

    private:
      /**
       * Pointer to the degree of freedom handler to work with.
       */
      SmartPointer<const hp::DoFHandler<dim, spacedim>,
                   ErrorPredictor<dim, spacedim>>
        dof_handler;

      /**
       * A vector that stores pointers to all the vectors we are supposed to
       * copy over from the old to the new mesh.
       */
      std::vector<const Vector<float> *> error_indicators;

      /**
       * The handle that the Triangulation has assigned to this object
       * with which we can access our memory offset and our pack function.
       */
      unsigned int handle;

      /**
       * Parameters.
       */
      double gamma_p, gamma_h, gamma_n;

      /**
       * A callback function used to pack the data on the current mesh into
       * objects that can later be retrieved after refinement, coarsening and
       * repartitioning.
       */
      std::vector<char>
      pack_callback(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell,
        const typename Triangulation<dim, spacedim>::CellStatus     status);

      /**
       * A callback function used to unpack the data on the current mesh that
       * has been packed up previously on the mesh before refinement,
       * coarsening and repartitioning.
       */
      void
      unpack_callback(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell,
        const typename Triangulation<dim, spacedim>::CellStatus     status,
        const boost::iterator_range<std::vector<char>::const_iterator>
          &                           data_range,
        std::vector<Vector<float> *> &all_out);


      /**
       * Registers the pack_callback() function to the
       * parallel::distributed::Triangulation that has been assigned to the
       * DoFHandler class member and stores the returning handle.
       */
      void
      register_data_attach();
    };


  } // namespace distributed
} // namespace parallel



DEAL_II_NAMESPACE_CLOSE

#endif
