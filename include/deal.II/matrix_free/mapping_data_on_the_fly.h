// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2014 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


#ifndef dealii_matrix_free_mapping_data_on_the_fly_h
#define dealii_matrix_free_mapping_data_on_the_fly_h


#include <deal.II/base/config.h>

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/enable_observer_pointer.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/matrix_free/mapping_info.h>
#include <deal.II/matrix_free/shape_info.h>


DEAL_II_NAMESPACE_OPEN


namespace internal
{
  namespace MatrixFreeFunctions
  {
    /**
     * This class provides evaluated mapping information using standard
     * deal.II information in a form that FEEvaluation and friends can use for
     * vectorized access. Since no vectorization over cells is available with
     * the DoFHandler/Triangulation cell iterators, the interface to
     * FEEvaluation's vectorization model is to use VectorizedArray::size()
     * copies of the same element. This interface is thus primarily useful for
     * evaluating several operators on the same cell, e.g., when assembling
     * cell matrices.
     *
     * As opposed to the Mapping classes in deal.II, this class does not
     * actually provide a boundary description that can be used to evaluate
     * the geometry, but it rather provides the evaluated geometry from a
     * given deal.II mapping (as passed to the constructor of this class) in a
     * form accessible to FEEvaluation.
     */
    template <int dim, typename Number>
    class MappingDataOnTheFly
    {
    public:
      /**
       * Default constructor.
       */
      MappingDataOnTheFly() = default;

      /**
       * Constructor, similar to FEValues. Since this class only evaluates the
       * geometry, no finite element has to be specified and the simplest
       * element, FE_Nothing, is used internally for the underlying FEValues
       * object.
       */
      MappingDataOnTheFly(const Mapping<dim>  &mapping,
                          const Quadrature<1> &quadrature,
                          const UpdateFlags    update_flags);

      /**
       * Constructor. This constructor is equivalent to the other one except
       * that it makes the object use a $Q_1$ mapping (i.e., an object of type
       * MappingQ(1)) implicitly.
       */
      MappingDataOnTheFly(const Quadrature<1> &quadrature,
                          const UpdateFlags    update_flags);

      /**
       * Initialize with the given cell iterator.
       */
      void
      reinit(typename dealii::Triangulation<dim>::cell_iterator cell);

      /**
       * Return whether reinit() has been called at least once, i.e., a cell
       * has been set.
       */
      bool
      is_initialized() const;

      /**
       * Return a triangulation iterator to the current cell.
       */
      typename dealii::Triangulation<dim>::cell_iterator
      get_cell() const;

      /**
       * Return a reference to the underlying FEValues object that evaluates
       * certain quantities (only mapping-related ones like Jacobians or
       * mapped quadrature points are accessible, as no finite element data is
       * actually used).
       */
      const dealii::FEValues<dim> &
      get_fe_values() const;

      /**
       * Return a reference to the underlying storage field of type
       * MappingInfoStorage of the same format as the data fields in
       * MappingInfo. This ensures compatibility with the precomputed data
       * fields in the MappingInfo class.
       */
      const MappingInfoStorage<dim, dim, Number> &
      get_data_storage() const;

      /**
       * Return a non-const reference to the underlying storage field
       * of type MappingInfoStorage of the same format as the data fields
       * in MappingInfo. This function can be used to manually fill the
       * data if the default constructor has been called and reinit() was
       * not called for a given cell.
       */
      MappingInfoStorage<dim, dim, Number> &
      get_data_storage();

      /**
       * Return a reference to 1d quadrature underlying this object.
       */
      const Quadrature<1> &
      get_quadrature() const;

    private:
      /**
       * A cell iterator in case we generate the data on the fly to be able to
       * check if we need to re-generate the information stored in this class.
       */
      typename dealii::Triangulation<dim>::cell_iterator present_cell;

      /**
       * Dummy finite element object necessary for initializing the FEValues
       * object.
       */
      std::unique_ptr<FE_Nothing<dim>> fe_dummy;

      /**
       * An underlying FEValues object that performs the (scalar) evaluation.
       */
      std::unique_ptr<dealii::FEValues<dim>> fe_values;

      /**
       * Get 1d quadrature formula to be used for reinitializing shape info.
       */
      const Quadrature<1> quadrature_1d;

      /**
       * The storage part created for a single cell and held in analogy to
       * MappingInfo.
       */
      MappingInfoStorage<dim, dim, Number> mapping_info_storage;
    };


    /*-------------------------- Inline functions ---------------------------*/

    template <int dim, typename Number>
    inline MappingDataOnTheFly<dim, Number>::MappingDataOnTheFly(
      const Mapping<dim>  &mapping,
      const Quadrature<1> &quadrature,
      const UpdateFlags    update_flags)
      : fe_dummy(std::make_unique<FE_Nothing<dim>>())
      , fe_values(std::make_unique<dealii::FEValues<dim>>(
          mapping,
          *fe_dummy,
          Quadrature<dim>(quadrature),
          MappingInfoStorage<dim, dim, Number>::compute_update_flags(
            update_flags)))
      , quadrature_1d(quadrature)
    {
      mapping_info_storage.descriptor.resize(1);
      mapping_info_storage.descriptor[0].initialize(quadrature);
      mapping_info_storage.data_index_offsets.resize(1);
      mapping_info_storage.JxW_values.resize(fe_values->n_quadrature_points);
      mapping_info_storage.jacobians[0].resize(fe_values->n_quadrature_points);
      if (update_flags & update_quadrature_points)
        {
          mapping_info_storage.quadrature_point_offsets.resize(1, 0);
          mapping_info_storage.quadrature_points.resize(
            fe_values->n_quadrature_points);
        }
      if (fe_values->get_update_flags() & update_normal_vectors)
        {
          mapping_info_storage.normal_vectors.resize(
            fe_values->n_quadrature_points);
          mapping_info_storage.normals_times_jacobians[0].resize(
            fe_values->n_quadrature_points);
        }
      Assert(!(fe_values->get_update_flags() & update_jacobian_grads),
             ExcNotImplemented());
    }



    template <int dim, typename Number>
    inline MappingDataOnTheFly<dim, Number>::MappingDataOnTheFly(
      const Quadrature<1> &quadrature,
      const UpdateFlags    update_flags)
      : MappingDataOnTheFly(::dealii::StaticMappingQ1<dim, dim>::mapping,
                            quadrature,
                            update_flags)
    {}



    template <int dim, typename Number>
    inline void
    MappingDataOnTheFly<dim, Number>::reinit(
      typename dealii::Triangulation<dim>::cell_iterator cell)
    {
      if (present_cell == cell)
        return;
      present_cell = cell;
      fe_values->reinit(present_cell);
      for (unsigned int q = 0; q < fe_values->get_quadrature().size(); ++q)
        {
          if (fe_values->get_update_flags() & update_JxW_values)
            mapping_info_storage.JxW_values[q] = fe_values->JxW(q);
          if (fe_values->get_update_flags() & update_jacobians)
            {
              Tensor<2, dim> jac = fe_values->jacobian(q);
              jac                = invert(transpose(jac));
              for (unsigned int d = 0; d < dim; ++d)
                for (unsigned int e = 0; e < dim; ++e)
                  mapping_info_storage.jacobians[0][q][d][e] = jac[d][e];
            }
          if (fe_values->get_update_flags() & update_quadrature_points)
            for (unsigned int d = 0; d < dim; ++d)
              mapping_info_storage.quadrature_points[q][d] =
                fe_values->quadrature_point(q)[d];
          if (fe_values->get_update_flags() & update_normal_vectors)
            {
              for (unsigned int d = 0; d < dim; ++d)
                mapping_info_storage.normal_vectors[q][d] =
                  fe_values->normal_vector(q)[d];
              mapping_info_storage.normals_times_jacobians[0][q] =
                mapping_info_storage.normal_vectors[q] *
                mapping_info_storage.jacobians[0][q];
            }
        }
    }



    template <int dim, typename Number>
    inline bool
    MappingDataOnTheFly<dim, Number>::is_initialized() const
    {
      return present_cell !=
             typename dealii::Triangulation<dim>::cell_iterator();
    }



    template <int dim, typename Number>
    inline typename dealii::Triangulation<dim>::cell_iterator
    MappingDataOnTheFly<dim, Number>::get_cell() const
    {
      return fe_values->get_cell();
    }



    template <int dim, typename Number>
    inline const dealii::FEValues<dim> &
    MappingDataOnTheFly<dim, Number>::get_fe_values() const
    {
      return *fe_values;
    }



    template <int dim, typename Number>
    inline const MappingInfoStorage<dim, dim, Number> &
    MappingDataOnTheFly<dim, Number>::get_data_storage() const
    {
      return mapping_info_storage;
    }



    template <int dim, typename Number>
    inline MappingInfoStorage<dim, dim, Number> &
    MappingDataOnTheFly<dim, Number>::get_data_storage()
    {
      return mapping_info_storage;
    }



    template <int dim, typename Number>
    inline const Quadrature<1> &
    MappingDataOnTheFly<dim, Number>::get_quadrature() const
    {
      return quadrature_1d;
    }


  } // end of namespace MatrixFreeFunctions
} // end of namespace internal


DEAL_II_NAMESPACE_CLOSE

#endif
