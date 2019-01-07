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

#include <deal.II/base/quadrature.h>

#include <deal.II/fe/fe_series.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/la_parallel_block_vector.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/petsc_block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/trilinos_epetra_vector.h>
#include <deal.II/lac/trilinos_parallel_block_vector.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/smoothness_estimator.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>


DEAL_II_NAMESPACE_OPEN


namespace SmoothnessEstimator
{
  namespace
  {
    /**
     * Resizes @p coeff to @p N in each dimension.
     */
    template <int dim, typename CoefficientType>
    void
    resize(Table<dim, CoefficientType> &coeff, const unsigned int N)
    {
      TableIndices<dim> size;
      for (unsigned int d = 0; d < dim; d++)
        size[d] = N;
      coeff.reinit(size);
    }



    /**
     * We will need to take the maximum absolute value of Fourier coefficients
     * which correspond to $k$-vector $|{\bf k}|= const$. To filter the
     * coefficients Table we will use the FESeries::process_coefficients() which
     * requires a predicate to be specified. The predicate should operate on
     * TableIndices and return a pair of <code>bool</code> and <code>unsigned
     * int</code>. The latter is the value of the map from TableIndicies to
     * unsigned int.  It is used to define subsets of coefficients from which we
     * search for the one with highest absolute value, i.e. $l^\infty$-norm. The
     * <code>bool</code> parameter defines which indices should be used in
     * processing. In the current case we are interested in coefficients which
     * correspond to $0 < i^2+j^2 < N^2$ and $0 < i^2+j^2+k^2 < N^2$ in 2D and
     * 3D, respectively.
     */
    template <int dim>
    std::pair<bool, unsigned int>
    index_norm_less_than_N_squared(const TableIndices<dim> &ind,
                                   const unsigned int       N)
    {
      unsigned int v = 0;
      for (unsigned int i = 0; i < dim; i++)
        v += ind[i] * ind[i];
      if (v > 0 && v < N * N)
        return std::make_pair(true, v);
      else
        return std::make_pair(false, v);
    }
  } // namespace



  template <int dim, int spacedim, typename VectorType>
  void
  legendre_coefficient_decay(
    FESeries::Legendre<dim, spacedim> &                  fe_legendre,
    const hp::DoFHandler<dim, spacedim> &                dof_handler,
    const VectorType &                                   solution,
    Vector<float> &                                      smoothness_indicators,
    const std::function<void(std::vector<bool> &flags)> &coefficients_predicate,
    const double smallest_abs_coefficient,
    const bool   only_flagged_cells)
  {
    Assert(smallest_abs_coefficient >= 0.,
           ExcMessage("smallest_abs_coefficient should be non-negative."));

    using number = typename VectorType::value_type;
    using number_coeff =
      typename FESeries::Legendre<dim, spacedim>::CoefficientType;

    smoothness_indicators.reinit(
      dof_handler.get_triangulation().n_active_cells());

    const unsigned int max_degree =
      dof_handler.get_fe_collection().max_degree();

    Table<dim, number_coeff> expansion_coefficients;
    resize(expansion_coefficients,
           fe_legendre.get_n_coefficients_per_direction());

    Vector<number> local_dof_values;

    // auxiliary vector to do linear regression
    std::vector<number_coeff> x;
    std::vector<number_coeff> y;

    x.reserve(max_degree);
    y.reserve(max_degree);

    // precalculate predicates for each degree:
    std::vector<std::vector<bool>> predicates(max_degree);
    for (unsigned int p = 1; p <= max_degree; ++p)
      {
        auto &pred = predicates[p - 1];
        // we have p+1 coefficients for degree p
        pred.resize(p + 1);
        coefficients_predicate(pred);
      }

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() &&
          (!only_flagged_cells || cell->refine_flag_set() ||
           cell->coarsen_flag_set()))
        {
          local_dof_values.reinit(cell->get_fe().dofs_per_cell);

          const unsigned int pe = cell->get_fe().degree;

          Assert(pe > 0, ExcInternalError());
          const auto &pred = predicates[pe - 1];

          // since we use coefficients with indices [1,pe] in each direction,
          // the number of coefficients we need to calculate is at least N=pe+1
          AssertIndexRange(pe, fe_legendre.get_n_coefficients_per_direction());

          cell->get_dof_values(solution, local_dof_values);
          fe_legendre.calculate(local_dof_values,
                                cell->active_fe_index(),
                                expansion_coefficients);

          // choose the smallest decay of coefficients in each direction,
          // i.e. the maximum decay slope k_v as in exp(-k_v)
          number_coeff k_v = std::numeric_limits<number_coeff>::infinity();
          for (unsigned int d = 0; d < dim; d++)
            {
              x.resize(0);
              y.resize(0);

              // will use all non-zero coefficients allowed by the predicate
              // function
              Assert(pred.size() == pe + 1, ExcInternalError());
              for (unsigned int i = 0; i <= pe; i++)
                if (pred[i])
                  {
                    TableIndices<dim> ind;
                    ind[d] = i;
                    const number_coeff coeff_abs =
                      std::abs(expansion_coefficients(ind));

                    if (coeff_abs > smallest_abs_coefficient)
                      {
                        y.push_back(std::log(coeff_abs));
                        x.push_back(i);
                      }
                  }

              // in case we don't have enough non-zero coefficient to fit,
              // skip this direction
              if (x.size() < 2)
                continue;

              const std::pair<number_coeff, number_coeff> fit =
                FESeries::linear_regression(x, y);

              // decay corresponds to negative slope
              // take the lesser negative slope along each direction
              k_v = std::min(k_v, -fit.first);
            }

          smoothness_indicators(cell->active_cell_index()) =
            static_cast<float>(k_v);
        }
  }



  template <int dim, int spacedim, typename VectorType>
  void
  legendre_coefficient_decay(
    const hp::DoFHandler<dim, spacedim> &                dof_handler,
    const VectorType &                                   solution,
    Vector<float> &                                      smoothness_indicators,
    const std::function<void(std::vector<bool> &flags)> &coefficients_predicate,
    const double smallest_abs_coefficient,
    const bool   only_flagged_cells)
  {
    const unsigned int max_degree =
      dof_handler.get_fe_collection().max_degree();
    const unsigned int n_modes = max_degree + 1;

    // We initialize a FESeries::Legendre expansion object object which will be
    // used to calculate the expansion coefficients. In addition to the
    // hp::FECollection, we need to provide quadrature rules hp::QCollection for
    // integration on the reference cell.
    // We will need to assemble the expansion matrices for each of the finite
    // elements we deal with, i.e. the matrices F_k,j. We have to do that for
    // each of the finite elements in use. To that end we need a quadrature
    // rule. As a default, we use the same quadrature formula for each finite
    // element, namely a Gauss formula that yields exact results for the
    // highest order Legendre polynomial used.
    const QGauss<dim>  quadrature(max_degree + 1);
    const QSorted<dim> quadrature_sorted(quadrature);

    hp::QCollection<dim> expansion_q_collection;
    for (unsigned int i = 0; i < dof_handler.get_fe_collection().size(); ++i)
      expansion_q_collection.push_back(quadrature_sorted);

    FESeries::Legendre<dim, spacedim> legendre(n_modes,
                                               dof_handler.get_fe_collection(),
                                               expansion_q_collection);

    legendre_coefficient_decay(legendre,
                               dof_handler,
                               solution,
                               smoothness_indicators,
                               coefficients_predicate,
                               smallest_abs_coefficient,
                               only_flagged_cells);
  }



  template <int dim, int spacedim, typename VectorType>
  void
  fourier_coefficient_decay(FESeries::Fourier<dim, spacedim> &   fe_series,
                            const hp::DoFHandler<dim, spacedim> &dof_handler,
                            const VectorType &                   solution,
                            Vector<float> &             smoothness_indicators,
                            const VectorTools::NormType regression_strategy,
                            const bool                  only_flagged_cells)
  {
    using number = typename VectorType::value_type;
    using number_coeff =
      typename FESeries::Fourier<dim, spacedim>::CoefficientType;

    smoothness_indicators.reinit(
      dof_handler.get_triangulation().n_active_cells());

    const unsigned int N = fe_series.get_n_coefficients_per_direction();

    Table<dim, number_coeff> expansion_coefficients;
    resize(expansion_coefficients, N);

    Vector<number>                                            local_dof_values;
    std::vector<double>                                       ln_k;
    std::pair<std::vector<unsigned int>, std::vector<double>> res;
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned() &&
          (!only_flagged_cells || cell->refine_flag_set() ||
           cell->coarsen_flag_set()))
        {
          local_dof_values.reinit(cell->get_fe().dofs_per_cell);

          // Inside the loop, we first need to get the values of the local
          // degrees of freedom and then need to compute the series
          // expansion by multiplying this vector with the matrix ${\cal F}$
          // corresponding to this finite element.
          cell->get_dof_values(solution, local_dof_values);

          fe_series.calculate(local_dof_values,
                              cell->active_fe_index(),
                              expansion_coefficients);

          // We fit our exponential decay of expansion coefficients to the
          // provided regression_strategy on each possible value of |k|. To
          // this end, we use FESeries::process_coefficients() to rework
          // coefficients into the desired format.
          res = FESeries::process_coefficients<dim>(
            expansion_coefficients,
            [N](const TableIndices<dim> &indices) {
              return index_norm_less_than_N_squared<dim>(indices, N);
            },
            regression_strategy);

          Assert(res.first.size() == res.second.size(), ExcInternalError());

          // Prepare linear equation for the logarithmic least squares fit.
          //
          // First, calculate ln(|k|).
          //
          // For Fourier expansion, this translates to
          // ln(2*pi*sqrt(predicate)) = ln(2*pi) + 0.5*ln(predicate). Since
          // we are just interested in the slope of a linear regression
          // later, we omit the ln(2*pi) factor.
          ln_k.resize(res.first.size());
          for (unsigned int f = 0; f < res.first.size(); ++f)
            ln_k[f] = 0.5 * std::log(static_cast<double>(res.first[f]));

          // Second, calculate ln(U_k).
          for (auto &residual_element : res.second)
            residual_element = std::log(residual_element);

          // Last, do the linear regression.
          float regularity = 0.;
          if (res.first.size() > 1)
            {
              const auto fit = FESeries::linear_regression(ln_k, res.second);
              // Compute regularity s = mu - dim/2
              regularity = static_cast<float>(-fit.first - .5 * dim);
            }

          // Store result in the vector of estimated values for each cell.
          smoothness_indicators(cell->active_cell_index()) = regularity;
        }
  }



  template <int dim, int spacedim, typename VectorType>
  void
  fourier_coefficient_decay(const hp::DoFHandler<dim, spacedim> &dof_handler,
                            const VectorType &                   solution,
                            Vector<float> &             smoothness_indicators,
                            const VectorTools::NormType regression_strategy,
                            const bool                  only_flagged_cells)
  {
    const unsigned int n_modes =
      std::max<unsigned int>(3,
                             dof_handler.get_fe_collection().max_degree() + 1);

    // We initialize a series expansion object object which will be used to
    // calculate the expansion coefficients. In addition to the
    // hp::FECollection, we need to provide quadrature rules hp::QCollection for
    // integration on the reference cell.
    // We will need to assemble the expansion matrices for each of the finite
    // elements we deal with, i.e. the matrices F_k,j. We have to do that for
    // each of the finite elements in use. To that end we need a quadrature
    // rule. As a default, we use the same quadrature formula for each finite
    // element, namely one that is obtained by iterating a 4-point Gauss formula
    // as many times as the maximal exponent we use for the term exp(ikx). Since
    // the first mode corresponds to k = 0, the maximal wave number is k =
    // n_modes - 1.
    const QGauss<1>      base_quadrature(4);
    const QIterated<dim> quadrature(base_quadrature, n_modes - 1);
    const QSorted<dim>   quadrature_sorted(quadrature);

    hp::QCollection<dim> expansion_q_collection;
    for (unsigned int i = 0; i < dof_handler.get_fe_collection().size(); ++i)
      expansion_q_collection.push_back(quadrature_sorted);

    // The FESeries::Fourier class' constructor first parameter $n_modes$
    // defines the number of coefficients in 1D with the total number of
    // coefficients being
    // $(n_modes)^dim$.
    FESeries::Fourier<dim, spacedim> fe_series(n_modes,
                                               dof_handler.get_fe_collection(),
                                               expansion_q_collection);

    fourier_coefficient_decay(fe_series,
                              dof_handler,
                              solution,
                              smoothness_indicators,
                              regression_strategy,
                              only_flagged_cells);
  }
} // namespace SmoothnessEstimator


// explicit instantiations
#include "smoothness_estimator.inst"

DEAL_II_NAMESPACE_CLOSE
