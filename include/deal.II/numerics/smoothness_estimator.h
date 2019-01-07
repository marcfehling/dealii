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

#ifndef dealii_smoothness_estimator_h
#define dealii_smoothness_estimator_h


#include <deal.II/base/config.h>

#include <deal.II/numerics/vector_tools.h>

#include <functional>
#include <vector>


DEAL_II_NAMESPACE_OPEN


// forward declarations
#ifndef DOXYGEN
template <typename Number>
class Vector;

namespace FESeries
{
  template <int dim, int spacedim>
  class Fourier;
  template <int dim, int spacedim>
  class Legendre;
} // namespace FESeries

namespace hp
{
  template <int dim, int spacedim>
  class DoFHandler;
} // namespace hp
#endif


/**
 * A namespace for various smoothness estimation strategies for hp-adaptive FEM.
 *
 * Smoothness estimation is one strategy to decide whether a cell with a large
 * error estimate should undergo h- or p-refinement. Typical strategies decide
 * to increase the polynomial degree on a cell if the solution is particularly
 * smooth, whereas one would refine the mesh if the solution on the cell is
 * singular, has kinks in some derivative, or is otherwise not particularly
 * smooth. All of these strategies rely on a way to identify how "smooth" a
 * function is on a given cell.
 */
namespace SmoothnessEstimator
{
  /**
   * Estimate smoothness from decay of Legendre absolute values of coefficients
   * on the reference cell.
   *
   * In one dimension, the finite element solution on the reference element with
   * polynomial degree $p$ can be written as
   * @f[
   *    u_h(\hat x) = \sum_{j=0}^{p} a_j P_j(\hat x)
   * @f]
   * where $\{P_j(x)\}$ are Legendre polynomials. The decay of the coefficients
   * is estimated by performing the linear regression fit of
   * @f[
   *   |a_j| \sim c \, \exp(-\sigma j)
   * @f]
   * or, equivalently
   * @f[
   *   \ln |a_j| \sim C - \sigma j
   * @f]
   * for $j=0,..,p$. The rate of the decay $\sigma$ can be used to estimate the
   * smoothness. For example, one strategy to implement hp-refinement
   * criteria is to perform p-refinement if $\sigma>1$.
   *
   * Extension to higher dimension is done by performing the fit in each
   * coordinate direction separately and then taking the lowest value of
   * $\sigma$.
   *
   * For each input vector of degrees of freedom defined on a DoFHandler,
   * this function returns a vector with as many elements as there are cells
   * where each element contains $\exp(-\sigma)$, which is a so-called
   * analyticity (see references below).
   *
   * @param [in] fe_series FESeries::Legendre object to calculate coefficients.
   * This object needs to be initialized to have at least $p+1$ coefficients in
   * each direction, where $p$ is the maximum polynomial degree to be used.
   * @param [in] dof_hander An hp::DoFHandler
   * @param [in] solution A solution vector
   * @param [out] smoothness_indicators A vector for smoothness indicators
   * @param [in] coefficients_predicate A predicate to select Legendre
   * coefficients $a_j \;\; j=0\dots p$ for linear regression in each coordinate
   * direction. The user is responsible for updating the vector of `flags`
   * provided to this function. Note that its size is $p+1$, where $p$ is the
   * polynomial degree of the FE basis on a given element. The default
   * implementation will use all Legendre coefficients in each coordinate
   * direction, i.e. set all elements of the vector to `true`.
   * @param [in] smallest_abs_coefficient The smallest absolute value of the
   * coefficient to be used in linear regression in each coordinate direction.
   * Note that Legendre coefficients of some functions may have a repeating
   * pattern of zero coefficients (i.e. for functions that are locally symmetric
   * or antisymmetric about the midpoint of the element in any coordinate
   * direction). Thus this parameters allows to ingore small (in absolute value)
   * coefficients within the linear regression fit. In case there are less than
   * two non-zero coefficients for a coordinate direction, this direction will
   * be skipped. If all coefficients are zero, the returned value for this cell
   * will be zero (i.e. corresponding to the $\sigma=\infty$).
   * @param [in] only_flagged_cells Smoothness indicators are usually used to
   * decide whether to perform h- or p-adaptation. So in most cases, we only
   * need to calculate those indicators on cells flagged for refinement or
   * coarsening. This parameter controls whether this particular subset or all
   * cells will be considered. By default, only flagged cells will be
   * considered: Smoothness indicators will only be set on those vector entries
   * of flagged cells; the others will be set to zero.
   *
   * For more theoretical details see @cite mavriplis1994hp @cite houston2005hp
   * @cite eibner2007hp and for the application within the deal.II library
   * @cite davydov2017hp .
   *
   * @ingroup numerics
   * @author Denis Davydov, 2018
   */
  template <int dim, int spacedim, typename VectorType>
  void
  legendre_coefficient_decay(FESeries::Legendre<dim, spacedim> &  fe_series,
                             const hp::DoFHandler<dim, spacedim> &dof_handler,
                             const VectorType &                   solution,
                             Vector<float> &smoothness_indicators,
                             const std::function<void(std::vector<bool> &flags)>
                               &coefficients_predicate =
                                 [](std::vector<bool> &flags) -> void {
                               std::fill(flags.begin(), flags.end(), true);
                             },
                             const double smallest_abs_coefficient = 1e-10,
                             const bool   only_flagged_cells       = false);

  /**
   * Same as the function above, but with a default configuration for the
   * Legendre series expansion FESeries::Legendre.
   *
   * We use as many modes as the highest polynomial of all finite elements used
   * plus one, since we start with the first Legendre polynomial which is just a
   * constant. Further, the function uses Gaussian quadrature designed to yield
   * exact results for the highest order Legendre polynomial used.
   */
  template <int dim, int spacedim, typename VectorType>
  void
  legendre_coefficient_decay(const hp::DoFHandler<dim, spacedim> &dof_handler,
                             const VectorType &                   solution,
                             Vector<float> &smoothness_indicators,
                             const std::function<void(std::vector<bool> &flags)>
                               &coefficients_predicate =
                                 [](std::vector<bool> &flags) -> void {
                               std::fill(flags.begin(), flags.end(), true);
                             },
                             const double smallest_abs_coefficient = 1e-10,
                             const bool   only_flagged_cells       = false);

  /**
   * Estimate the smoothness of a solution based on the decay of coefficients
   * from a Fourier series expansion.
   *
   * From the definition, we can write our Fourier series expansion
   * $\hat U_{\bf k}$ as a matrix product
   * @f[
   *    \hat U_{\bf k}
   *    = {\cal F}_{{\bf k},j} u_j,
   * @f]
   * with $u_j$ the coefficients and ${\cal F}_{{\bf k},j}$ the transformation
   * matrix from the Fourier expansion of each shape function. We use the class
   * FESeries::Fourier to determine all coefficients $U_{\bf k}$.
   *
   * The next step is that we have to estimate how fast these coefficients
   * decay with $|{\bf k}|$. Thus, we perform a least-squares fit
   * @f[
   *    \min_{\alpha,\mu}
   *    \frac 12 \sum_{{\bf k}, |{\bf k}|\le N}
   *    \left( |\hat U_{\bf k}| - \alpha |{\bf k}|^{-\mu}\right)^2
   * @f]
   * with regression coefficients $\alpha$ and $\mu$. For simplification, we
   * apply a logarithm on our minimization problem
   * @f[
   *    \min_{\beta,\mu}
   *    Q(\beta,\mu) =
   *    \frac 12 \sum_{{\bf k}, |{\bf k}|\le N}
   *    \left( \ln |\hat U_{\bf k}| - \beta + \mu \ln |{\bf k}|\right)^2,
   * @f]
   * where $\beta=\ln \alpha$. This is now a problem for which the
   * optimality conditions $\frac{\partial Q}{\partial\beta}=0,
   * \frac{\partial Q}{\partial\mu}=0$, are linear in $\beta,\mu$. We can
   * write these conditions as follows:
   * @f[
   *    \left(\begin{array}{cc}
   *    \sum_{{\bf k}, |{\bf k}|\le N} 1 &
   *    \sum_{{\bf k}, |{\bf k}|\le N} \ln |{\bf k}|
   *    \\
   *    \sum_{{\bf k}, |{\bf k}|\le N} \ln |{\bf k}| &
   *    \sum_{{\bf k}, |{\bf k}|\le N} (\ln |{\bf k}|)^2
   *    \end{array}\right)
   *    \left(\begin{array}{c}
   *    \beta \\ -\mu
   *    \end{array}\right)
   *    =
   *    \left(\begin{array}{c}
   *    \sum_{{\bf k}, |{\bf k}|\le N} \ln |\hat U_{{\bf k}}|
   *    \\
   *    \sum_{{\bf k}, |{\bf k}|\le N} \ln |\hat U_{{\bf k}}| \ln |{\bf k}|
   *    \end{array}\right)
   * @f]
   * Solving for $\beta$ and $\mu$ is just a linear regression fit and to do
   * that we will use FESeries::linear_regression().
   *
   * While we are not particularly interested in the actual value of
   * $\beta$, the formula above gives us a means to calculate the value of
   * the exponent $\mu$ that we can then use to determine that
   * $\hat u(\hat{\bf x})$ is in $H^s(\hat K)$ with $s=\mu-\frac d2$. These
   * regularity estimates $s$ will suffice as our smoothness indicators and will
   * be calculated on each cell for any provided solution.
   *
   * @note An extensive demonstration of the use of these functions is provided
   * in step-27.
   *
   * The @p regression_strategy parameter determines which norm will be used
   * on the subset of coefficients $\mathbf{k}$ with the same absolute value
   * $|\mathbf{k}|$. Default is VectorTools::Linfty_norm for a maximum
   * approximation.
   *
   * For a provided solution vector defined on a DoFHandler, this function
   * returns a vector with as many elements as there are cells where each
   * element contains the estimated regularity $s$.
   *
   * An individual @p fe_series object can be supplied, which has to be
   * constructed with the same FECollection object as the @p dof_handler.
   *
   * Smoothness indicators are usually used to decide whether to perform h- or
   * p-adaptation. So in most cases, we only need to calculate those indicators
   * on cells flagged for refinement or coarsening. The parameter @p only_flagged_cells
   * controls whether this particular subset or all cells will be considered. By
   * default, only flagged cells will be considered: Smoothness indicators will
   * only be set on those vector entries of flagged cells; the others will be
   * set to zero.
   *
   * @ingroup numerics
   * @author Denis Davydov, 2016, Marc Fehling, 2018
   */
  template <int dim, int spacedim, typename VectorType>
  void
  fourier_coefficient_decay(
    FESeries::Fourier<dim, spacedim> &   fe_series,
    const hp::DoFHandler<dim, spacedim> &dof_handler,
    const VectorType &                   solution,
    Vector<float> &                      smoothness_indicators,
    const VectorTools::NormType regression_strategy = VectorTools::Linfty_norm,
    const bool                  only_flagged_cells  = false);

  /**
   * Same as the function above, but with a default configuration for the
   * Fourier series expansion FESeries::Fourier.
   *
   * We use as many modes as the highest polynomial degree of all finite
   * elements used plus one, and at least three modes. Further, the function
   * uses a 4-point Gaussian quarature iterated in each dimension by the maximal
   * wave number, which is the number of modes decresed by one since we start
   * with $k = 0$.
   */
  template <int dim, int spacedim, typename VectorType>
  void
  fourier_coefficient_decay(
    const hp::DoFHandler<dim, spacedim> &dof_handler,
    const VectorType &                   solution,
    Vector<float> &                      smoothness_indicators,
    const VectorTools::NormType regression_strategy = VectorTools::Linfty_norm,
    const bool                  only_flagged_cells  = false);
} // namespace SmoothnessEstimator


DEAL_II_NAMESPACE_CLOSE

#endif
