// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2017 - 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#ifndef dealii_optimization_rol_vector_adaptor_h
#define dealii_optimization_rol_vector_adaptor_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_TRILINOS_WITH_ROL
#  include <deal.II/base/exceptions.h>
#  include <deal.II/base/index_set.h>

#  include <deal.II/lac/vector.h>

#  include <ROL_Vector.hpp>

#  include <limits>
#  include <type_traits>


DEAL_II_NAMESPACE_OPEN

/**
 * A namespace that provides an interface to the
 * <a href="https://trilinos.org/docs/dev/packages/rol/doc/html/index.html">
 * Rapid Optimization Library</a> (ROL), a Trilinos package.
 */
namespace Rol
{
  /**
   * An adaptor that provides the implementation of the ROL::Vector interface
   * for vectors of type <tt>VectorType</tt>.
   *
   * This class supports vectors that satisfy the following requirements:
   *
   * The <tt>VectorType</tt> should contain the following types.
   * ```
   * VectorType::size_type;  // The type for size of the vector.
   * VectorType::value_type; // The type for elements stored in the vector.
   * VectorType::real_type;  // The type for real-valued numbers.
   * ```
   *
   * However, ROL doesn't distinguish VectorAdaptor::value_type from
   * VectorAdaptor::real_type. This is due to ROL's assumption that the
   * VectorAdaptor::value_type itself is a type for real-valued numbers.
   * Therefore, VectorAdaptor supports vectors whose real_type is
   * convertible to value_type in the sense that
   * <code>std::is_convertible_v<real_type, value_type></code> yields
   * <code>true</code>.
   *
   * The <tt>VectorType</tt> should contain the following methods.
   * @code
   * // Reinitialize the current vector using a given vector's
   * // size (and the parallel distribution) without copying
   * // the elements.
   * VectorType::reinit(const VectorType &, ...);
   *
   * // Globally add a given vector to the current.
   * VectorType::operator+=(const VectorType &);
   *
   * // Scale all elements by a given scalar.
   * VectorType::operator*=(const VectorType::value_type &);
   *
   * // Perform dot product with a given vector.
   * VectorType::operator*=(const VectorType &);
   *
   * // Scale all elements of the current vector and globally
   * // add a given vector to it.
   * VectorType::add(const VectorType::value_type, const VectorType &);
   *
   * // Copies the data of a given vector to the current.
   * // Resize the current vector if necessary (MPI safe).
   * VectorType::operation=(const VectorType &);
   *
   * // Return the global size of the current vector.
   * VectorType::size();
   *
   * // Return L^2 norm of the current vector
   * VectorType::l2_norm();
   *
   * // Iterator to the start of the (locally owned) element
   * // of the current vector.
   * VectorType::begin();
   *
   * // Iterator to the one past the last (locally owned)
   * // element of the current vector.
   * VectorType::end();
   *
   * // Compress the vector i.e., flush the buffers of the
   * // vector object if it has any.
   * VectorType::compress(VectorOperation::insert);
   * @endcode
   *
   * @note The current implementation in ROL doesn't support vector sizes above
   * the largest value of int type. Some of the vectors in deal.II (see
   * @ref Vector)
   * may not satisfy the above requirements.
   */
  template <typename VectorType>
  class VectorAdaptor : public ROL::Vector<typename VectorType::value_type>
  {
    /**
     * An alias for size type of <tt>VectorType</tt>.
     */
    using size_type = typename VectorType::size_type;

    /**
     * An alias for element type stored in the <tt>VectorType</tt>.
     */
    using value_type = typename VectorType::value_type;

    /**
     * An alias for real-valued numbers.
     */
    using real_type = typename VectorType::real_type;

    static_assert(std::is_convertible_v<real_type, value_type>,
                  "The real_type of the current VectorType is not "
                  "convertible to the value_type.");

  private:
    /**
     * ROL pointer to the underlying vector of type <tt>VectorType</tt>.
     */
    ROL::Ptr<VectorType> vector_ptr;

    /**
     * IndexSet of degrees of freedom we want to optimize.
     */
    IndexSet indices_to_optimize;

  public:
    /**
     * Constructor.
     */
    VectorAdaptor(const ROL::Ptr<VectorType> &vector_ptr);

    /**
     * Overload
     *
     * TODO: I do not know yet what indices_to_optimize needs to contain
     */
    VectorAdaptor(const ROL::Ptr<VectorType> &vector_ptr, const IndexSet &indices_to_optimize);

    /**
     * Return the ROL pointer to the wrapper vector, #vector_ptr.
     */
    ROL::Ptr<VectorType>
    getVector();

    /**
     * Return the ROL pointer to const vector.
     */
    ROL::Ptr<const VectorType>
    getVector() const;

    /**
     * Return the dimension (global vector size) of the wrapped vector.
     */
    int
    dimension() const;

    /**
     * Set the wrapper vector to a given ROL::Vector @p rol_vector by
     * overwriting its contents.
     *
     * If the current wrapper vector has ghost elements,
     * then <code> VectorType::operator=(const VectorType&) </code> should still
     * be allowed on it.
     */
    void
    set(const ROL::Vector<value_type> &rol_vector);

    /**
     * Perform addition.
     */
    void
    plus(const ROL::Vector<value_type> &rol_vector);

    /**
     * Scale the wrapper vector by @p alpha and add ROL::Vector @p rol_vector
     * to it.
     */
    void
    axpy(const value_type alpha, const ROL::Vector<value_type> &rol_vector);

    /**
     * Scale the wrapper vector.
     */
    void
    scale(const value_type alpha);

    /**
     * Return the dot product with a given ROL::Vector @p rol_vector.
     */
    value_type
    dot(const ROL::Vector<value_type> &rol_vector) const;

    /**
     * Return the $L^{2}$ norm of the wrapped vector.
     *
     * The returned type is of VectorAdaptor::value_type so as to maintain
     * consistency with ROL::Vector<VectorAdaptor::value_type> and
     * more importantly to not to create an overloaded version namely,
     * <code> VectorAdaptor::real_type norm() const; </code>
     * if real_type and value_type are not of the same type.
     */
    value_type
    norm() const;

    /**
     * Return a clone of the wrapped vector.
     */
    ROL::Ptr<ROL::Vector<value_type>>
    clone() const;

    /**
     * Create and return a ROL pointer to the basis vector corresponding to the
     * @p i ${}^{th}$ element of the wrapper vector.
     */
    ROL::Ptr<ROL::Vector<value_type>>
    basis(const int i) const;

    /**
     * Apply unary function @p f to all the elements of the wrapped vector.
     */
    void
    applyUnary(const ROL::Elementwise::UnaryFunction<value_type> &f);

    /**
     * Apply binary function @p f along with ROL::Vector @p rol_vector to all
     * the elements of the wrapped vector.
     */
    void
    applyBinary(const ROL::Elementwise::BinaryFunction<value_type> &f,
                const ROL::Vector<value_type>                      &rol_vector);

    /**
     * Return the accumulated value on applying reduction operation @p r on
     * all the elements of the wrapped vector.
     */
    value_type
    reduce(const ROL::Elementwise::ReductionOp<value_type> &r) const;

    /**
     * Print the wrapped vector to the output stream @p outStream.
     */
    void
    print(std::ostream &outStream) const;
  };


  /*------------------------------member definitions--------------------------*/
#  ifndef DOXYGEN


  template <typename VectorType>
  VectorAdaptor<VectorType>::VectorAdaptor(
    const ROL::Ptr<VectorType> &vector_ptr)
    : vector_ptr(vector_ptr)
    , indices_to_optimize(vector_ptr->locally_owned_elements())
  {}



  template <typename VectorType>
  VectorAdaptor<VectorType>::VectorAdaptor(
    const ROL::Ptr<VectorType> &vector_ptr,
    const IndexSet& indices_to_optimize)
    : vector_ptr(vector_ptr)
    , indices_to_optimize(indices_to_optimize)
  {
    Assert(indices_to_optimize.is_subset_of(vector_ptr->locally_owned_elements()),
           ExcMessage("Provided IndexSet needs to be a subset of locally owned indices."));
  }



  template <typename VectorType>
  ROL::Ptr<VectorType>
  VectorAdaptor<VectorType>::getVector()
  {
    return vector_ptr;
  }



  template <typename VectorType>
  ROL::Ptr<const VectorType>
  VectorAdaptor<VectorType>::getVector() const
  {
    return vector_ptr;
  }



  template <typename VectorType>
  void
  VectorAdaptor<VectorType>::set(const ROL::Vector<value_type> &rol_vector)
  {
    const VectorAdaptor &vector_adaptor =
      dynamic_cast<const VectorAdaptor &>(rol_vector);

    *vector_ptr = *vector_adaptor.getVector();
    indices_to_optimize = vector_adaptor.indices_to_optimize;
  }



  template <typename VectorType>
  void
  VectorAdaptor<VectorType>::plus(const ROL::Vector<value_type> &rol_vector)
  {
    Assert(this->dimension() == rol_vector.dimension(),
           ExcDimensionMismatch(this->dimension(), rol_vector.dimension()));

    const VectorAdaptor &vector_adaptor =
      dynamic_cast<const VectorAdaptor &>(rol_vector);

    for (auto i : indices_to_optimize)
      (*vector_ptr)[i] += (*vector_adaptor.getVector())[i];
  }



  template <typename VectorType>
  void
  VectorAdaptor<VectorType>::axpy(const value_type               alpha,
                                  const ROL::Vector<value_type> &rol_vector)
  {
    Assert(this->dimension() == rol_vector.dimension(),
           ExcDimensionMismatch(this->dimension(), rol_vector.dimension()));

    const VectorAdaptor &vector_adaptor =
      dynamic_cast<const VectorAdaptor &>(rol_vector);

    for (auto i : indices_to_optimize)
      (*vector_ptr)[i] += alpha * (*vector_adaptor.getVector())[i];
  }



  template <typename VectorType>
  int
  VectorAdaptor<VectorType>::dimension() const
  {
    const types::global_dof_index n_elements = indices_to_optimize.n_elements();

    Assert(n_elements < std::numeric_limits<int>::max(),
           ExcMessage("The number of elements to optimize is greater than the "
                      "largest value of type int."));

    return static_cast<int>(n_elements);
  }



  template <typename VectorType>
  void
  VectorAdaptor<VectorType>::scale(const value_type alpha)
  {
    for (auto i : indices_to_optimize)
      (*vector_ptr)[i] *= alpha;
  }



  template <typename VectorType>
  typename VectorType::value_type
  VectorAdaptor<VectorType>::dot(
    const ROL::Vector<value_type> &rol_vector) const
  {
    Assert(this->dimension() == rol_vector.dimension(),
           ExcDimensionMismatch(this->dimension(), rol_vector.dimension()));

    const VectorAdaptor &vector_adaptor =
      dynamic_cast<const VectorAdaptor &>(rol_vector);

    value_type dot(0);
    for (auto i : indices_to_optimize)
      dot += (*vector_ptr)[i] * (*vector_adaptor.getVector())[i];

    return dot;
  }



  template <typename VectorType>
  typename VectorType::value_type
  VectorAdaptor<VectorType>::norm() const
  {
    return std::sqrt(this->dot(*this));
  }



  template <typename VectorType>
  ROL::Ptr<ROL::Vector<typename VectorType::value_type>>
  VectorAdaptor<VectorType>::clone() const
  {
    return ROL::makePtr<VectorAdaptor>(ROL::makePtr<VectorType>(*vector_ptr), indices_to_optimize);
  }



  template <typename VectorType>
  ROL::Ptr<ROL::Vector<typename VectorType::value_type>>
  VectorAdaptor<VectorType>::basis(const int i) const
  {
    // Create empty vector like dealii vector.
    ROL::Ptr<VectorType> vec = ROL::makePtr<VectorType>();
    vec->reinit(*vector_ptr, false);

    // global dof index corresponding to i-th basis in optimization
    const types::global_dof_index global_dof_index = indices_to_optimize.nth_index_in_set(i);

    if (indices_to_optimize.is_element(global_dof_index))
      (*vec)[global_dof_index] = 1.;

    if (vec->has_ghost_elements())
      vec->update_ghost_values();
    else
      vec->compress(VectorOperation::insert);

    return ROL::makePtr<VectorAdaptor>(vec, indices_to_optimize);
  }



  template <typename VectorType>
  void
  VectorAdaptor<VectorType>::applyUnary(
    const ROL::Elementwise::UnaryFunction<value_type> &f)
  {
    for (value_type &entry : *vector_ptr)
      entry = f.apply(entry);

    if (vector_ptr->has_ghost_elements())
      vector_ptr->update_ghost_values();
    else
      vector_ptr->compress(VectorOperation::insert);
  }



  template <typename VectorType>
  void
  VectorAdaptor<VectorType>::applyBinary(
    const ROL::Elementwise::BinaryFunction<value_type> &f,
    const ROL::Vector<value_type>                      &rol_vector)
  {
    Assert(this->dimension() == rol_vector.dimension(),
           ExcDimensionMismatch(this->dimension(), rol_vector.dimension()));

    const VectorAdaptor &vector_adaptor =
      dynamic_cast<const VectorAdaptor &>(rol_vector);

    const VectorType &given_rol_vector = *(vector_adaptor.getVector());

    const typename VectorType::iterator       vend   = vector_ptr->end();
    const typename VectorType::const_iterator rolend = given_rol_vector.end();

    typename VectorType::const_iterator r_iterator = given_rol_vector.begin();
    for (typename VectorType::iterator l_iterator = vector_ptr->begin();
         l_iterator != vend && r_iterator != rolend;
         l_iterator++, r_iterator++)
      *l_iterator = f.apply(*l_iterator, *r_iterator);

    if (vector_ptr->has_ghost_elements())
      vector_ptr->update_ghost_values();
    else
      vector_ptr->compress(VectorOperation::insert);
  }



  template <typename VectorType>
  typename VectorType::value_type
  VectorAdaptor<VectorType>::reduce(
    const ROL::Elementwise::ReductionOp<value_type> &r) const
  {
    value_type result = r.initialValue();

    for (value_type &entry : *vector_ptr)
      r.reduce(entry, result);
    // Parallel reduction among processes is handled internally.

    return result;
  }



  template <typename VectorType>
  void
  VectorAdaptor<VectorType>::print(std::ostream &outStream) const
  {
    vector_ptr->print(outStream);
  }


#  endif // DOXYGEN


} // namespace Rol


DEAL_II_NAMESPACE_CLOSE


#else

// Make sure the scripts that create the C++20 module input files have
// something to latch on if the preprocessor #ifdef above would
// otherwise lead to an empty content of the file.
DEAL_II_NAMESPACE_OPEN
DEAL_II_NAMESPACE_CLOSE

#endif // DEAL_II_TRILINOS_WITH_ROL

#endif // dealii_optimization_rol_vector_adaptor_h
