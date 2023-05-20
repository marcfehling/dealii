// ---------------------------------------------------------------------
//
// Copyright (C) 2000 - 2021 by the deal.II authors
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


#include <deal.II/fe/fe_tools.templates.h>

DEAL_II_NAMESPACE_OPEN


namespace FETools
{
  template <int dim, int spacedim>
  std::set<types::fe_index>
  find_common_fes(const hp::FECollection<dim, spacedim> &fe_collection,
                  const std::set<types::fe_index> &      fe_indices,
                  const unsigned int                     codim)
  {
#ifdef DEBUG
    // Validate user inputs.
    Assert(codim <= dim, ExcImpossibleInDim(dim));
    Assert(fe_collection.size() > 0, ExcEmptyObject());
    for (const auto &fe : fe_indices)
      AssertIndexRange(fe, fe_collection.size());
#endif

    // Check if any element of the fe_collection is able to dominate all
    // elements of @p fe_indices. If one was found, we add it to the set of
    // dominating elements.
    std::set<types::fe_index> dominating_fes;
    for (types::fe_index current_fe = 0; current_fe < fe_collection.size();
         ++current_fe)
      {
        // Check if current_fe can dominate all elements in @p fe_indices.
        FiniteElementDomination::Domination domination =
          FiniteElementDomination::no_requirements;
        for (const auto &other_fe : fe_indices)
          domination =
            domination & fe_collection[current_fe].compare_for_domination(
                           fe_collection[other_fe], codim);

        // If current_fe dominates, add it to the set.
        if ((domination == FiniteElementDomination::this_element_dominates) ||
            (domination == FiniteElementDomination::either_element_can_dominate
             /*covers cases like {Q2,Q3,Q1,Q1} with fe_indices={2,3}*/))
          dominating_fes.insert(current_fe);
      }
    return dominating_fes;
  }



  template <int dim, int spacedim>
  std::set<types::fe_index>
  find_enclosing_fes(const hp::FECollection<dim, spacedim> &fe_collection,
                     const std::set<types::fe_index> &      fe_indices,
                     const unsigned int                     codim)
  {
#ifdef DEBUG
    // Validate user inputs.
    Assert(codim <= dim, ExcImpossibleInDim(dim));
    Assert(fe_collection.size() > 0, ExcEmptyObject());
    for (const auto &fe : fe_indices)
      AssertIndexRange(fe, fe_collection.size());
#endif

    // Check if any element of the fe_collection is dominated by all
    // elements of @p fe_indices. If one was found, we add it to the set of
    // dominated elements.
    std::set<types::fe_index> dominated_fes;
    for (types::fe_index current_fe = 0; current_fe < fe_collection.size();
         ++current_fe)
      {
        // Check if current_fe is dominated by all other elements in @p fe_indices.
        FiniteElementDomination::Domination domination =
          FiniteElementDomination::no_requirements;
        for (const auto &other_fe : fe_indices)
          domination =
            domination & fe_collection[current_fe].compare_for_domination(
                           fe_collection[other_fe], codim);

        // If current_fe is dominated, add it to the set.
        if ((domination == FiniteElementDomination::other_element_dominates) ||
            (domination == FiniteElementDomination::either_element_can_dominate
             /*covers cases like {Q2,Q3,Q1,Q1} with fe_indices={2,3}*/))
          dominated_fes.insert(current_fe);
      }
    return dominated_fes;
  }



  template <int dim, int spacedim>
  types::fe_index
  find_dominating_fe(const hp::FECollection<dim, spacedim> &fe_collection,
                     const std::set<types::fe_index> &      fe_indices,
                     const unsigned int                     codim)
  {
    // If the set of elements contains only a single element,
    // then this very element is considered to be the dominating one.
    if (fe_indices.size() == 1)
      return *fe_indices.begin();

#ifdef DEBUG
    // Validate user inputs.
    Assert(codim <= dim, ExcImpossibleInDim(dim));
    Assert(fe_collection.size() > 0, ExcEmptyObject());
    for (const auto &fe : fe_indices)
      AssertIndexRange(fe, fe_collection.size());
#endif

    // There may also be others, in which case we'll check if any of these
    // elements is able to dominate all others. If one was found, we stop
    // looking further and return the dominating element.
    for (const auto &current_fe : fe_indices)
      {
        // Check if current_fe can dominate all elements in @p fe_indices.
        FiniteElementDomination::Domination domination =
          FiniteElementDomination::no_requirements;
        for (const auto &other_fe : fe_indices)
          if (current_fe != other_fe)
            domination =
              domination & fe_collection[current_fe].compare_for_domination(
                             fe_collection[other_fe], codim);

        // If current_fe dominates, return its index.
        if ((domination == FiniteElementDomination::this_element_dominates) ||
            (domination == FiniteElementDomination::either_element_can_dominate
             /*covers cases like {Q2,Q3,Q1,Q1} with fe_indices={2,3}*/))
          return current_fe;
      }

    // If we couldn't find the dominating object, return an invalid one.
    return numbers::invalid_fe_index;
  }



  template <int dim, int spacedim>
  types::fe_index
  find_dominated_fe(const hp::FECollection<dim, spacedim> &fe_collection,
                    const std::set<types::fe_index> &      fe_indices,
                    const unsigned int                     codim)
  {
    // If the set of elements contains only a single element,
    // then this very element is considered to be the dominated one.
    if (fe_indices.size() == 1)
      return *fe_indices.begin();

#ifdef DEBUG
    // Validate user inputs.
    Assert(codim <= dim, ExcImpossibleInDim(dim));
    Assert(fe_collection.size() > 0, ExcEmptyObject());
    for (const auto &fe : fe_indices)
      AssertIndexRange(fe, fe_collection.size());
#endif

    // There may also be others, in which case we'll check if any of these
    // elements is dominated by all others. If one was found, we stop
    // looking further and return the dominated element.
    for (const auto &current_fe : fe_indices)
      {
        // Check if current_fe is dominated by all other elements in @p fe_indices.
        FiniteElementDomination::Domination domination =
          FiniteElementDomination::no_requirements;
        for (const auto &other_fe : fe_indices)
          if (current_fe != other_fe)
            domination =
              domination & fe_collection[current_fe].compare_for_domination(
                             fe_collection[other_fe], codim);

        // If current_fe is dominated, return its index.
        if ((domination == FiniteElementDomination::other_element_dominates) ||
            (domination == FiniteElementDomination::either_element_can_dominate
             /*covers cases like {Q2,Q3,Q1,Q1} with fe_indices={2,3}*/))
          return current_fe;
      }

    // If we couldn't find the dominated object, return an invalid one.
    return numbers::invalid_fe_index;
  }



  template <int dim, int spacedim>
  types::fe_index
  find_dominating_fe_extended(
    const hp::FECollection<dim, spacedim> &fe_collection,
    const std::set<types::fe_index> &      fe_indices,
    const unsigned int                     codim)
  {
    types::fe_index fe_index =
      find_dominating_fe(fe_collection, fe_indices, codim);

    if (fe_index == numbers::invalid_fe_index)
      {
        const std::set<types::fe_index> dominating_fes =
          find_common_fes(fe_collection, fe_indices, codim);
        fe_index = find_dominated_fe(fe_collection, dominating_fes, codim);
      }

    return fe_index;
  }



  template <int dim, int spacedim>
  types::fe_index
  find_dominated_fe_extended(
    const hp::FECollection<dim, spacedim> &fe_collection,
    const std::set<types::fe_index> &      fe_indices,
    const unsigned int                     codim)
  {
    types::fe_index fe_index =
      find_dominated_fe(fe_collection, fe_indices, codim);

    if (fe_index == numbers::invalid_fe_index)
      {
        const std::set<types::fe_index> dominated_fes =
          find_enclosing_fes(fe_collection, fe_indices, codim);
        fe_index = find_dominating_fe(fe_collection, dominated_fes, codim);
      }

    return fe_index;
  }
} // namespace FETools



/*-------------- Explicit Instantiations -------------------------------*/
#include "fe_tools.inst"

// these do not fit into the templates of the dimension in the inst file
namespace FETools
{
  // Specializations for FE_Q.
  template <>
  std::unique_ptr<FiniteElement<1, 1>>
  FEFactory<FE_Q<1, 1>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_Q<1>>(quad);
  }

  template <>
  std::unique_ptr<FiniteElement<2, 2>>
  FEFactory<FE_Q<2, 2>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_Q<2>>(quad);
  }

  template <>
  std::unique_ptr<FiniteElement<3, 3>>
  FEFactory<FE_Q<3, 3>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_Q<3>>(quad);
  }

  // Specializations for FE_Q_DG0.
  template <>
  std::unique_ptr<FiniteElement<1, 1>>
  FEFactory<FE_Q_DG0<1, 1>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_Q_DG0<1>>(quad);
  }

  template <>
  std::unique_ptr<FiniteElement<2, 2>>
  FEFactory<FE_Q_DG0<2, 2>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_Q_DG0<2>>(quad);
  }

  template <>
  std::unique_ptr<FiniteElement<3, 3>>
  FEFactory<FE_Q_DG0<3, 3>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_Q_DG0<3>>(quad);
  }

  // Specializations for FE_Q_Bubbles.
  template <>
  std::unique_ptr<FiniteElement<1, 1>>
  FEFactory<FE_Q_Bubbles<1, 1>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_Q_Bubbles<1>>(quad);
  }

  template <>
  std::unique_ptr<FiniteElement<2, 2>>
  FEFactory<FE_Q_Bubbles<2, 2>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_Q_Bubbles<2>>(quad);
  }

  template <>
  std::unique_ptr<FiniteElement<3, 3>>
  FEFactory<FE_Q_Bubbles<3, 3>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_Q_Bubbles<3>>(quad);
  }

  // Specializations for FE_DGQArbitraryNodes.
  template <>
  std::unique_ptr<FiniteElement<1, 1>>
  FEFactory<FE_DGQ<1>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_DGQArbitraryNodes<1>>(quad);
  }

  template <>
  std::unique_ptr<FiniteElement<1, 2>>
  FEFactory<FE_DGQ<1, 2>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_DGQArbitraryNodes<1, 2>>(quad);
  }

  template <>
  std::unique_ptr<FiniteElement<1, 3>>
  FEFactory<FE_DGQ<1, 3>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_DGQArbitraryNodes<1, 3>>(quad);
  }

  template <>
  std::unique_ptr<FiniteElement<2, 2>>
  FEFactory<FE_DGQ<2>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_DGQArbitraryNodes<2>>(quad);
  }

  template <>
  std::unique_ptr<FiniteElement<2, 3>>
  FEFactory<FE_DGQ<2, 3>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_DGQArbitraryNodes<2, 3>>(quad);
  }

  template <>
  std::unique_ptr<FiniteElement<3, 3>>
  FEFactory<FE_DGQ<3>>::get(const Quadrature<1> &quad) const
  {
    return std::make_unique<FE_DGQArbitraryNodes<3>>(quad);
  }

  template std::vector<unsigned int>
  hierarchic_to_lexicographic_numbering<0>(unsigned int);

  template std::vector<unsigned int>
  lexicographic_to_hierarchic_numbering<0>(unsigned int);
} // namespace FETools

DEAL_II_NAMESPACE_CLOSE
