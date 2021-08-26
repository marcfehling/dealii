// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
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



// Create a 2x2x1 grid with FE_Q elements with degrees 1 - 4 and 2 - 5 assigned
// in two respective scenarios.
// Verify that we do not unify dofs on lines in 3D.
// On each of the four lines on the interface between the Q2 and Q4 element, the
// central dofs are identical and will be treated with constraints.
//
// If the dominating element in the collection of finite elements on the central
// line does not have a central dof, the dof duplicates on the elements Q2 and
// Q4 will not be recognized with constraints.
// This is the reason why there is one identity constraint less in scenario 1
// than in scenario 2 -- the dominating Q1 element has no dofs on lines.
// This is a bug.
//
// Scenario 1:    Scenario 2:
// +----+----+    +----+----+
// | Q3 | Q4 |    | Q4 | Q5 |
// +----+----+    +----+----+
// | Q1 | Q2 |    | Q2 | Q3 |
// +----+----+    +----+----+

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/affine_constraints.h>

#include <array>

#include "../tests.h"


template <int dim>
void
test(std::array<unsigned int, 4> fe_degrees)
{
  Triangulation<dim> tria;
  {
    std::vector<unsigned int> repetitions(dim);
    Point<dim>                bottom_left, top_right;
    for (unsigned int d = 0; d < dim; ++d)
      if (d < 2)
        {
          repetitions[d] = 2;
          bottom_left[d] = -1.;
          top_right[d]   = 1.;
        }
      else
        {
          repetitions[d] = 1;
          bottom_left[d] = 0.;
          top_right[d]   = 1.;
        }
    GridGenerator::subdivided_hyper_rectangle(tria,
                                              repetitions,
                                              bottom_left,
                                              top_right);
  }

  hp::FECollection<dim> fe;
  for (const auto d : fe_degrees)
    fe.push_back(FE_Q<dim>(d));

  DoFHandler<dim> dh(tria);
  {
    unsigned int i = 0;
    for (const auto &cell : dh.active_cell_iterators())
      cell->set_active_fe_index(i++);
    Assert(i == 4, ExcInternalError());
  }
  dh.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dh, constraints);
  constraints.close();

  deallog << "Total constraints:          " << constraints.n_constraints()
          << std::endl
          << "  Inhomogenous constraints: " << constraints.n_inhomogeneities()
          << std::endl
          << "  Identity constraints:     " << constraints.n_identities()
          << std::endl;


  std::vector<types::global_dof_index> dof_indices;
  for (const auto &cell : dh.active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() == false)
        {
          // name of fe on this cell
          std::cout << cell->get_fe().get_name() << std::endl;

          // name of all fes on this face
          for (const auto &fe_index : face->get_active_fe_indices())
              std::cout << " " << dh.get_fe(fe_index).get_name();
          std::cout << std::endl;

          // all dofs on this face that belong to this cell
          dof_indices.resize(cell->get_fe().n_dofs_per_face());
          face->get_dof_indices(dof_indices, cell->active_fe_index());

          for (const auto& dof_index : dof_indices)
            std::cout << " " << dof_index;
          std::cout << std::endl;
          for (const auto& dof_index : dof_indices)
            std::cout << " " << constraints.is_constrained(dof_index);
          std::cout << std::endl;
        }
}


int
main()
{
  initlog();

//  deallog << "FE degrees: 1 - 4" << std::endl;
//  deallog.push("2d");
//  test<2>({{1, 2, 3, 4}});
//  deallog.pop();
  deallog.push("3d");
  test<3>({{1, 2, 3, 4}});
  deallog.pop();

//  deallog << "FE degrees: 2 - 5" << std::endl;
//  deallog.push("2d");
//  test<2>({{2, 3, 4, 5}});
//  deallog.pop();
//  deallog.push("3d");
//  test<3>({{2, 3, 4, 5}});
//  deallog.pop();
}
