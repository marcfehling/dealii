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



// Check for continuity requirements in SolutionTransfer if p-adaptation
// has been performed.
// Transferring a solution from FE_Nothing to a different finite element
// and vice versa is only allowed if the FE_Nothing element has the
// FiniteElementDomination logic enabled.


#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools_project.h>

#include "../tests.h"



template <int dim>
void
test(const bool fe_nothing_dominates)
{
  const unsigned int n_cells = 2;

  // setup
  // +-----+-----+
  // | FEN | FEQ |
  // +-----+-----+
  Triangulation<dim>        tria;
  std::vector<unsigned int> rep(dim, 1);
  rep[0] = n_cells;
  Point<dim> p1, p2;
  for (unsigned int d = 0; d < dim; ++d)
    {
      p1[d] = 0;
      p2[d] = (d == 0) ? n_cells : 1;
    }
  GridGenerator::subdivided_hyper_rectangle(tria, rep, p1, p2);

  hp::FECollection<dim> fes;
  fes.push_back(FE_Q<dim>(1));
  fes.push_back(FE_Nothing<dim>(/*n_components=*/1, fe_nothing_dominates));

  DoFHandler<dim> dofh(tria);
  dofh.begin_active()->set_active_fe_index(1);
  dofh.distribute_dofs(fes);

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dofh, constraints);
  constraints.close();
  deallog << "pre-refinement constraints:" << std::endl;
  constraints.print(deallog.get_file_stream());

  // init constant solution with all constraints
  Vector<double> solution(dofh.n_dofs());
  VectorTools::project(dofh,
                       constraints,
                       hp::QCollection<dim>(QGauss<dim>(2), Quadrature<dim>(1)),
                       Functions::ConstantFunction<dim>(1.),
                       solution);

  // turn FE_Nothing cell into FE_Q cell, transfer solution across refinement
  dofh.begin_active()->set_future_fe_index(0);
  SolutionTransfer<dim> soltrans(dofh);
  soltrans.prepare_for_pure_refinement();
  tria.execute_coarsening_and_refinement();
  dofh.distribute_dofs(fes);

  // interpolate solution on new grid
  Vector<double> new_solution(dofh.n_dofs());
  try
    {
      // will work only if FE_Nothing dominates
      soltrans.refine_interpolate(solution, new_solution);
    }
  catch (const ExcMessage &)
    {
      deallog << "Interpolation from FE_Nothing to FE_Q failed." << std::endl;
      return;
    }
#ifndef DEBUG
  // output exception which is not triggered in release mode
  if (!fe_nothing_dominates)
    {
      deallog << "Interpolation from FE_Nothing to FE_Q failed." << std::endl;
      return;
    }
#endif

  // verify output
  deallog << "post-refinement dof values:" << std::endl;
  Vector<double> cell_values;
  for (const auto &cell : dofh.active_cell_iterators())
    {
      cell_values.reinit(cell->get_fe().n_dofs_per_cell());
      cell->get_dof_values(new_solution, cell_values);

      deallog << "cell " << cell->active_cell_index() << ":";
      for (const auto &value : cell_values)
        deallog << " " << value;
      deallog << std::endl;
    }

  deallog << "OK" << std::endl;
}


int
main()
{
  initlog();

  deal_II_exceptions::disable_abort_on_exception();

  deallog << std::boolalpha;
  for (const bool fe_nothing_dominates : {false, true})
    {
      deallog << "FE_Nothing dominates: " << fe_nothing_dominates << std::endl;
      deallog.push("1d");
      test<1>(fe_nothing_dominates);
      deallog.pop();
      deallog.push("2d");
      test<2>(fe_nothing_dominates);
      deallog.pop();
      deallog.push("3d");
      test<3>(fe_nothing_dominates);
      deallog.pop();
    }
}
