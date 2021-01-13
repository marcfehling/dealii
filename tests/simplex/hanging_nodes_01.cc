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



// Verify hanging node constraints on locally adapted simplex mesh

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/simplex/fe_lib.h>

//#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
//#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
//#include <deal.II/grid/tria_accessor.h>
//#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/simplex/grid_generator.h>

#include <fstream>
//#include <iostream>

#include "../tests.h"


template <int dim>
void
test()
{
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 1);

  tria.begin_active()->set_refine_flag();
  tria.execute_coarsening_and_refinement();

  GridOut grid_out;
#if true
  std::ofstream out("mesh.out.vtk");
  grid_out.write_vtk(tria, out);
#else
  grid_out.write_vtk(tria, deallog.get_file_stream());
#endif


  DoFHandler<dim> dofh(tria);
  //Simplex::FE_P<dim> fe(1);
  //dofh.distribute_dofs(fe);
  dofh.distribute_dofs(Simplex::FE_P<dim>(1));

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dofh, constraints);
  constraints.print(deallog.get_file_stream());

  deallog << "OK" << std::endl;
}


int
main()
{
  initlog();

  deallog.push("2d");
  test<2>();
  deallog.pop();
}
