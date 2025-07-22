// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2025 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// anisotropic refinement

// #include <deal.II/base/geometry_info.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>

#include "../tests.h"



void
test(const double extension = 0.)
{
  constexpr int dim = 2;

  // generate simplex out of quad cells
  Triangulation<dim> triangulation;
  {
    const Point<dim> p1(-1., 1.);
    const Point<dim> p2(0., 0.);
    const Point<dim> p3(0., 1. + extension);
    GridGenerator::simplex(triangulation, {p1, p2, p3});
  }

  GridOut       gridout;
  std::ofstream ofile_before("grid-before" + Utilities::to_string(extension) +
                             ".vtk");
  gridout.write_vtk(triangulation, ofile_before);

  // perform anisotropic refinement globally
  for (const auto &cell : triangulation.active_cell_iterators())
    cell->set_refine_flag(RefinementCase<dim>::cut_y);
  triangulation.execute_coarsening_and_refinement();

  // log results
  // GridOut gridout;
  gridout.write_vtk(triangulation, deallog.get_file_stream());

#if true
  std::ofstream ofile("grid" + Utilities::to_string(extension) + ".vtk");
  gridout.write_vtk(triangulation, ofile);
#endif

  deallog << "OK" << std::endl;
}


int
main(int argc, char *argv[])
{
  initlog();

  test(0.);
  test(1.);
}
