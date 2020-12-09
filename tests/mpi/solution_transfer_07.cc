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


// Test case based on the one written by K. Bzowski

#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/vector.h>

#include "../tests.h"


template <int dim, int spacedim = dim>
void
test()
{
  MPI_Comm mpi_communicator = MPI_COMM_WORLD;

  parallel::distributed::Triangulation<dim, spacedim> tria(mpi_communicator);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);

  hp::FECollection<dim> fes{FE_Q<dim>(1), FE_Nothing<dim>()};
  //  fes.push_back(FE_Q<dim>(1));
  //  fes.push_back(FE_Nothing<dim>());

  DoFHandler<dim> dofh(tria);

  // Assign FE indices
  //  +---+---+
  //  | 0 | 0 |  FE_Q
  //  +---+---+
  //  | 1 | 1 |  FE_Nothing
  //  +---+---+
  for (const auto &cell : dofh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        if (cell->center()(1) > 0.5)
          cell->set_active_fe_index(0);
        else
          cell->set_active_fe_index(1);
      }

  dofh.distribute_dofs(fes);

  // Prepare FE solution for transfer
  IndexSet locally_owned_dofs = dofh.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dofh, locally_relevant_dofs);

  LinearAlgebraTrilinos::MPI::Vector completely_distributed_solution;
  LinearAlgebraTrilinos::MPI::Vector locally_relevant_solution;

  completely_distributed_solution.reinit(locally_owned_dofs, mpi_communicator);
  locally_relevant_solution.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator);

  completely_distributed_solution = 1.0;
  locally_relevant_solution       = completely_distributed_solution;

  // Set refine flags
  //  +---+---+
  //  | R | R |  FE_Q
  //  +---+---+
  //  |   |   |	 FE_Nothing
  //  +---+---+
  for (const auto &cell : dofh.active_cell_iterators())
    if (cell->is_locally_owned())
      if (cell->center()(1) > 0.5)
        cell->set_refine_flag();

  // Perform refinement
  parallel::distributed::SolutionTransfer<dim,
                                          LinearAlgebraTrilinos::MPI::Vector>
    soltrans(dofh);
  soltrans.prepare_for_coarsening_and_refinement(locally_relevant_solution);

  tria.execute_coarsening_and_refinement();
  dofh.distribute_dofs(fes);

  // Interpolate solution on new grid
  locally_owned_dofs = dofh.locally_owned_dofs();
  locally_relevant_dofs.clear();
  DoFTools::extract_locally_relevant_dofs(dofh, locally_relevant_dofs);

  completely_distributed_solution.reinit(locally_owned_dofs, mpi_communicator);
  locally_relevant_solution.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator);

  solution_trans.interpolate(completely_distributed_solution);
  locally_relevant_solution = completely_distributed_solution;
}


int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    log;

  deallog.push("2d");
  test<2>();
  deallog.pop();

  return 0;


//  IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
//  IndexSet locally_relevant_dofs;
//  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

//  LinearAlgebraTrilinos::MPI::Vector completely_distributed_solution;
//  LinearAlgebraTrilinos::MPI::Vector locally_relevant_solution;

//  completely_distributed_solution.reinit(locally_owned_dofs, mpi_communicator);
//  locally_relevant_solution.reinit(locally_owned_dofs,
//                                   locally_relevant_dofs,
//                                   mpi_communicator);

//  completely_distributed_solution = 1.0;

//  locally_relevant_solution = completely_distributed_solution;

//  Vector<double> FE_Type(triangulation.n_active_cells());
//  Vector<float>  subdomain(triangulation.n_active_cells());
//  int            i = 0;
//  for (auto &cell : dof_handler.active_cell_iterators())
//    {
//      if (cell->is_locally_owned())
//        {
//          FE_Type(i)   = cell->active_fe_index();
//          subdomain(i) = triangulation.locally_owned_subdomain();
//        }
//      else
//        {
//          FE_Type(i)   = -1;
//          subdomain(i) = -1;
//        }
//      i++;
//    }


//  /* Set refine flags:
//   * -----------
//   * |  R |  R |  FEQ
//   * -----------
//   * |    |    |	FE_Nothing
//   * -----------
//   */

//  for (auto &cell : dof_handler.active_cell_iterators())
//    {
//      if (cell->is_locally_owned())
//        {
//          auto center = cell->center();
//          if (center(1) > 0.5)
//            {
//              cell->set_refine_flag();
//            }
//        }
//    }

//  LA::MPI::Vector previous_locally_relevant_solution;
//  previous_locally_relevant_solution = locally_relevant_solution;

//  parallel::distributed::SolutionTransfer<2, LA::MPI::Vector, hp::DoFHandler<2>>
//    solution_trans(dof_handler);

//  triangulation.prepare_coarsening_and_refinement();
//  solution_trans.prepare_for_coarsening_and_refinement(
//    previous_locally_relevant_solution);

//  triangulation.execute_coarsening_and_refinement();

//  for (auto &cell : dof_handler.active_cell_iterators())
//    {
//      if (cell->is_locally_owned())
//        {
//          auto center = cell->center();
//          if (center(1) < 0.5)
//            {
//              cell->set_active_fe_index(1);
//            }
//          else
//            {
//              cell->set_active_fe_index(0);
//            }
//        }
//    }

//  dof_handler.distribute_dofs(fe_collection);

//  locally_owned_dofs = dof_handler.locally_owned_dofs();
//  locally_relevant_dofs.clear();
//  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
//  completely_distributed_solution.reinit(locally_owned_dofs, mpi_communicator);
//  locally_relevant_solution.reinit(locally_owned_dofs,
//                                   locally_relevant_dofs,
//                                   mpi_communicator);

//  solution_trans.interpolate(completely_distributed_solution);
//  locally_relevant_solution = completely_distributed_solution;

//  FE_Type.reinit(triangulation.n_active_cells());
//  subdomain.reinit(triangulation.n_active_cells());
//  i = 0;
//  for (auto &cell : dof_handler.active_cell_iterators())
//    {
//      if (cell->is_locally_owned())
//        {
//          FE_Type(i)   = cell->active_fe_index();
//          subdomain(i) = triangulation.locally_owned_subdomain();
//        }
//      else
//        {
//          FE_Type(i)   = -1;
//          subdomain(i) = -1;
//        }
//      i++;
//    }

//  // Save output
//  {
//    DataOut<2, hp::DoFHandler<2>> data_out;
//    data_out.attach_dof_handler(dof_handler);
//    data_out.add_data_vector(locally_relevant_solution, "Solution");
//    data_out.add_data_vector(FE_Type, "FE_Type");
//    data_out.add_data_vector(subdomain, "subdomain");
//    data_out.build_patches();

//    data_out.write_vtu_with_pvtu_record(
//      "./", "solution", 2, mpi_communicator, 2);
//  }
}
