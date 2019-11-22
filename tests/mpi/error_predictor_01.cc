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



// Test to check if error_indicatorsTransfer works in parallel with
// hp::DoFHandler. This tests is based on mpi/feindices_transfer.cc


#include <deal.II/distributed/error_predictor.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/hp/dof_handler.h>

#include <deal.II/lac/vector.h>

#include "../tests.h"


template <int dim>
void
test()
{
  const unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // ------ setup ------
  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);

  GridGenerator::subdivided_hyper_cube(tria, 2);
  tria.refine_global(1);
  deallog << "cells before: " << tria.n_global_active_cells() << std::endl;

  hp::DoFHandler<dim>   dh(tria);
  hp::FECollection<dim> fe_collection;

  // prepare FECollection with arbitrary number of entries
  const unsigned int max_degree = 3;
  for (unsigned int p = 1; p <= max_degree; ++p)
    fe_collection.push_back(FE_Q<dim>(p));

  typename hp::DoFHandler<dim, dim>::active_cell_iterator cell;
  unsigned int                                            i = 0;

  for (cell = dh.begin_active(); cell != dh.end(); ++cell)
    if (cell->is_locally_owned())
      {
        // set active fe index
        cell->set_active_fe_index(1);

        // set refinement/coarsening flags
        if (cell->id().to_string() == "0_1:0")
          cell->set_refine_flag();
        else if (cell->parent()->id().to_string() ==
                 ((dim == 2) ? "3_0:" : "7_0:"))
          cell->set_coarsen_flag();

        // set future fe indices
        if (cell->parent()->id().to_string() == "1_0:")
          cell->set_future_fe_index(2);
        else if (cell->parent()->id().to_string() == "2_0:")
          cell->set_future_fe_index(0);
      }


  // ----- prepare error indicators -----
  dh.distribute_dofs(fe_collection);

  Vector<float> error_indicators;
  error_indicators.reinit(tria.n_active_cells());
  for (unsigned int i = 0; i < error_indicators.size(); ++i)
    error_indicators(i) = 10.;

  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        deallog << " cell:" << cell->id().to_string()
                << " fe_deg:" << cell->get_fe().degree
                << " error:" << error_indicators[cell->active_cell_index()];

        if (cell->coarsen_flag_set())
          deallog << " coarsening";
        else if (cell->refine_flag_set())
          deallog << " refining";

        if (cell->future_fe_index_set())
          deallog << " future_fe_deg:"
                  << fe_collection[cell->future_fe_index()].degree;

        deallog << std::endl;
      }

  // ----- transfer -----
  parallel::distributed::ErrorPredictor<dim> predictor(dh);

  predictor.prepare_for_coarsening_and_refinement(error_indicators,
                                                  /*gamma_p=*/0.5,
                                                  /*gamma_h=*/1.,
                                                  /*gamma_n=*/1.);
  tria.execute_coarsening_and_refinement();
  deallog << "cells after: " << tria.n_global_active_cells() << std::endl;

  dh.distribute_dofs(fe_collection);

  Vector<float> predicted_errors;
  predicted_errors.reinit(tria.n_active_cells());
  predictor.unpack(predicted_errors);


  // ------ verify ------
  // check if all children adopted the correct id
  for (const auto &cell : dh.active_cell_iterators())
    if (cell->is_locally_owned())
      deallog << " cell:" << cell->id().to_string()
              << " predict:" << predicted_errors(cell->active_cell_index())
              << std::endl;

  // make sure no processor is hanging
  MPI_Barrier(MPI_COMM_WORLD);

  deallog << "OK" << std::endl;
}



int
main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll                    log;

  deallog.push("2d");
  test<2>();
  deallog.pop();
  deallog.push("3d");
  test<3>();
  deallog.pop();
}
