// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2003 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------


#include <deal.II/base/mg_level_object.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/trilinos_vector.h>

#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>

#include <algorithm>

#include "../tests.h"



template <int dim, int spacedim>
void
reinit_vector(const dealii::DoFHandler<dim, spacedim>      &mg_dof,
              MGLevelObject<TrilinosWrappers::MPI::Vector> &v)
{
  const dealii::parallel::distributed::Triangulation<dim, spacedim> *tria =
    (dynamic_cast<const parallel::distributed::Triangulation<dim, spacedim> *>(
      &mg_dof.get_triangulation()));
  AssertThrow(
    tria != nullptr,
    ExcMessage(
      "multigrid with Trilinos vectors only works with distributed Triangulation!"));

  for (unsigned int level = v.min_level(); level <= v.max_level(); ++level)
    {
      v[level].reinit(mg_dof.locally_owned_mg_dofs(level),
                      tria->get_mpi_communicator());
    }
}


template <int dim>
void
check_simple(const FiniteElement<dim> &fe)
{
  deallog << fe.get_name() << std::endl;

  parallel::distributed::Triangulation<dim> tr(
    MPI_COMM_WORLD,
    Triangulation<dim>::limit_level_difference_at_vertices,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);

  GridGenerator::hyper_cube(tr);
  tr.refine_global(2);

  DoFHandler<dim> mgdof(tr);
  mgdof.distribute_dofs(fe);
  mgdof.distribute_mg_dofs();

  MGTransferPrebuilt<TrilinosWrappers::MPI::Vector> transfer;
  transfer.build(mgdof);

  MGLevelObject<TrilinosWrappers::MPI::Vector> u(0, tr.n_levels() - 1);
  reinit_vector(mgdof, u);
  // First prolongate the constant
  // vector.  For Lagrange elements,
  // the values are just the number
  // of degrees of freedom.
  u[0] = 1.;
  transfer.prolongate(1, u[1], u[0]);
  transfer.prolongate(2, u[2], u[1]);
  deallog << "u0\t" << (u[0] * u[0] + .5) << std::endl
          << "u1\t" << (u[1] * u[1] + .5) << std::endl
          << "u2\t" << (u[2] * u[2] + .5) << std::endl;
  // Now restrict the same vectors.
  u[1] = 0.;
  u[0] = 0.;
  transfer.restrict_and_add(2, u[1], u[2]);
  transfer.restrict_and_add(1, u[0], u[1]);
  deallog << "u1\t" << (u[1] * u[1] + .5) << std::endl
          << "u0\t" << (u[0] * u[0] + .5) << std::endl;

  // Now the same for a non-constant
  // vector
  for (unsigned int i = 0; i < u[0].size(); ++i)
    u[0](i) = i;
  transfer.prolongate(1, u[1], u[0]);
  transfer.prolongate(2, u[2], u[1]);
  deallog << "u0\t" << (u[0] * u[0] + .5) << std::endl
          << "u1\t" << (u[1] * u[1] + .5) << std::endl
          << "u2\t" << (u[2] * u[2] + .5) << std::endl;
  // Now restrict the same vectors.
  u[1] = 0.;
  u[0] = 0.;
  transfer.restrict_and_add(2, u[1], u[2]);
  transfer.restrict_and_add(1, u[0], u[1]);
  deallog << "u1\t" << (u[1] * u[1] + .5) << std::endl
          << "u0\t" << (u[0] * u[0] + .5) << std::endl;

  // Fill a global vector by counting
  // from one up
  TrilinosWrappers::MPI::Vector v;
  v.reinit(mgdof.locally_owned_dofs(), MPI_COMM_WORLD);
  for (unsigned int i = 0; i < v.size(); ++i)
    v(i) = i + 1;

  transfer.copy_to_mg(mgdof, u, v);
  for (unsigned int i = 0; i < u[2].size(); ++i)
    deallog << ' ' << (int)u[2](i);
  deallog << std::endl;

  // Now do the opposite: fill a
  // multigrid vector counting the
  // dofs and see where the numbers go
  v = 0.;
  for (unsigned int i = 0; i < u[2].size(); ++i)
    u[2](i) = i + 1;
  transfer.copy_from_mg(mgdof, v, u);
  for (unsigned int i = 0; i < v.size(); ++i)
    deallog << ' ' << (int)v(i);
  deallog << std::endl;
  v.equ(-1., v);
  transfer.copy_from_mg_add(mgdof, v, u);
  deallog << "diff " << v.l2_norm() << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  initlog();
  deallog << std::setprecision(10);

  check_simple(FE_DGP<2>(0));
  check_simple(FE_DGP<2>(1));
  check_simple(FE_DGQ<2>(1));
  check_simple(FE_DGQ<2>(2));
  check_simple(FE_Q<2>(1));
  check_simple(FE_Q<2>(2));
  check_simple(FESystem<2>(FE_DGQ<2>(1), 2));
  check_simple(FESystem<2>(FE_DGP<2>(1), 2, FE_DGQ<2>(1), 3));

  check_simple(FE_RaviartThomasNodal<2>(1));
  check_simple(FESystem<2>(FE_RaviartThomas<2>(1), 1, FE_DGQ<2>(0), 2));

  // TODO: Reenable after implementing renumber_dofs in 3D

  // check_simple (FE_DGQ<3>(1));
  // check_simple (FE_Q<3>(2));
}
