// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2015 - 2024 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// test running a multigrid solver on continuous FE_Q finite elements with
// MatrixFree data structures with parallel::fullydistributd::Triangulation
// (otherwise similar to tests/matrix_free/parallel_multigrid_mf.cc)

#include <deal.II/base/utilities.h>

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_description.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"


template <int dim,
          int fe_degree,
          int n_q_points_1d = fe_degree + 1,
          typename number   = double>
class LaplaceOperator : public EnableObserverPointer
{
public:
  using value_type = number;

  LaplaceOperator(){};

  void
  initialize(const Mapping<dim>                 &mapping,
             const DoFHandler<dim>              &dof_handler,
             const std::set<types::boundary_id> &dirichlet_boundaries,
             const unsigned int level = numbers::invalid_unsigned_int)
  {
    const QGauss<1>                                  quad(n_q_points_1d);
    typename MatrixFree<dim, number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme =
      MatrixFree<dim, number>::AdditionalData::none;
    addit_data.mg_level = level;

    // extract the constraints due to Dirichlet boundary conditions
    AffineConstraints<double>                           constraints;
    Functions::ZeroFunction<dim>                        zero;
    std::map<types::boundary_id, const Function<dim> *> functions;
    for (std::set<types::boundary_id>::const_iterator it =
           dirichlet_boundaries.begin();
         it != dirichlet_boundaries.end();
         ++it)
      functions[*it] = &zero;
    if (level == numbers::invalid_unsigned_int)
      VectorTools::interpolate_boundary_values(dof_handler,
                                               functions,
                                               constraints);
    else
      {
        std::vector<types::global_dof_index>    local_dofs;
        typename DoFHandler<dim>::cell_iterator cell = dof_handler.begin(level),
                                                endc = dof_handler.end(level);
        for (; cell != endc; ++cell)
          {
            if (dof_handler.get_triangulation().locally_owned_subdomain() !=
                  numbers::invalid_subdomain_id &&
                cell->level_subdomain_id() == numbers::artificial_subdomain_id)
              continue;
            const FiniteElement<dim> &fe = cell->get_fe();
            local_dofs.resize(fe.dofs_per_face);

            for (const unsigned int face_no : GeometryInfo<dim>::face_indices())
              if (cell->at_boundary(face_no) == true)
                {
                  const typename DoFHandler<dim>::face_iterator face =
                    cell->face(face_no);
                  const types::boundary_id bi = face->boundary_id();
                  if (functions.find(bi) != functions.end())
                    {
                      face->get_mg_dof_indices(level, local_dofs);
                      for (unsigned int i = 0; i < fe.dofs_per_face; ++i)
                        if (constraints.is_constrained(local_dofs[i]) == false)
                          constraints.constrain_dof_to_zero(local_dofs[i]);
                    }
                }
          }
      }
    constraints.close();

    data.reinit(mapping, dof_handler, constraints, quad, addit_data);

    compute_inverse_diagonal();
  }

  void
  vmult(LinearAlgebra::distributed::Vector<number>       &dst,
        const LinearAlgebra::distributed::Vector<number> &src) const
  {
    dst = 0;
    vmult_add(dst, src);
  }

  void
  Tvmult(LinearAlgebra::distributed::Vector<number>       &dst,
         const LinearAlgebra::distributed::Vector<number> &src) const
  {
    dst = 0;
    vmult_add(dst, src);
  }

  void
  Tvmult_add(LinearAlgebra::distributed::Vector<number>       &dst,
             const LinearAlgebra::distributed::Vector<number> &src) const
  {
    vmult_add(dst, src);
  }

  void
  vmult_add(LinearAlgebra::distributed::Vector<number>       &dst,
            const LinearAlgebra::distributed::Vector<number> &src) const
  {
    data.cell_loop(&LaplaceOperator::local_apply, this, dst, src);

    const std::vector<unsigned int> &constrained_dofs =
      data.get_constrained_dofs();
    for (unsigned int i = 0; i < constrained_dofs.size(); ++i)
      dst.local_element(constrained_dofs[i]) +=
        src.local_element(constrained_dofs[i]);
  }

  types::global_dof_index
  m() const
  {
    return data.get_vector_partitioner()->size();
  }

  types::global_dof_index
  n() const
  {
    return data.get_vector_partitioner()->size();
  }

  number
  el(const unsigned int row, const unsigned int col) const
  {
    AssertThrow(false,
                ExcMessage("Matrix-free does not allow for entry access"));
    return number();
  }

  void
  initialize_dof_vector(
    LinearAlgebra::distributed::Vector<number> &vector) const
  {
    if (!vector.partitioners_are_compatible(
          *data.get_dof_info(0).vector_partitioner))
      data.initialize_dof_vector(vector);
    Assert(vector.partitioners_are_globally_compatible(
             *data.get_dof_info(0).vector_partitioner),
           ExcInternalError());
  }

  const LinearAlgebra::distributed::Vector<number> &
  get_matrix_diagonal_inverse() const
  {
    Assert(inverse_diagonal_entries.size() > 0, ExcNotInitialized());
    return inverse_diagonal_entries;
  }

  const std::shared_ptr<const Utilities::MPI::Partitioner> &
  get_vector_partitioner() const
  {
    return data.get_vector_partitioner();
  }


private:
  void
  local_apply(const MatrixFree<dim, number>                    &data,
              LinearAlgebra::distributed::Vector<number>       &dst,
              const LinearAlgebra::distributed::Vector<number> &src,
              const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);
        phi.read_dof_values(src);
        phi.evaluate(EvaluationFlags::gradients);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_gradient(phi.get_gradient(q), q);
        phi.integrate(EvaluationFlags::gradients);
        phi.distribute_local_to_global(dst);
      }
  }

  void
  compute_inverse_diagonal()
  {
    data.initialize_dof_vector(inverse_diagonal_entries);
    unsigned int dummy = 0;
    data.cell_loop(&LaplaceOperator::local_diagonal_cell,
                   this,
                   inverse_diagonal_entries,
                   dummy);

    for (unsigned int i = 0; i < inverse_diagonal_entries.locally_owned_size();
         ++i)
      if (std::abs(inverse_diagonal_entries.local_element(i)) > 1e-10)
        inverse_diagonal_entries.local_element(i) =
          1. / inverse_diagonal_entries.local_element(i);
      else
        inverse_diagonal_entries.local_element(i) = 1.;
  }

  void
  local_diagonal_cell(
    const MatrixFree<dim, number>              &data,
    LinearAlgebra::distributed::Vector<number> &dst,
    const unsigned int &,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        phi.reinit(cell);

        VectorizedArray<number> local_diagonal_vector[phi.tensor_dofs_per_cell];
        for (unsigned int i = 0; i < phi.dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < phi.dofs_per_cell; ++j)
              phi.begin_dof_values()[j] = VectorizedArray<number>();
            phi.begin_dof_values()[i] = 1.;
            phi.evaluate(EvaluationFlags::gradients);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_gradient(phi.get_gradient(q), q);
            phi.integrate(EvaluationFlags::gradients);
            local_diagonal_vector[i] = phi.begin_dof_values()[i];
          }
        for (unsigned int i = 0; i < phi.tensor_dofs_per_cell; ++i)
          phi.begin_dof_values()[i] = local_diagonal_vector[i];
        phi.distribute_local_to_global(dst);
      }
  }

  MatrixFree<dim, number>                    data;
  LinearAlgebra::distributed::Vector<number> inverse_diagonal_entries;
};



template <typename MatrixType, typename Number>
class MGCoarseIterative
  : public MGCoarseGridBase<LinearAlgebra::distributed::Vector<Number>>
{
public:
  MGCoarseIterative()
  {}

  void
  initialize(const MatrixType &matrix)
  {
    coarse_matrix = &matrix;
  }

  virtual void
  operator()(const unsigned int                                level,
             LinearAlgebra::distributed::Vector<Number>       &dst,
             const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    ReductionControl solver_control(1e4, 1e-50, 1e-10);
    SolverCG<LinearAlgebra::distributed::Vector<Number>> solver_coarse(
      solver_control);
    solver_coarse.solve(*coarse_matrix, dst, src, PreconditionIdentity());
  }

  const MatrixType *coarse_matrix;
};



template <int dim, int fe_degree, int n_q_points_1d, typename number>
void
do_test(const DoFHandler<dim> &dof)
{
  deallog << "Testing " << dof.get_fe().get_name();
  deallog << std::endl;
  deallog << "Number of degrees of freedom: " << dof.n_dofs() << std::endl;

  MappingQ<dim>                                          mapping(fe_degree + 1);
  LaplaceOperator<dim, fe_degree, n_q_points_1d, double> fine_matrix;
  const std::set<types::boundary_id> dirichlet_boundaries = {0};
  fine_matrix.initialize(mapping, dof, dirichlet_boundaries);

  LinearAlgebra::distributed::Vector<double> in, sol;
  fine_matrix.initialize_dof_vector(in);
  fine_matrix.initialize_dof_vector(sol);

  // set constant rhs vector
  in = 1.;

  // set up multigrid in analogy to step-37
  using LevelMatrixType =
    LaplaceOperator<dim, fe_degree, n_q_points_1d, number>;

  MGLevelObject<LevelMatrixType> mg_matrices;
  mg_matrices.resize(0, dof.get_triangulation().n_global_levels() - 1);
  for (unsigned int level = 0;
       level < dof.get_triangulation().n_global_levels();
       ++level)
    {
      mg_matrices[level].initialize(mapping, dof, dirichlet_boundaries, level);
    }

  MGConstrainedDoFs mg_constrained_dofs;
  mg_constrained_dofs.initialize(dof);
  mg_constrained_dofs.make_zero_boundary_constraints(dof, {0});

  std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners;
  for (unsigned int level = mg_matrices.min_level();
       level <= mg_matrices.max_level();
       ++level)
    partitioners.push_back(mg_matrices[level].get_vector_partitioner());

  MGTransferMatrixFree<dim, number> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof, partitioners);

  MGCoarseIterative<LevelMatrixType, number> mg_coarse;
  mg_coarse.initialize(mg_matrices[0]);

  using SMOOTHER =
    PreconditionChebyshev<LevelMatrixType,
                          LinearAlgebra::distributed::Vector<number>>;
  MGSmootherPrecondition<LevelMatrixType,
                         SMOOTHER,
                         LinearAlgebra::distributed::Vector<number>>
    mg_smoother;

  MGLevelObject<typename SMOOTHER::AdditionalData> smoother_data;
  smoother_data.resize(0, dof.get_triangulation().n_global_levels() - 1);
  for (unsigned int level = 0;
       level < dof.get_triangulation().n_global_levels();
       ++level)
    {
      smoother_data[level].smoothing_range     = 15.;
      smoother_data[level].degree              = 5;
      smoother_data[level].eig_cg_n_iterations = 15;
      auto preconditioner                      = std::make_shared<
        DiagonalMatrix<LinearAlgebra::distributed::Vector<number>>>();
      preconditioner->reinit(mg_matrices[level].get_matrix_diagonal_inverse());
      smoother_data[level].preconditioner = std::move(preconditioner);
    }

  // temporarily disable deallog for the setup of the preconditioner that
  // involves a CG solver for eigenvalue estimation
  mg_smoother.initialize(mg_matrices, smoother_data);

  mg::Matrix<LinearAlgebra::distributed::Vector<number>> mg_matrix(mg_matrices);

  Multigrid<LinearAlgebra::distributed::Vector<number>> mg(
    mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
  PreconditionMG<dim,
                 LinearAlgebra::distributed::Vector<number>,
                 MGTransferMatrixFree<dim, number>>
    preconditioner(dof, mg, mg_transfer);

  {
    // avoid output from inner (coarse-level) solver
    deallog.depth_file(2);
    ReductionControl control(30, 1e-20, 1e-7);
    SolverCG<LinearAlgebra::distributed::Vector<double>> solver(control);
    solver.solve(fine_matrix, sol, in, preconditioner);
  }
}



template <int dim, int fe_degree, typename number>
void
test()
{
  for (unsigned int i = 5; i < 7; ++i)
    {
      dealii::Triangulation<dim> tria(
        Triangulation<dim>::limit_level_difference_at_vertices);
      GridGenerator::hyper_cube(tria);
      tria.refine_global(i - dim);

      GridTools::partition_triangulation_zorder(
        Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), tria);
      GridTools::partition_multigrid_levels(tria);


      parallel::fullydistributed::Triangulation<dim> tria_pft(MPI_COMM_WORLD);

      // extract relevant information form pdt
      auto construction_data = TriangulationDescription::Utilities::
        create_description_from_triangulation(
          tria,
          MPI_COMM_WORLD,
          TriangulationDescription::Settings::construct_multigrid_hierarchy);

      // actually create triangulation
      tria_pft.create_triangulation(construction_data);

      FE_Q<dim>       fe(fe_degree);
      DoFHandler<dim> dof(tria_pft);
      dof.distribute_dofs(fe);
      dof.distribute_mg_dofs();

      do_test<dim, fe_degree, fe_degree + 1, number>(dof);
    }

  if (false)
    for (unsigned int i = 5; i < 7; ++i)
      {
        parallel::distributed::Triangulation<dim> tria(
          MPI_COMM_WORLD,
          Triangulation<dim>::limit_level_difference_at_vertices,
          parallel::distributed::Triangulation<
            dim>::construct_multigrid_hierarchy);
        GridGenerator::hyper_cube(tria);
        tria.refine_global(i - dim);

        parallel::fullydistributed::Triangulation<dim> tria_pft(MPI_COMM_WORLD);

        // extract relevant information form pdt
        auto construction_data = TriangulationDescription::Utilities::
          create_description_from_triangulation(
            tria,
            MPI_COMM_WORLD,
            TriangulationDescription::Settings::construct_multigrid_hierarchy);

        // actually create triangulation
        tria_pft.create_triangulation(construction_data);

        FE_Q<dim>       fe(fe_degree);
        DoFHandler<dim> dof(tria_pft);
        dof.distribute_dofs(fe);
        dof.distribute_mg_dofs();

        do_test<dim, fe_degree, fe_degree + 1, number>(dof);
      }
}



int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  mpi_initlog();

  {
    test<2, 1, double>();
    test<2, 2, float>();

    test<3, 1, double>();
    test<3, 2, float>();
  }
}
