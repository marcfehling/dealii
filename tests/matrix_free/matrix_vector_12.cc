// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2013 - 2023 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------



// this tests the correctness of matrix free matrix-vector products for two
// vectors on the same DoFHandler. Otherwise the same as matrix_vector_10.cc

#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_sparsity_pattern.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include <iostream>

#include "../tests.h"


template <int dim, int fe_degree, typename Number>
void
helmholtz_operator(
  const MatrixFree<dim, Number>                                 &data,
  std::vector<LinearAlgebra::distributed::Vector<Number>>       &dst,
  const std::vector<LinearAlgebra::distributed::Vector<Number>> &src,
  const std::pair<unsigned int, unsigned int>                   &cell_range)
{
  FEEvaluation<dim, fe_degree, fe_degree + 1, 2, Number> fe_eval(data);
  const unsigned int n_q_points = fe_eval.n_q_points;

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
      for (unsigned int q = 0; q < n_q_points; ++q)
        {
          fe_eval.submit_value(make_vectorized_array(Number(10)) *
                                 fe_eval.get_value(q),
                               q);
          fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
        }
      fe_eval.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
      fe_eval.distribute_local_to_global(dst);
    }
}



template <int dim, int fe_degree, typename Number>
class MatrixFreeTest
{
public:
  using vector_t                     = VectorizedArray<Number>;
  static const std::size_t n_vectors = VectorizedArray<Number>::size();

  MatrixFreeTest(const MatrixFree<dim, Number> &data_in)
    : data(data_in){};

  void
  vmult(
    std::vector<LinearAlgebra::distributed::Vector<Number>>       &dst,
    const std::vector<LinearAlgebra::distributed::Vector<Number>> &src) const
  {
    for (unsigned int i = 0; i < dst.size(); ++i)
      dst[i] = 0;
    const std::function<
      void(const MatrixFree<dim, Number> &,
           std::vector<LinearAlgebra::distributed::Vector<Number>> &,
           const std::vector<LinearAlgebra::distributed::Vector<Number>> &,
           const std::pair<unsigned int, unsigned int> &)>
      wrap = helmholtz_operator<dim, fe_degree, Number>;
    data.cell_loop(wrap, dst, src);
  };

private:
  const MatrixFree<dim, Number> &data;
};



template <int dim, int fe_degree>
void
test()
{
  using number = double;

  parallel::distributed::Triangulation<dim> tria(MPI_COMM_WORLD);
  GridGenerator::hyper_cube(tria);
  tria.refine_global(1);
  typename Triangulation<dim>::active_cell_iterator cell = tria.begin_active(),
                                                    endc = tria.end();
  cell                                                   = tria.begin_active();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      if (cell->center().norm() < 0.2)
        cell->set_refine_flag();
  tria.execute_coarsening_and_refinement();
  if (dim < 3 && fe_degree < 2)
    tria.refine_global(2);
  else
    tria.refine_global(1);
  if (tria.begin(tria.n_levels() - 1)->is_locally_owned())
    tria.begin(tria.n_levels() - 1)->set_refine_flag();
  if (tria.last()->is_locally_owned())
    tria.last()->set_refine_flag();
  tria.execute_coarsening_and_refinement();
  cell = tria.begin_active();
  for (unsigned int i = 0; i < 10 - 3 * dim; ++i)
    {
      cell                 = tria.begin_active();
      unsigned int counter = 0;
      for (; cell != endc; ++cell, ++counter)
        if (cell->is_locally_owned())
          if (counter % (7 - i) == 0)
            cell->set_refine_flag();
      tria.execute_coarsening_and_refinement();
    }

  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);

  const IndexSet &owned_set    = dof.locally_owned_dofs();
  const IndexSet  relevant_set = DoFTools::extract_locally_relevant_dofs(dof);

  AffineConstraints<double> constraints(owned_set, relevant_set);
  DoFTools::make_hanging_node_constraints(dof, constraints);
  VectorTools::interpolate_boundary_values(dof,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  deallog << "Testing " << dof.get_fe().get_name() << std::endl;
  // std::cout << "Number of cells: " << tria.n_global_active_cells() <<
  // std::endl; std::cout << "Number of degrees of freedom: " << dof.n_dofs() <<
  // std::endl; std::cout << "Number of constraints: " <<
  // constraints.n_constraints() << std::endl;

  MatrixFree<dim, number> mf_data;
  {
    const QGauss<1>                                  quad(fe_degree + 1);
    typename MatrixFree<dim, number>::AdditionalData data;
    data.tasks_parallel_scheme = MatrixFree<dim, number>::AdditionalData::none;
    data.tasks_block_size      = 7;
    mf_data.reinit(MappingQ1<dim>{}, dof, constraints, quad, data);
  }

  MatrixFreeTest<dim, fe_degree, number>                  mf(mf_data);
  LinearAlgebra::distributed::Vector<number>              ref;
  std::vector<LinearAlgebra::distributed::Vector<number>> in(2), out(2);
  for (unsigned int i = 0; i < 2; ++i)
    {
      mf_data.initialize_dof_vector(in[i]);
      mf_data.initialize_dof_vector(out[i]);
    }
  mf_data.initialize_dof_vector(ref);

  for (unsigned int i = 0; i < in[0].locally_owned_size(); ++i)
    {
      const unsigned int glob_index = owned_set.nth_index_in_set(i);
      if (constraints.is_constrained(glob_index))
        continue;
      in[0].local_element(i) = random_value<double>();
      in[1].local_element(i) = random_value<double>();
    }

  mf.vmult(out, in);


  // assemble trilinos sparse matrix with
  // (\nabla v, \nabla u) + (v, 10 * u) for
  // reference
  TrilinosWrappers::SparseMatrix sparse_matrix;
  {
    TrilinosWrappers::SparsityPattern csp(owned_set, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof,
                                    csp,
                                    constraints,
                                    true,
                                    Utilities::MPI::this_mpi_process(
                                      MPI_COMM_WORLD));
    csp.compress();
    sparse_matrix.reinit(csp);
  }
  {
    QGauss<dim> quadrature_formula(fe_degree + 1);

    FEValues<dim> fe_values(dof.get_fe(),
                            quadrature_formula,
                            update_values | update_gradients |
                              update_JxW_values);

    const unsigned int dofs_per_cell = dof.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator cell = dof.begin_active(),
                                                   endc = dof.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          cell_matrix = 0;
          fe_values.reinit(cell);

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) +=
                    ((fe_values.shape_grad(i, q_point) *
                        fe_values.shape_grad(j, q_point) +
                      10. * fe_values.shape_value(i, q_point) *
                        fe_values.shape_value(j, q_point)) *
                     fe_values.JxW(q_point));
              }

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix,
                                                 local_dof_indices,
                                                 sparse_matrix);
        }
  }
  sparse_matrix.compress(VectorOperation::add);

  deallog << "Norm of difference (component 1/2): ";
  for (unsigned int i = 0; i < 2; ++i)
    {
      sparse_matrix.vmult(ref, in[i]);
      out[i] -= ref;
      const double diff_norm = out[i].linfty_norm();
      deallog << diff_norm << ' ';
    }
  deallog << std::endl << std::endl;
}


int
main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(
    argc, argv, testing_max_num_threads());

  unsigned int myid = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  deallog.push(Utilities::int_to_string(myid));

  if (myid == 0)
    {
      initlog();
      deallog << std::setprecision(4);

      deallog.push("2d");
      test<2, 1>();
      test<2, 2>();
      deallog.pop();

      deallog.push("3d");
      test<3, 1>();
      test<3, 2>();
      deallog.pop();
    }
  else
    {
      test<2, 1>();
      test<2, 2>();
      test<3, 1>();
      test<3, 2>();
    }
}
