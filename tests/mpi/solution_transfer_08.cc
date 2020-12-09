// Test case based on the one written by K. Bzowski
#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/hp/dof_handler.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/hp/fe_collection.h>

#include <deal.II/numerics/data_out.h>
#include <iostream>
#include <fstream>

#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/lac/generic_linear_algebra.h>

// uncomment the following \#define if you have PETSc and Trilinos installed
// and you prefer using Trilinos in this example:
// @code
// #define FORCE_USE_OF_TRILINOS
// @endcode

// This will either import PETSc or TrilinosWrappers into the namespace
// LA. Note that we are defining the macro USE_PETSC_LA so that we can detect
// if we are using PETSc (see solve() for an example where this is necessary)

// This LA namespace must be after including <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
	!(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
	using namespace dealii::LinearAlgebraPETSc;
#define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
	using namespace dealii::LinearAlgebraTrilinos;
#else
#error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/lac/vector.h>

#include "../tests.h"

int main(int argc, char *argv[])
{
	Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

	std::ofstream logfile("output");
	deallog.attach(logfile);
	deallog.depth_console(0);

	MPI_Comm mpi_communicator(MPI_COMM_WORLD);

	parallel::distributed::Triangulation<2, 2> triangulation(
		mpi_communicator,
		typename Triangulation<2>::MeshSmoothing(Triangulation<2>::none));

	GridGenerator::hyper_cube(triangulation);
	triangulation.refine_global(1);

	hp::FECollection<2> fe_collection;
	fe_collection.push_back(FE_Q<2>(1));
	fe_collection.push_back(FE_Nothing<2>());
	//fe_collection.push_back(FE_Q<2>(1));

	hp::DoFHandler<2> dof_handler(triangulation);

	// Assign FE
	/*
	 * -----------
	 * |  0 |  0 |
	 * -----------
	 * |  1 |  1 |		0 - FEQ, 1 - FE_Nothing
	 * -----------
	 */

	for (auto &cell : dof_handler.active_cell_iterators())
	{
		if (cell->is_locally_owned())
		{
			auto center = cell->center();
			if (center(1) < 0.5)
			{
				cell->set_active_fe_index(1);
			}
			else
			{
				cell->set_active_fe_index(0);
			}
		}
	}

	dof_handler.distribute_dofs(fe_collection);

	IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
	IndexSet locally_relevant_dofs;
	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

	LA::MPI::Vector completely_distributed_solution;
	LA::MPI::Vector locally_relevant_solution;

	completely_distributed_solution.reinit(locally_owned_dofs, mpi_communicator);
	locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

	completely_distributed_solution = 1.0;

	locally_relevant_solution = completely_distributed_solution;

	Vector<double> FE_Type(triangulation.n_active_cells());
	Vector<float> subdomain(triangulation.n_active_cells());
	int i = 0;
	for (auto &cell : dof_handler.active_cell_iterators())
	{
		if (cell->is_locally_owned())
		{
			FE_Type(i) = cell->active_fe_index();
			subdomain(i) = triangulation.locally_owned_subdomain();
		}
		else
		{
			FE_Type(i) = -1;
			subdomain(i) = -1;
		}
		i++;
	}

	// Save output
	{
		DataOut<2, hp::DoFHandler<2>> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(locally_relevant_solution, "Solution");
		data_out.add_data_vector(FE_Type, "FE_Type");
		data_out.add_data_vector(subdomain, "subdomain");
		data_out.build_patches();

		data_out.write_vtu_with_pvtu_record(
			"./", "solution", 1, mpi_communicator, 2);
	}

	/* Set refine flags:
	 * -----------
	 * |  R |  R |  FEQ
	 * -----------
	 * |    |    |	FE_Nothing
	 * -----------
	 */

	for (auto &cell : dof_handler.active_cell_iterators())
	{
		if (cell->is_locally_owned())
		{
			auto center = cell->center();
			if (center(1) > 0.5)
			{
				cell->set_refine_flag();
			}
		}
	}

	LA::MPI::Vector previous_locally_relevant_solution;
	previous_locally_relevant_solution = locally_relevant_solution;

	parallel::distributed::SolutionTransfer<2, LA::MPI::Vector, hp::DoFHandler<2>> solution_trans(dof_handler);
	
	triangulation.prepare_coarsening_and_refinement();
	solution_trans.prepare_for_coarsening_and_refinement(previous_locally_relevant_solution);

	triangulation.execute_coarsening_and_refinement();

	for (auto &cell : dof_handler.active_cell_iterators())
	{
		if (cell->is_locally_owned())
		{
			auto center = cell->center();
			if (center(1) < 0.5)
			{
				cell->set_active_fe_index(1);
			}
			else
			{
				cell->set_active_fe_index(0);
			}
		}
	}

	dof_handler.distribute_dofs(fe_collection);

	locally_owned_dofs = dof_handler.locally_owned_dofs();
	locally_relevant_dofs.clear();
	DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
	completely_distributed_solution.reinit(locally_owned_dofs, mpi_communicator);
	locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

	solution_trans.interpolate(completely_distributed_solution);
	locally_relevant_solution = completely_distributed_solution;

	FE_Type.reinit(triangulation.n_active_cells());
	subdomain.reinit(triangulation.n_active_cells());
	i = 0;
	for (auto &cell : dof_handler.active_cell_iterators())
	{
		if (cell->is_locally_owned())
		{
			FE_Type(i) = cell->active_fe_index();
			subdomain(i) = triangulation.locally_owned_subdomain();
		}
		else
		{
			FE_Type(i) = -1;
			subdomain(i) = -1;
		}
		i++;
	}

	// Save output
	{
		DataOut<2, hp::DoFHandler<2>> data_out;
		data_out.attach_dof_handler(dof_handler);
		data_out.add_data_vector(locally_relevant_solution, "Solution");
		data_out.add_data_vector(FE_Type, "FE_Type");
		data_out.add_data_vector(subdomain, "subdomain");
		data_out.build_patches();

		data_out.write_vtu_with_pvtu_record(
			"./", "solution", 2, mpi_communicator, 2);
	}

}
