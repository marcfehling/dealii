/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2008 - 2024 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * Part of the source code is dual licensed under Apache-2.0 WITH
 * LLVM-exception OR LGPL-2.1-or-later. Detailed license information
 * governing the source code and code contributions can be found in
 * LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
 *
 * ------------------------------------------------------------------------
 *
 * Author: Tobias Leicht, 2007
 */


// The deal.II include files have already been covered in previous examples
// and will thus not be further commented on.
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/derivative_approximation.h>

// And this again is C++:
#include <array>
#include <iostream>
#include <fstream>

// The last step is as in all previous programs:
namespace Step30
{
  using namespace dealii;

  // @sect3{Equation data}
  //
  // The classes describing equation data and the actual assembly of
  // individual terms are almost entirely copied from step-12. We will comment
  // on differences.
  template <int dim>
  class RHS : public Function<dim>
  {
  public:
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double>           &values,
                            const unsigned int /*component*/ = 0) const override
    {
      (void)points;
      AssertDimension(values.size(), points.size());

      std::fill(values.begin(), values.end(), 0.);
    }
  };


  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double>           &values,
                            const unsigned int /*component*/ = 0) const override
    {
      AssertDimension(values.size(), points.size());

      for (unsigned int i = 0; i < values.size(); ++i)
        {
          if (points[i][0] < 0.5)
            values[i] = 1.;
          else
            values[i] = 0.;
        }
    }
  };


  template <int dim>
  class Beta
  {
  public:
    // The flow field is chosen to be a quarter circle with counterclockwise
    // flow direction and with the origin as midpoint for the right half of the
    // domain with positive $x$ values, whereas the flow simply goes to the left
    // in the left part of the domain at a velocity that matches the one coming
    // in from the right. In the circular part the magnitude of the flow
    // velocity is proportional to the distance from the origin. This is a
    // difference to step-12, where the magnitude was 1 everywhere. the new
    // definition leads to a linear variation of $\beta$ along each given face
    // of a cell. On the other hand, the solution $u(x,y)$ is exactly the same
    // as before.
    void value_list(const std::vector<Point<dim>> &points,
                    std::vector<Point<dim>>       &values) const
    {
      AssertDimension(values.size(), points.size());

      for (unsigned int i = 0; i < points.size(); ++i)
        {
          if (points[i][0] > 0)
            {
              values[i][0] = -points[i][1];
              values[i][1] = points[i][0];
            }
          else
            {
              values[i]    = Point<dim>();
              values[i][0] = -points[i][1];
            }
        }
    }
  };



  // @sect3{Class: DGTransportEquation}
  //
  // This declaration of this class is utterly unaffected by our current
  // changes.
  template <int dim>
  class DGTransportEquation
  {
  public:
    DGTransportEquation();

    void assemble_cell_term(const FEValues<dim> &fe_v,
                            FullMatrix<double>  &ui_vi_matrix,
                            Vector<double>      &cell_vector) const;

    void assemble_boundary_term(const FEFaceValues<dim> &fe_v,
                                FullMatrix<double>      &ui_vi_matrix,
                                Vector<double>          &cell_vector) const;

    void assemble_face_term(const FEFaceValuesBase<dim> &fe_v,
                            const FEFaceValuesBase<dim> &fe_v_neighbor,
                            FullMatrix<double>          &ui_vi_matrix,
                            FullMatrix<double>          &ue_vi_matrix,
                            FullMatrix<double>          &ui_ve_matrix,
                            FullMatrix<double>          &ue_ve_matrix) const;

  private:
    const Beta<dim>           beta_function;
    const RHS<dim>            rhs_function;
    const BoundaryValues<dim> boundary_function;
  };



  // Likewise, the constructor of the class as well as the functions
  // assembling the terms corresponding to cell interiors and boundary faces
  // are unchanged from before. The function that assembles face terms between
  // cells also did not change because all it does is operate on two objects
  // of type FEFaceValuesBase (which is the base class of both FEFaceValues
  // and FESubfaceValues). Where these objects come from, i.e. how they are
  // initialized, is of no concern to this function: it simply assumes that
  // the quadrature points on faces or subfaces represented by the two objects
  // correspond to the same points in physical space.
  template <int dim>
  DGTransportEquation<dim>::DGTransportEquation()
    : beta_function()
    , rhs_function()
    , boundary_function()
  {}



  template <int dim>
  void DGTransportEquation<dim>::assemble_cell_term(
    const FEValues<dim> &fe_v,
    FullMatrix<double>  &ui_vi_matrix,
    Vector<double>      &cell_vector) const
  {
    const std::vector<double> &JxW = fe_v.get_JxW_values();

    std::vector<Point<dim>> beta(fe_v.n_quadrature_points);
    std::vector<double>     rhs(fe_v.n_quadrature_points);

    beta_function.value_list(fe_v.get_quadrature_points(), beta);
    rhs_function.value_list(fe_v.get_quadrature_points(), rhs);

    for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
      for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
        {
          for (unsigned int j = 0; j < fe_v.dofs_per_cell; ++j)
            ui_vi_matrix(i, j) -= beta[point] * fe_v.shape_grad(i, point) *
                                  fe_v.shape_value(j, point) * JxW[point];

          cell_vector(i) +=
            rhs[point] * fe_v.shape_value(i, point) * JxW[point];
        }
  }


  template <int dim>
  void DGTransportEquation<dim>::assemble_boundary_term(
    const FEFaceValues<dim> &fe_v,
    FullMatrix<double>      &ui_vi_matrix,
    Vector<double>          &cell_vector) const
  {
    const std::vector<double>         &JxW     = fe_v.get_JxW_values();
    const std::vector<Tensor<1, dim>> &normals = fe_v.get_normal_vectors();

    std::vector<Point<dim>> beta(fe_v.n_quadrature_points);
    std::vector<double>     g(fe_v.n_quadrature_points);

    beta_function.value_list(fe_v.get_quadrature_points(), beta);
    boundary_function.value_list(fe_v.get_quadrature_points(), g);

    for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
      {
        const double beta_n = beta[point] * normals[point];
        if (beta_n > 0)
          for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
            for (unsigned int j = 0; j < fe_v.dofs_per_cell; ++j)
              ui_vi_matrix(i, j) += beta_n * fe_v.shape_value(j, point) *
                                    fe_v.shape_value(i, point) * JxW[point];
        else
          for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
            cell_vector(i) -=
              beta_n * g[point] * fe_v.shape_value(i, point) * JxW[point];
      }
  }


  template <int dim>
  void DGTransportEquation<dim>::assemble_face_term(
    const FEFaceValuesBase<dim> &fe_v,
    const FEFaceValuesBase<dim> &fe_v_neighbor,
    FullMatrix<double>          &ui_vi_matrix,
    FullMatrix<double>          &ue_vi_matrix,
    FullMatrix<double>          &ui_ve_matrix,
    FullMatrix<double>          &ue_ve_matrix) const
  {
    const std::vector<double>         &JxW     = fe_v.get_JxW_values();
    const std::vector<Tensor<1, dim>> &normals = fe_v.get_normal_vectors();

    std::vector<Point<dim>> beta(fe_v.n_quadrature_points);

    beta_function.value_list(fe_v.get_quadrature_points(), beta);

    for (unsigned int point = 0; point < fe_v.n_quadrature_points; ++point)
      {
        const double beta_n = beta[point] * normals[point];
        if (beta_n > 0)
          {
            for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
              for (unsigned int j = 0; j < fe_v.dofs_per_cell; ++j)
                ui_vi_matrix(i, j) += beta_n * fe_v.shape_value(j, point) *
                                      fe_v.shape_value(i, point) * JxW[point];

            for (unsigned int k = 0; k < fe_v_neighbor.dofs_per_cell; ++k)
              for (unsigned int j = 0; j < fe_v.dofs_per_cell; ++j)
                ui_ve_matrix(k, j) -= beta_n * fe_v.shape_value(j, point) *
                                      fe_v_neighbor.shape_value(k, point) *
                                      JxW[point];
          }
        else
          {
            for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
              for (unsigned int l = 0; l < fe_v_neighbor.dofs_per_cell; ++l)
                ue_vi_matrix(i, l) += beta_n *
                                      fe_v_neighbor.shape_value(l, point) *
                                      fe_v.shape_value(i, point) * JxW[point];

            for (unsigned int k = 0; k < fe_v_neighbor.dofs_per_cell; ++k)
              for (unsigned int l = 0; l < fe_v_neighbor.dofs_per_cell; ++l)
                ue_ve_matrix(k, l) -=
                  beta_n * fe_v_neighbor.shape_value(l, point) *
                  fe_v_neighbor.shape_value(k, point) * JxW[point];
          }
      }
  }


  // @sect3{Class: DGMethod}
  //
  // This declaration is much like that of step-12. However, we introduce a
  // new routine (set_anisotropic_flags) and modify another one (refine_grid).
  template <int dim>
  class DGMethod
  {
  public:
    DGMethod(const bool anisotropic);

    void run();

  private:
    void setup_system();
    void assemble_system();
    void solve(Vector<double> &solution);
    void refine_grid();
    void set_anisotropic_flags();
    void output_results(const unsigned int cycle) const;

    Triangulation<dim>   triangulation;
    const MappingQ1<dim> mapping;
    // Again we want to use DG elements of degree 1 (but this is only
    // specified in the constructor). If you want to use a DG method of a
    // different degree replace 1 in the constructor by the new degree.
    const unsigned int degree;
    const FE_DGQ<dim>  fe;
    DoFHandler<dim>    dof_handler;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    // This is new, the threshold value used in the evaluation of the
    // anisotropic jump indicator explained in the introduction. Its value is
    // set to 3.0 in the constructor, but it can easily be changed to a
    // different value greater than 1.
    const double anisotropic_threshold_ratio;
    // This is a bool flag indicating whether anisotropic refinement shall be
    // used or not. It is set by the constructor, which takes an argument of
    // the same name.
    const bool anisotropic;

    const QGauss<dim>     quadrature;
    const QGauss<dim - 1> face_quadrature;

    Vector<double> solution2;
    Vector<double> right_hand_side;

    const DGTransportEquation<dim> dg;
  };


  template <int dim>
  DGMethod<dim>::DGMethod(const bool anisotropic)
    : mapping()
    // Change here for DG methods of different degrees.
    , degree(1)
    , fe(degree)
    , dof_handler(triangulation)
    , anisotropic_threshold_ratio(3.)
    , anisotropic(anisotropic)
    // As beta is a linear function, we can choose the degree of the
    // quadrature for which the resulting integration is correct. Thus, we
    // choose to use <code>degree+1</code> Gauss points, which enables us to
    // integrate exactly polynomials of degree <code>2*degree+1</code>, enough
    // for all the integrals we will perform in this program.
    , quadrature(degree + 1)
    , face_quadrature(degree + 1)
    , dg()
  {}



  template <int dim>
  void DGMethod<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    sparsity_pattern.reinit(dof_handler.n_dofs(),
                            dof_handler.n_dofs(),
                            (GeometryInfo<dim>::faces_per_cell *
                               GeometryInfo<dim>::max_children_per_face +
                             1) *
                              fe.n_dofs_per_cell());

    DoFTools::make_flux_sparsity_pattern(dof_handler, sparsity_pattern);

    sparsity_pattern.compress();

    system_matrix.reinit(sparsity_pattern);

    solution2.reinit(dof_handler.n_dofs());
    right_hand_side.reinit(dof_handler.n_dofs());
  }


  // @sect4{Function: assemble_system}
  //
  // We proceed with the <code>assemble_system</code> function that implements
  // the DG discretization. This function does the same thing as the
  // <code>assemble_system</code> function from step-12 (but without
  // MeshWorker).  The four cases considered for the neighbor-relations of a
  // cell are the same as the isotropic case, namely a) cell is at the
  // boundary, b) there are finer neighboring cells, c) the neighbor is
  // neither coarser nor finer and d) the neighbor is coarser.  However, the
  // way in which we decide upon which case we have are modified in the way
  // described in the introduction.
  template <int dim>
  void DGMethod<dim>::assemble_system()
  {
    const unsigned int dofs_per_cell = dof_handler.get_fe().n_dofs_per_cell();
    std::vector<types::global_dof_index> dofs(dofs_per_cell);
    std::vector<types::global_dof_index> dofs_neighbor(dofs_per_cell);

    const UpdateFlags update_flags = update_values | update_gradients |
                                     update_quadrature_points |
                                     update_JxW_values;

    const UpdateFlags face_update_flags =
      update_values | update_quadrature_points | update_JxW_values |
      update_normal_vectors;

    const UpdateFlags neighbor_face_update_flags = update_values;

    FEValues<dim>        fe_v(mapping, fe, quadrature, update_flags);
    FEFaceValues<dim>    fe_v_face(mapping,
                                fe,
                                face_quadrature,
                                face_update_flags);
    FESubfaceValues<dim> fe_v_subface(mapping,
                                      fe,
                                      face_quadrature,
                                      face_update_flags);
    FEFaceValues<dim>    fe_v_face_neighbor(mapping,
                                         fe,
                                         face_quadrature,
                                         neighbor_face_update_flags);


    FullMatrix<double> ui_vi_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> ue_vi_matrix(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> ui_ve_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> ue_ve_matrix(dofs_per_cell, dofs_per_cell);

    Vector<double> cell_vector(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        ui_vi_matrix = 0;
        cell_vector  = 0;

        fe_v.reinit(cell);

        dg.assemble_cell_term(fe_v, ui_vi_matrix, cell_vector);

        cell->get_dof_indices(dofs);

        for (const auto face_no : cell->face_indices())
          {
            const auto face = cell->face(face_no);

            // Case (a): The face is at the boundary.
            if (face->at_boundary())
              {
                fe_v_face.reinit(cell, face_no);

                dg.assemble_boundary_term(fe_v_face, ui_vi_matrix, cell_vector);
              }
            else
              {
                Assert(cell->neighbor(face_no).state() == IteratorState::valid,
                       ExcInternalError());
                const auto neighbor = cell->neighbor(face_no);

                // Case (b): This is an internal face and the neighbor
                // is refined (which we can test by asking whether the
                // face of the current cell has children). In this
                // case, we will need to integrate over the
                // "sub-faces", i.e., the children of the face of the
                // current cell.
                //
                // (There is a slightly confusing corner case: If we
                // are in 1d -- where admittedly the current program
                // and its demonstration of anisotropic refinement is
                // not particularly relevant -- then the faces between
                // cells are always the same: they are just
                // vertices. In other words, in 1d, we do not want to
                // treat faces between cells of different level
                // differently. The condition `face->has_children()`
                // we check here ensures this: in 1d, this function
                // always returns `false`, and consequently in 1d we
                // will not ever go into this `if` branch. But we will
                // have to come back to this corner case below in case
                // (c).)
                if (face->has_children())
                  {
                    // We need to know, which of the neighbors faces points in
                    // the direction of our cell. Using the @p
                    // neighbor_face_no function we get this information for
                    // both coarser and non-coarser neighbors.
                    const unsigned int neighbor2 =
                      cell->neighbor_face_no(face_no);

                    // Now we loop over all subfaces, i.e. the children and
                    // possibly grandchildren of the current face.
                    for (unsigned int subface_no = 0;
                         subface_no < face->n_active_descendants();
                         ++subface_no)
                      {
                        // To get the cell behind the current subface we can
                        // use the @p neighbor_child_on_subface function. it
                        // takes care of all the complicated situations of
                        // anisotropic refinement and non-standard faces.
                        const auto neighbor_child =
                          cell->neighbor_child_on_subface(face_no, subface_no);
                        Assert(!neighbor_child->has_children(),
                               ExcInternalError());

                        // The remaining part of this case is unchanged.
                        ue_vi_matrix = 0;
                        ui_ve_matrix = 0;
                        ue_ve_matrix = 0;

                        fe_v_subface.reinit(cell, face_no, subface_no);
                        fe_v_face_neighbor.reinit(neighbor_child, neighbor2);

                        dg.assemble_face_term(fe_v_subface,
                                              fe_v_face_neighbor,
                                              ui_vi_matrix,
                                              ue_vi_matrix,
                                              ui_ve_matrix,
                                              ue_ve_matrix);

                        neighbor_child->get_dof_indices(dofs_neighbor);

                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              system_matrix.add(dofs[i],
                                                dofs_neighbor[j],
                                                ue_vi_matrix(i, j));
                              system_matrix.add(dofs_neighbor[i],
                                                dofs[j],
                                                ui_ve_matrix(i, j));
                              system_matrix.add(dofs_neighbor[i],
                                                dofs_neighbor[j],
                                                ue_ve_matrix(i, j));
                            }
                      }
                  }
                else
                  {
                    // Case (c). We get here if this is an internal
                    // face and if the neighbor is not further refined
                    // (or, as mentioned above, we are in 1d in which
                    // case we get here for every internal face). We
                    // then need to decide whether we want to
                    // integrate over the current face. If the
                    // neighbor is in fact coarser, then we ignore the
                    // face and instead handle it when we visit the
                    // neighboring cell and look at the current face
                    // (except in 1d, where as mentioned above this is
                    // not happening):
                    if (dim > 1 && cell->neighbor_is_coarser(face_no))
                      continue;

                    // On the other hand, if the neighbor is more
                    // refined, then we have already handled the face
                    // in case (b) above (except in 1d). So for 2d and
                    // 3d, we just have to decide whether we want to
                    // handle a face between cells at the same level
                    // from the current side or from the neighboring
                    // side.  We do this by introducing a tie-breaker:
                    // We'll just take the cell with the smaller index
                    // (within the current refinement level). In 1d,
                    // we take either the coarser cell, or if they are
                    // on the same level, the one with the smaller
                    // index within that level. This leads to a
                    // complicated condition that, hopefully, makes
                    // sense given the description above:
                    if (((dim > 1) && (cell->index() < neighbor->index())) ||
                        ((dim == 1) && ((cell->level() < neighbor->level()) ||
                                        ((cell->level() == neighbor->level()) &&
                                         (cell->index() < neighbor->index())))))
                      {
                        // Here we know, that the neighbor is not coarser so we
                        // can use the usual @p neighbor_of_neighbor
                        // function. However, we could also use the more
                        // general @p neighbor_face_no function.
                        const unsigned int neighbor2 =
                          cell->neighbor_of_neighbor(face_no);

                        ue_vi_matrix = 0;
                        ui_ve_matrix = 0;
                        ue_ve_matrix = 0;

                        fe_v_face.reinit(cell, face_no);
                        fe_v_face_neighbor.reinit(neighbor, neighbor2);

                        dg.assemble_face_term(fe_v_face,
                                              fe_v_face_neighbor,
                                              ui_vi_matrix,
                                              ue_vi_matrix,
                                              ui_ve_matrix,
                                              ue_ve_matrix);

                        neighbor->get_dof_indices(dofs_neighbor);

                        for (unsigned int i = 0; i < dofs_per_cell; ++i)
                          for (unsigned int j = 0; j < dofs_per_cell; ++j)
                            {
                              system_matrix.add(dofs[i],
                                                dofs_neighbor[j],
                                                ue_vi_matrix(i, j));
                              system_matrix.add(dofs_neighbor[i],
                                                dofs[j],
                                                ui_ve_matrix(i, j));
                              system_matrix.add(dofs_neighbor[i],
                                                dofs_neighbor[j],
                                                ue_ve_matrix(i, j));
                            }
                      }

                    // We do not need to consider a case (d), as those
                    // faces are treated 'from the other side within
                    // case (b).
                  }
              }
          }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(dofs[i], dofs[j], ui_vi_matrix(i, j));

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          right_hand_side(dofs[i]) += cell_vector(i);
      }
  }


  // @sect3{Solver}
  //
  // For this simple problem we use the simple Richardson iteration again. The
  // solver is completely unaffected by our anisotropic changes.
  template <int dim>
  void DGMethod<dim>::solve(Vector<double> &solution)
  {
    SolverControl                    solver_control(1000, 1e-12, false, false);
    SolverRichardson<Vector<double>> solver(solver_control);

    PreconditionBlockSSOR<SparseMatrix<double>> preconditioner;

    preconditioner.initialize(system_matrix, fe.n_dofs_per_cell());

    solver.solve(system_matrix, solution, right_hand_side, preconditioner);
  }


  // @sect3{Refinement}
  //
  // We refine the grid according to the same simple refinement criterion used
  // in step-12, namely an approximation to the gradient of the solution.
  template <int dim>
  void DGMethod<dim>::refine_grid()
  {
    Vector<float> gradient_indicator(triangulation.n_active_cells());

    // We approximate the gradient,
    DerivativeApproximation::approximate_gradient(mapping,
                                                  dof_handler,
                                                  solution2,
                                                  gradient_indicator);

    // and scale it to obtain an error indicator.
    for (const auto &cell : triangulation.active_cell_iterators())
      gradient_indicator[cell->active_cell_index()] *=
        std::pow(cell->diameter(), 1 + 1.0 * dim / 2);
    // Then we use this indicator to flag the 30 percent of the cells with
    // highest error indicator to be refined.
    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    gradient_indicator,
                                                    0.3,
                                                    0.1);
    // Now the refinement flags are set for those cells with a large error
    // indicator. If nothing is done to change this, those cells will be
    // refined isotropically. If the @p anisotropic flag given to this
    // function is set, we now call the set_anisotropic_flags() function,
    // which uses the jump indicator to reset some of the refinement flags to
    // anisotropic refinement.
    if (anisotropic)
      set_anisotropic_flags();
    // Now execute the refinement considering anisotropic as well as isotropic
    // refinement flags.
    triangulation.execute_coarsening_and_refinement();
  }

  // Once an error indicator has been evaluated and the cells with largest
  // error are flagged for refinement we want to loop over the flagged cells
  // again to decide whether they need isotropic refinement or whether
  // anisotropic refinement is more appropriate. This is the anisotropic jump
  // indicator explained in the introduction.
  template <int dim>
  void DGMethod<dim>::set_anisotropic_flags()
  {
    // We want to evaluate the jump over faces of the flagged cells, so we
    // need some objects to evaluate values of the solution on faces.
    UpdateFlags face_update_flags =
      UpdateFlags(update_values | update_JxW_values);

    FEFaceValues<dim>    fe_v_face(mapping,
                                fe,
                                face_quadrature,
                                face_update_flags);
    FESubfaceValues<dim> fe_v_subface(mapping,
                                      fe,
                                      face_quadrature,
                                      face_update_flags);
    FEFaceValues<dim>    fe_v_face_neighbor(mapping,
                                         fe,
                                         face_quadrature,
                                         update_values);

    // Now we need to loop over all active cells.
    for (const auto &cell : dof_handler.active_cell_iterators())
      // We only need to consider cells which are flagged for refinement.
      if (cell->refine_flag_set())
        {
          std::array<double, dim> jump_in_coordinate_direction      = {};
          std::array<double, dim> face_area_in_coordinate_direction = {};

          for (const auto face_no : cell->face_indices())
            {
              const auto face = cell->face(face_no);

              if (!face->at_boundary())
                {
                  Assert(cell->neighbor(face_no).state() ==
                           IteratorState::valid,
                         ExcInternalError());
                  const auto neighbor = cell->neighbor(face_no);

                  std::vector<double> u(fe_v_face.n_quadrature_points);
                  std::vector<double> u_neighbor(fe_v_face.n_quadrature_points);

                  // The four cases of different neighbor relations seen in
                  // the assembly routines are repeated much in the same way
                  // here.
                  if (face->has_children())
                    {
                      // The neighbor is refined.  First we store the
                      // information, which of the neighbor's faces points in
                      // the direction of our current cell. This property is
                      // inherited to the children.
                      unsigned int neighbor2 = cell->neighbor_face_no(face_no);
                      // Now we loop over all subfaces,
                      for (unsigned int subface_no = 0;
                           subface_no < face->n_active_descendants();
                           ++subface_no)
                        {
                          // get an iterator pointing to the cell behind the
                          // present subface...
                          const auto neighbor_child =
                            cell->neighbor_child_on_subface(face_no,
                                                            subface_no);
                          Assert(!neighbor_child->has_children(),
                                 ExcInternalError());
                          // ... and reinit the respective FEFaceValues and
                          // FESubFaceValues objects.
                          fe_v_subface.reinit(cell, face_no, subface_no);
                          fe_v_face_neighbor.reinit(neighbor_child, neighbor2);
                          // We obtain the function values
                          fe_v_subface.get_function_values(solution2, u);
                          fe_v_face_neighbor.get_function_values(solution2,
                                                                 u_neighbor);
                          // as well as the quadrature weights, multiplied by
                          // the Jacobian determinant.
                          const std::vector<double> &JxW =
                            fe_v_subface.get_JxW_values();
                          // Now we loop over all quadrature points
                          for (unsigned int x = 0;
                               x < fe_v_subface.n_quadrature_points;
                               ++x)
                            {
                              // and integrate the absolute value of the jump
                              // of the solution, i.e. the absolute value of
                              // the difference between the function value
                              // seen from the current cell and the
                              // neighboring cell, respectively. We know, that
                              // the first two faces are orthogonal to the
                              // first coordinate direction on the unit cell,
                              // the second two faces are orthogonal to the
                              // second coordinate direction and so on, so we
                              // accumulate these values into vectors with
                              // <code>dim</code> components.
                              jump_in_coordinate_direction[face_no / 2] +=
                                std::abs(u[x] - u_neighbor[x]) * JxW[x];
                              // We also sum up the scaled weights to obtain
                              // the measure of the face.
                              face_area_in_coordinate_direction[face_no / 2] +=
                                JxW[x];
                            }
                        }
                    }
                  else
                    {
                      if (!cell->neighbor_is_coarser(face_no))
                        {
                          // Our current cell and the neighbor have the same
                          // refinement along the face under
                          // consideration. Apart from that, we do much the
                          // same as with one of the subcells in the above
                          // case.
                          unsigned int neighbor2 =
                            cell->neighbor_of_neighbor(face_no);

                          fe_v_face.reinit(cell, face_no);
                          fe_v_face_neighbor.reinit(neighbor, neighbor2);

                          fe_v_face.get_function_values(solution2, u);
                          fe_v_face_neighbor.get_function_values(solution2,
                                                                 u_neighbor);

                          const std::vector<double> &JxW =
                            fe_v_face.get_JxW_values();

                          for (unsigned int x = 0;
                               x < fe_v_face.n_quadrature_points;
                               ++x)
                            {
                              jump_in_coordinate_direction[face_no / 2] +=
                                std::abs(u[x] - u_neighbor[x]) * JxW[x];
                              face_area_in_coordinate_direction[face_no / 2] +=
                                JxW[x];
                            }
                        }
                      else // i.e. neighbor is coarser than cell
                        {
                          // Now the neighbor is actually coarser. This case
                          // is new, in that it did not occur in the assembly
                          // routine. Here, we have to consider it, but this
                          // is not overly complicated. We simply use the @p
                          // neighbor_of_coarser_neighbor function, which
                          // again takes care of anisotropic refinement and
                          // non-standard face orientation by itself.
                          std::pair<unsigned int, unsigned int>
                            neighbor_face_subface =
                              cell->neighbor_of_coarser_neighbor(face_no);
                          Assert(neighbor_face_subface.first < cell->n_faces(),
                                 ExcInternalError());
                          Assert(neighbor_face_subface.second <
                                   neighbor->face(neighbor_face_subface.first)
                                     ->n_active_descendants(),
                                 ExcInternalError());
                          Assert(neighbor->neighbor_child_on_subface(
                                   neighbor_face_subface.first,
                                   neighbor_face_subface.second) == cell,
                                 ExcInternalError());

                          fe_v_face.reinit(cell, face_no);
                          fe_v_subface.reinit(neighbor,
                                              neighbor_face_subface.first,
                                              neighbor_face_subface.second);

                          fe_v_face.get_function_values(solution2, u);
                          fe_v_subface.get_function_values(solution2,
                                                           u_neighbor);

                          const std::vector<double> &JxW =
                            fe_v_face.get_JxW_values();

                          for (unsigned int x = 0;
                               x < fe_v_face.n_quadrature_points;
                               ++x)
                            {
                              jump_in_coordinate_direction[face_no / 2] +=
                                std::abs(u[x] - u_neighbor[x]) * JxW[x];
                              face_area_in_coordinate_direction[face_no / 2] +=
                                JxW[x];
                            }
                        }
                    }
                }
            }
          // Now we analyze the size of the mean jumps, which we get dividing
          // the jumps by the measure of the respective faces.
          std::array<double, dim> average_jumps;
          double                  sum_of_average_jumps = 0.;
          for (unsigned int i = 0; i < dim; ++i)
            {
              average_jumps[i] = jump_in_coordinate_direction[i] /
                                 face_area_in_coordinate_direction[i];
              sum_of_average_jumps += average_jumps[i];
            }

          // Now we loop over the <code>dim</code> coordinate directions of
          // the unit cell and compare the average jump over the faces
          // orthogonal to that direction with the average jumps over faces
          // orthogonal to the remaining direction(s). If the first is larger
          // than the latter by a given factor, we refine only along hat
          // axis. Otherwise we leave the refinement flag unchanged, resulting
          // in isotropic refinement.
          for (unsigned int i = 0; i < dim; ++i)
            if (average_jumps[i] > anisotropic_threshold_ratio *
                                     (sum_of_average_jumps - average_jumps[i]))
              cell->set_refine_flag(RefinementCase<dim>::cut_axis(i));
        }
  }

  // @sect3{The Rest}
  //
  // The remaining part of the program very much follows the scheme of
  // previous tutorial programs. We output the mesh in VTU format (just
  // as we did in step-1, for example), and the visualization output
  // in VTU format as we almost always do.
  template <int dim>
  void DGMethod<dim>::output_results(const unsigned int cycle) const
  {
    std::string refine_type;
    if (anisotropic)
      refine_type = ".aniso";
    else
      refine_type = ".iso";

    {
      const std::string filename =
        "grid-" + std::to_string(cycle) + refine_type + ".svg";
      std::cout << "   Writing grid to <" << filename << ">..." << std::endl;
      std::ofstream svg_output(filename);

      GridOut grid_out;
      grid_out.write_svg(triangulation, svg_output);
    }

    {
      const std::string filename =
        "sol-" + std::to_string(cycle) + refine_type + ".vtu";
      std::cout << "   Writing solution to <" << filename << ">..."
                << std::endl;
      std::ofstream gnuplot_output(filename);

      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution2, "u");

      data_out.build_patches(degree);

      data_out.write_vtu(gnuplot_output);
    }
  }



  template <int dim>
  void DGMethod<dim>::run()
  {
    for (unsigned int cycle = 0; cycle < 6; ++cycle)
      {
        std::cout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            // Create the rectangular domain.
            Point<dim> p1, p2;
            p1[0] = -1.;
            for (unsigned int i = 0; i < dim; ++i)
              p2[i] = 1.;
            // Adjust the number of cells in different directions to obtain
            // completely isotropic cells for the original mesh.
            std::vector<unsigned int> repetitions(dim, 1);
            repetitions[0] = 2;
            GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                      repetitions,
                                                      p1,
                                                      p2);

            triangulation.refine_global(5 - dim);
          }
        else
          refine_grid();


        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;

        setup_system();

        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                  << std::endl;

        Timer assemble_timer;
        assemble_system();
        std::cout << "   Time of assemble_system: " << assemble_timer.cpu_time()
                  << std::endl;
        solve(solution2);

        output_results(cycle);

        std::cout << std::endl;
      }
  }
} // namespace Step30



int main()
{
  try
    {
      using namespace Step30;

      // If you want to run the program in 3d, simply change the following
      // line to <code>const unsigned int dim = 3;</code>.
      const unsigned int dim = 2;

      {
        // First, we perform a run with isotropic refinement.
        std::cout << "Performing a " << dim
                  << "D run with isotropic refinement..." << std::endl
                  << "------------------------------------------------"
                  << std::endl;
        DGMethod<dim> dgmethod_iso(false);
        dgmethod_iso.run();
      }

      {
        // Now we do a second run, this time with anisotropic refinement.
        std::cout << std::endl
                  << "Performing a " << dim
                  << "D run with anisotropic refinement..." << std::endl
                  << "--------------------------------------------------"
                  << std::endl;
        DGMethod<dim> dgmethod_aniso(true);
        dgmethod_aniso.run();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };

  return 0;
}
