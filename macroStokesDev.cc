/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 */


#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/fe/fe_macroStokes.h>
#include <deal.II/fe/fe_bdm.h>
#include <deal.II/fe/fe_poly_tensor.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/function.h>
#include <deal.II/base/job_identifier.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/geometry_info.h>


#include <vector>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

class macroStokesTest
{

public:
  macroStokesTest();
  void run();

private:
  void build_grid();
  void setup_system();
  void project();
  void output_results() const;
  void plot_basis_function();  

  Triangulation<2> triangulation;
  FE_MacroStokes<2> fe;
  DoFHandler<2> dof_handler;
  ConstraintMatrix constraints;


  Vector<double> solution;
  
};

class TestFunction : public Function<2>
{
public:
  TestFunction() : Function<2> (2) {}

  virtual void vector_value(const Point<2> &p,
			    Vector<double> &value ) const;

};

void
TestFunction::vector_value(const Point<2> &p,
			     Vector<double> &values) const
{
  values(0) = p[0]*p[0];
  values(1) = p[1]*p[1];
}


macroStokesTest::macroStokesTest ()
  :
  fe(2),
  dof_handler(triangulation)
{}


void macroStokesTest::build_grid()
{
  GridGenerator::hyper_cube (triangulation,0,1);
  triangulation.refine_global(1);
}

void macroStokesTest::setup_system()
{
  dof_handler.distribute_dofs(fe);

  solution.reinit (dof_handler.n_dofs());

  constraints.clear();
  DoFTools::make_hanging_node_constraints (dof_handler,
					   constraints);
  constraints.close();

}

void macroStokesTest::project()
{
  const QGauss<2> quadrature_formula(3);
  
  TestFunction test_function;

  VectorTools::project(dof_handler,
		       constraints,
                       quadrature_formula,
		       test_function,
		       solution);
}

void macroStokesTest::plot_basis_function()
{

solution[0] = 1;

/*
for (unsigned int j = 0; j < solution.size(); ++j)
  std::cout << "Solution " << j << " = " << solution[j] << std::endl;
*/

}

void macroStokesTest::output_results() const
{
  std::vector<std::string> solution_names(2,"u");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  interpretation (2, DataComponentInterpretation::component_is_part_of_vector);

  DataOut<2> data_out;
  data_out.add_data_vector (dof_handler, solution, solution_names, interpretation);

  data_out.build_patches (3);

  std::ofstream output("solution.vtu");
  data_out.write_vtu (output);
}

void macroStokesTest::run()
{
  build_grid();
  setup_system();
//  plot_basis_function();
  project();
  output_results();
}


template <int dim>
void plot()
{

  std::cout << GeometryInfo<dim>::vertices_per_cell << std::endl;
}

template <int dim,int spacedim>
void 
supportPointTests(const FiniteElement<dim,spacedim> &fe)
{
  const std::vector<Point<dim> > &points = fe.get_generalized_support_points();
  for (unsigned int i = 0; i < points.size(); ++i)
  {
   std::cout << "Point " << i << " " << points[i] << std::endl;
  }

}


int main()
{
  macroStokesTest projection_test;
  projection_test.run();

std::cout << "ran" << std::endl;

  return 0;
}
