#include <stdio.h>
//#include <crtdbg.h>
//#include <Windows.h>
#include <memory>
#include <vector>

#include "GpDataset.h"
#include "AnalyticApproximation.h"
#include "SmmcOptions.h"
#include "SmoothedModelCheker.h"
#include "ProbitRegressionPosterior.h"

#include <gsl/gsl_blas.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_ieee_utils.h>

#include "DebugLogMatrix.h"

using namespace std;


void test() {
	double beta = 2.0;
	const int ncolumns = 1;
	double x_points[][ncolumns] = { {0.005},
	{0.015263157894736843} ,
	{0.025526315789473685 },
	{0.035789473684210524},
	{0.046052631578947366},
	{0.05631578947368421},
	{0.06657894736842104},
	{0.07684210526315788},
	{0.08710526315789471},
	{0.09736842105263155},
	{0.10763157894736838},
	{0.11789473684210522},
	{0.12815789473684205},
	{0.1384210526315789},
	{0.14868421052631572},
	{0.15894736842105256},
	{0.1692105263157894},
	{0.17947368421052623},
	{0.18973684210526306},
	{0.1999999999999999} };
	// набор из письма
	double y_points[] = { 0.0,
0.0,
0.0,
0.0,
0.0,
0.0,
0.1,
0.15,
0.2,
0.25,
0.25,
0.15,
0.0,
0.0,
0.0,
0.05,
0.0,
0.0,
0.0,
0.0 };

	double xt_points[][ncolumns] = { {0.005},
	{0.015263157894736843},
	{0.025526315789473685},
	{0.035789473684210524},
	{0.046052631578947366},
	{0.05631578947368421},
	{0.06657894736842104},
	{0.07684210526315788},
	{0.08710526315789471},
	{0.09736842105263155},
	{0.10763157894736838},
	{0.11789473684210522},
	{0.12815789473684205},
	{0.1384210526315789},
	{0.14868421052631572},
	{0.15894736842105256},
	{0.1692105263157894},
	{0.17947368421052623},
	{0.18973684210526306},
	{0.192},
	{0.194},
	{0.196},
	{0.198},
	{0.1999999999999999}
	};

	vector<vector<double> > x(1), *xt;
	vector<double> y;
	x.resize(sizeof(x_points) / sizeof(x_points[0]));
	y.resize(sizeof(y_points) / sizeof(y_points[0]));
	xt = new std::vector<std::vector<double> >();
	xt->resize(sizeof(xt_points) / sizeof(xt_points[0]));
	for (size_t i = 0; i < x.size(); i++) { 
		x[i].resize(ncolumns); 
		for (size_t j = 0; j < ncolumns; j++) {
			x[i][j] = x_points[i][j];
		}
	};
	for (size_t i = 0; i < xt->size(); i++) { 
		(*xt)[i].resize(ncolumns);
		for (size_t j = 0; j < ncolumns; j++) {
			(*xt)[i][j] = xt_points[i][j];
		}
	};
	copy(&y_points[0], &y_points[y.size()], y.begin());

	shared_ptr<GpDataset> dataset = make_shared<GpDataset>(x, y);
	shared_ptr<SmmcOptions> options = make_shared<SmmcOptions>();
	shared_ptr<SmoothedModelCheker> mc = make_shared<SmoothedModelCheker>();
	std::vector<std::shared_ptr<Parameter> > params;
	shared_ptr<AnalyticApproximation> approx = mc->getAnalyticApproximation(dataset, params, options);
	std::shared_ptr<std::vector<std::vector<double> > > ptr_xt(xt);
	shared_ptr<ProbitRegressionPosterior> result = approx->getValuesAt(ptr_xt);
	DebugLogMatrix::printMatrix("getClassProbabilities", __LINE__, result->getClassProbabilities());
	DebugLogMatrix::printMatrix("getLowerBound(beta)", __LINE__, result->getLowerBound(beta));
	DebugLogMatrix::printMatrix("getUpperBound(beta)", __LINE__, result->getUpperBound(beta));
}

int main(int argc, char* argv[]) {
	//const gsl_multimin_fminimizer_type *T = gsl_multimin_fminimizer_nmsimplex;
	//gsl_multimin_fminimizer *s = NULL;
	//s = gsl_multimin_fminimizer_alloc(T, 2);
	//gsl_multimin_fminimizer_free(s);
	test();
	//	_CrtDumpMemoryLeaks();
	//	::system("pause");
	return 0;
}

