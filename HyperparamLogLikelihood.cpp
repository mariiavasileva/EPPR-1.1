#include "HyperparamLogLikelihood.h"
#include <cmath>
#include <limits>
#include "AbstractGP.h"
#include "KernelFunction.h"
#include <gsl/gsl_math.h>
HyperparamLogLikelihood::HyperparamLogLikelihood(std::shared_ptr<AbstractGP> gp, bool logspace)
{
	this->gp = gp;
	this->logspace = logspace;
}


HyperparamLogLikelihood::~HyperparamLogLikelihood()
{
}


double HyperparamLogLikelihood::getValueAt(std::shared_ptr<std::vector<double> > point)
{
	std::shared_ptr<std::vector<double> > vec = std::make_shared<std::vector<double> >(point->size());
	std::copy(point->begin(), point->end(), vec->begin());

	if (logspace)
		for (size_t i = 0; i < vec->size(); i++)
			(*vec)[i] = exp((*vec)[i]);
	gp->getKernel()->setHyperarameters(*vec.get());
	double lik = GSL_NAN;//-std::numeric_limits<double>::infinity();
	try {
		lik = gp->getMarginalLikelihood();
	}
	catch (char* ) {
		// System.out.println(e.getMessage());
	}
	return lik;
}


std::shared_ptr<std::vector<double> > HyperparamLogLikelihood::getGradientAt(std::shared_ptr<std::vector<double> > point)
{
	std::shared_ptr<std::vector<double> > vec = std::make_shared<std::vector<double> >(point->size());
	std::copy(point->begin(), point->end(), vec->begin());
	if (logspace)
		for (size_t i = 0; i < vec->size(); i++)
			(*vec)[i]= exp((*vec)[i]);
	gp->getKernel()->setHyperarameters(*vec.get());
	try {
		std::shared_ptr<std::vector<double> > grad = gp->getMarginalLikelihoodGradient();
		return grad;
	}
	catch (char e) {
		throw (e);
	}

}

