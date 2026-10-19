Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

#include "createPhi.H"

singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

Info<< "Creating TBNN Anisotropic Reynolds Stress field a_TBNN\n" << endl;
volSymmTensorField a_TBNN
(
    IOobject
    (
        "a_TBNN",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedSymmTensor("zero", dimless, symmTensor::zero)
);

Info<< "Creating TBNN Extra Stress field tauExtra\n" << endl;
volSymmTensorField tauExtra
(
    IOobject
    (
        "tauExtra",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedSymmTensor("zero", sqr(dimVelocity), symmTensor::zero)
);

tbnnEvaluator tbnn;
fileName weightsFile = runTime.constant()/"tbnn"/"weights.json";
if (!tbnn.loadWeights(weightsFile))
{
    Info<< "Looking in fallback export_weights/weights.json..." << endl;
    tbnn.loadWeights("export_weights/weights.json");
}

#include "createMRF.H"
#include "createFvOptions.H"
