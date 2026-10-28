#ifndef CREATE_FIELDS_H
#define CREATE_FIELDS_H

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

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, simple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());

singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

#include "initContinuityErrs.H"

Info<< "Creating TBNN Reynolds Stress and Anisotropy fields\n" << endl;

// READ_IF_PRESENT makes restarts continue the relaxed TBNN stress state
// instead of re-ramping from zero (zero-init on fresh runs).
volSymmTensorField aTBNN
(
    IOobject
    (
        "aTBNN",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedSymmTensor("aTBNN", dimless, symmTensor::zero)
);

volSymmTensorField tauR
(
    IOobject
    (
        "tauR",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedSymmTensor("tauR", dimArea/sqr(dimTime), symmTensor::zero)
);

volSymmTensorField tauDelta
(
    IOobject
    (
        "tauDelta",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::AUTO_WRITE
    ),
    mesh,
    dimensionedSymmTensor("tauDelta", dimArea/sqr(dimTime), symmTensor::zero)
);

#include "createMRF.H"
#include "createFvOptions.H"

#endif
