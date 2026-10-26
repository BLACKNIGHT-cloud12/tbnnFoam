#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include <iostream>
#include <iomanip>

using namespace Foam;

// Infinity norm over every stored matrix coefficient accessible via the
// standard fvMatrix/lduMatrix interface.
static scalar matrixNorm(const fvVectorMatrix& M)
{
    scalar norm = 0.0;

    const scalarField& diagF = M.diag();
    forAll(diagF, i) { norm = max(norm, mag(diagF[i])); }

    const Field<vector>& src = M.source();
    forAll(src, i) { norm = max(norm, cmptMax(cmptMag(src[i]))); }

    if (M.hasUpper())
    {
        const scalarField& upperF = M.upper();
        forAll(upperF, i) { norm = max(norm, mag(upperF[i])); }
    }
    if (M.hasLower())
    {
        const scalarField& lowerF = M.lower();
        forAll(lowerF, i) { norm = max(norm, mag(lowerF[i])); }
    }

    const FieldField<Field, vector>& ic = M.internalCoeffs();
    forAll(ic, patchI)
    {
        const Field<vector>& pf = ic[patchI];
        forAll(pf, i) { norm = max(norm, cmptMax(cmptMag(pf[i]))); }
    }

    const FieldField<Field, vector>& bc = M.boundaryCoeffs();
    forAll(bc, patchI)
    {
        const Field<vector>& pf = bc[patchI];
        forAll(pf, i) { norm = max(norm, cmptMax(cmptMag(pf[i]))); }
    }

    return norm;
}

static scalar cellMax(const volSymmTensorField& f)
{
    scalar v = 0.0;
    forAll(f, cellI) { v = max(v, cmptMax(cmptMag(f[cellI]))); }
    return v;
}

int main(int argc, char *argv[])
{
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    std::cout << std::setprecision(16);

    IOdictionary testDict
    (
        IOobject
        (
            "tbnnMomentumTestDict",
            runTime.constant(),
            runTime,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
    const tensor G = testDict.get<tensor>("gradU");
    const vector U0 = testDict.get<vector>("U0");

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

    // Prescribe the exact linear velocity field U = U0 + G.x at cell centres
    // and boundary face centres, so fvc::grad(U) is exact on this mesh.
    forAll(mesh.C(), cellI)
    {
        U[cellI] = U0 + (G & mesh.C()[cellI]);
    }
    forAll(U.boundaryField(), patchI)
    {
        fvPatchField<vector>& pf = U.boundaryFieldRef()[patchI];
        pf == (U0 + (G & pf.patch().Cf()));
    }
    U.correctBoundaryConditions();

    turbulence->validate();
    turbulence->correct();

    volScalarField k(turbulence->k());
    volScalarField omega(turbulence->omega());
    volScalarField nut(turbulence->nut());
    volTensorField gradU(fvc::grad(U));
    volSymmTensorField S(symm(gradU));

    // Sanity 1: prescribed gradient reproduced exactly. Note OpenFOAM's
    // fvc::grad(U) convention: component (i,j) = dU_j/dx_i, i.e. the
    // transpose of the Jacobian G used to prescribe U = U0 + G.x.
    scalar gradErr = 0.0;
    scalar gradRef = SMALL;
    forAll(gradU, cellI)
    {
        gradErr = max(gradErr, cmptMax(cmptMag(gradU[cellI] - G.T())));
        gradRef = max(gradRef, cmptMax(cmptMag(G)));
    }
    const scalar gradRelErr = gradErr / gradRef;

    // Sanity 2: nut actually varies (guards the negative-control validity).
    scalar nutMin = VGREAT;
    scalar nutMax = 0.0;
    forAll(nut, cellI)
    {
        nutMin = min(nutMin, nut[cellI]);
        nutMax = max(nutMax, nut[cellI]);
    }
    const scalar nutRelVar = (nutMax > 0.0 ? (nutMax - nutMin)/nutMax : 0.0);

    // Boussinesq-consistent anisotropy a = -(nut/k)*S makes the TBNN stress
    // 2*k*a identical to the linear stress -2*nut*S, hence tauExtra ~= 0.
    const dimensionedScalar kSmall("kSmall", sqr(dimVelocity), 1e-8);
    volScalarField kSafe(max(k, kSmall));
    volSymmTensorField aBouss("aBouss", -(nut/kSafe)*S);
    volSymmTensorField tauDelta("tauDelta", 2.0*kSafe*aBouss + 2.0*nut*S);

    scalar tdNorm = cellMax(tauDelta);
    scalar tdRef = 0.0;
    forAll(S, cellI)
    {
        tdRef = max(tdRef, cmptMax(cmptMag(2.0*nut[cellI]*S[cellI])));
    }
    const scalar tdRel = (tdRef > 0.0 ? tdNorm/tdRef : 0.0);

    // The three momentum assemblies:
    //   ABase : baseline linear model (tauExtra absent)
    //   ANew  : fixed tbnnSimpleFoam coupling, identical to tbnnFoam.C
    //   AOld  : pre-fix tbnnSimpleFoam coupling (bare -laplacian(nuEff,U))
    fvVectorMatrix ABase(fvm::div(phi, U) + turbulence->divDevReff(U));
    fvVectorMatrix ANew
    (
        fvm::div(phi, U)
      + turbulence->divDevReff(U)
      + fvc::div(tauDelta)
    );
    fvVectorMatrix AOld
    (
        fvm::div(phi, U)
      - fvm::laplacian(turbulence->nuEff(), U)
      + fvc::div(tauDelta)
    );

    const scalar baseNorm = max(matrixNorm(ABase), SMALL);
    const scalar newDiffRel = matrixNorm(ANew - ABase)/baseNorm;
    const scalar oldDiffRel = matrixNorm(AOld - ABase)/baseNorm;

    // Load-bearing premise of the coupling: beyond the implicit laplacian,
    // divDevReff(U) injects the transpose divergence of the linear stress.
    // In fvMatrix form the explicit term enters the assembled equation as
    // +fvc::div(nuEff*dev2(T(gradU))), which for the incompressible field
    // used here (tr(gradU) = 0) equals +fvc::div(nuEff*gradU.T()), so that
    // laplacian + explicit transpose sum to the full linear stress
    // divergence div(2*nuEff*S) - the stress that tauExtra's 2*nut*S part
    // must cancel.
    fvVectorMatrix ALap(fvm::div(phi, U) - fvm::laplacian(turbulence->nuEff(), U));
    fvVectorMatrix DExp(ABase - ALap);
    volTensorField gradUT(gradU.T());
    volVectorField explicitExpected(fvc::div(turbulence->nuEff()*gradUT));
    // fvMatrix::source() is volume-integrated, so the explicit fvc field
    // enters the source scaled by the cell volume.
    scalar premiseDiff = 0.0;
    forAll(DExp.source(), cellI)
    {
        premiseDiff = max
        (
            premiseDiff,
            cmptMax(cmptMag(DExp.source()[cellI]/mesh.V()[cellI] - explicitExpected[cellI]))
        );
    }
    const scalar premiseRel = premiseDiff/max(baseNorm, SMALL);

    std::cout << "MPJ_BEGIN\n";
    std::cout << "{\n";
    std::cout << "  \"status\": \"success\",\n";
    std::cout << "  \"gradU_rel_error\": " << gradRelErr << ",\n";
    std::cout << "  \"nut_rel_variation\": " << nutRelVar << ",\n";
    std::cout << "  \"tauDelta_rel_norm\": " << tdRel << ",\n";
    std::cout << "  \"boussinesq_diff_rel\": " << newDiffRel << ",\n";
    std::cout << "  \"old_form_diff_rel\": " << oldDiffRel << ",\n";
    std::cout << "  \"premise_diff_rel\": " << premiseRel << ",\n";
    std::cout << "  \"base_norm\": " << baseNorm << "\n";
    std::cout << "}\n";
    std::cout << "MPJ_END\n";

    return 0;
}
