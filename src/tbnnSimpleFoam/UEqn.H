{
    const volTensorField gradU(fvc::grad(U));
    const volScalarField k(turbulence->k());
    const volScalarField omega(turbulence->omega());
    const volScalarField nut(turbulence->nut());

    scalar I[5];
    tensor T[10];
    scalar g[10];

    const scalar relaxFactor = 0.3;

    forAll(mesh.C(), cellI)
    {
        tbnnEvaluator::calcBasis
        (
            gradU[cellI],
            omega[cellI],
            I,
            T
        );

        tbnn.predict_g(I, g);

        const symmTensor aNew = tbnnEvaluator::calcAnisotropy(g, T);

        a_TBNN[cellI] = relaxFactor * aNew + (1.0 - relaxFactor) * a_TBNN[cellI];

        const symmTensor S = symm(gradU[cellI]);

        const symmTensor tauNew = 2.0 * k[cellI] * a_TBNN[cellI] + 2.0 * nut[cellI] * S;
        tauExtra[cellI] = relaxFactor * tauNew + (1.0 - relaxFactor) * tauExtra[cellI];
    }

    a_TBNN.correctBoundaryConditions();
    tauExtra.correctBoundaryConditions();
}

MRF.correctBoundaryVelocity(U);

tmp<fvVectorMatrix> tUEqn
(
    fvm::div(phi, U)
  + MRF.DDt(U)
  - fvm::laplacian(turbulence->nuEff(), U)
  + fvc::div(tauExtra)
 ==
    fvModels.source(U)
);
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

fvConstraints.constrain(UEqn);

if (simple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    fvConstraints.constrain(U);
}
