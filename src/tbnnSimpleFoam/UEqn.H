// Update TBNN Anisotropic Reynolds Stress fields
{
    const volTensorField gradU(fvc::grad(U));
    const volScalarField k(turbulence->k());
    const volScalarField omega(turbulence->omega());
    const volScalarField nut(turbulence->nut());

    scalar I[5];
    tensor T[10];
    scalar g[10];

    const scalar relaxFactor = 0.3; // Relaxation for nonlinear stress stability

    forAll(mesh.C(), cellI)
    {
        tbnnEvaluator::computeBasisAndInvariants
        (
            gradU[cellI],
            omega[cellI],
            I,
            T
        );

        tbnn.predict_g(I, g);

        const symmTensor aNew = tbnnEvaluator::evaluateAnisotropy(g, T);
        
        // Under-relax anisotropy field
        a_TBNN[cellI] = relaxFactor * aNew + (1.0 - relaxFactor) * a_TBNN[cellI];

        // S symmetric strain tensor
        const symmTensor S = symm(gradU[cellI]);

        // Explicit correction term: tauExtra = 2*k*a + 2*nut*S
        // When subtracted from -laplacian(nuEff, U), this introduces the non-linear Reynolds stress
        const symmTensor tauNew = 2.0 * k[cellI] * a_TBNN[cellI] + 2.0 * nut[cellI] * S;
        tauExtra[cellI] = relaxFactor * tauNew + (1.0 - relaxFactor) * tauExtra[cellI];
    }

    a_TBNN.correctBoundaryConditions();
    tauExtra.correctBoundaryConditions();
}

// Assemble Momentum Predictor
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
