/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2026 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Application
    tbnnFoam

Description
    Steady-state, incompressible RANS solver coupled in-situ with LibTorch
    (PyTorch C++ API) executing a Tensor Basis Neural Network (TBNN) closure.
    
    The Reynolds stress anisotropy tensor is predicted via Pope's 10-tensor 
    integrity basis expansion:
        a_ij = sum_{n=1}^{10} g^(n)(lambda_1..5) * T_ij^(n)
        
    Numerical stability is guaranteed via semi-implicit eddy-viscosity splitting:
        div(tauR) = - fvm::laplacian(nuEff, U) + fvc::div(tauDelta)
        tauDelta  = alpha * (tauR_TBNN - tauR_Boussinesq)
\*---------------------------------------------------------------------------*/

// 1. LibTorch must be included FIRST before OpenFOAM headers to avoid macro collisions
#include <torch/script.h>

// 2. OpenFOAM Core Headers
#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

    // -------------------------------------------------------------------------
    // Load Pre-trained TorchScript TBNN Model
    // -------------------------------------------------------------------------
    torch::jit::script::Module tbnnModule;
    fileName modelPath = mesh.time().path()/"constant"/"tbnn_model.pt";
    
    Info<< "\n--> [TBNN-OpenFOAM] Loading TorchScript model from: " << modelPath << endl;
    try
    {
        tbnnModule = torch::jit::load(modelPath.c_str());
        tbnnModule.eval();
        Info<< "--> [TBNN-OpenFOAM] Model loaded successfully onto CPU device.\n" << endl;
    }
    catch (const c10::Error& e)
    {
        FatalErrorInFunction
            << "Failed to load TorchScript model: " << modelPath << nl
            << "Details: " << e.what() << nl
            << "Please run 'python3 tbnn_export.py' to generate 'constant/tbnn_model.pt'"
            << exit(FatalError);
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting SIMPLE time loop\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- 1. Update Baseline Turbulence Model (k, epsilon/omega, nut)
        turbulence->correct();
        volScalarField k = max(turbulence->k(), dimensionedScalar("kFloor", dimArea/sqr(dimTime), 1e-8));
        volScalarField nut = max(turbulence->nut(), dimensionedScalar("nutFloor", dimArea/dimTime, 1e-8));
        volScalarField epsilon = max(turbulence->epsilon(), dimensionedScalar("epsFloor", dimArea/pow3(dimTime), 1e-8));

        // Calculate turbulence timescale tau = k / epsilon (bounded to prevent near-wall singularity)
        volScalarField tau = min(k / epsilon, dimensionedScalar("tauMax", dimTime, 5.0));
        
        // --- 2. Calculate Mean Strain and Rotation Tensors
        volTensorField gradU(fvc::grad(U));
        volSymmTensorField S(symm(gradU));
        volTensorField Omega(skew(gradU));

        // --- 3. Evaluate TBNN Invariants and Tensor Bases per Cell
        const label nCells = mesh.nCells();
        
        // Allocate LibTorch Tensors for batch evaluation
        auto invariantsTensor = torch::zeros({nCells, 5}, torch::kFloat32);
        auto tensorBasesTensor = torch::zeros({nCells, 10, 3, 3}, torch::kFloat32);

        auto invAccessor = invariantsTensor.accessor<float, 2>();
        auto basisAccessor = tensorBasesTensor.accessor<float, 4>();

        for (label cellI = 0; cellI < nCells; ++cellI)
        {
            scalar tVal = tau[cellI];
            symmTensor sMat = tVal * S[cellI];
            tensor wMat = tVal * Omega[cellI];

            // Convert to 3x3 matrices
            tensor sTensor(sMat.xx(), sMat.xy(), sMat.xz(),
                           sMat.xy(), sMat.yy(), sMat.yz(),
                           sMat.xz(), sMat.yz(), sMat.zz());

            // Compute powers and products
            tensor s2 = sTensor & sTensor;
            tensor w2 = wMat & wMat;
            tensor sw = sTensor & wMat;
            tensor ws = wMat & sTensor;

            // Invariants lambda_1 and lambda_2
            scalar tr_s2 = tr(s2);
            scalar tr_w2 = tr(w2);
            scalar tr_s3 = tr(s2 & sTensor);
            scalar tr_w2s = tr(w2 & sTensor);
            scalar tr_w2s2 = tr(w2 & s2);

            // Numerical bounding on invariants
            invAccessor[cellI][0] = std::clamp(static_cast<float>(tr_s2), -50.0f, 50.0f);
            invAccessor[cellI][1] = std::clamp(static_cast<float>(tr_w2), -50.0f, 50.0f);
            invAccessor[cellI][2] = std::clamp(static_cast<float>(tr_s3), -50.0f, 50.0f);
            invAccessor[cellI][3] = std::clamp(static_cast<float>(tr_w2s), -50.0f, 50.0f);
            invAccessor[cellI][4] = std::clamp(static_cast<float>(tr_w2s2), -50.0f, 50.0f);

            // Basis 1: s
            tensor T1 = sTensor;
            // Basis 2: s*w - w*s
            tensor T2 = sw - ws;
            // Basis 3: s^2 - 1/3 * tr(s^2) * I
            tensor T3 = s2 - (1.0/3.0)*tr_s2*tensor::I;
            // Basis 4: w^2 - 1/3 * tr(w^2) * I
            tensor T4 = w2 - (1.0/3.0)*tr_w2*tensor::I;

            // Fill bases into accessor
            for (int r = 0; r < 3; ++r)
            {
                for (int c = 0; c < 3; ++c)
                {
                    basisAccessor[cellI][0][r][c] = static_cast<float>(T1[r*3 + c]);
                    basisAccessor[cellI][1][r][c] = static_cast<float>(T2[r*3 + c]);
                    basisAccessor[cellI][2][r][c] = static_cast<float>(T3[r*3 + c]);
                    basisAccessor[cellI][3][r][c] = static_cast<float>(T4[r*3 + c]);
                }
            }
        }

        // --- 4. Execute Neural Network Forward Pass in C++
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(invariantsTensor);
        inputs.push_back(tensorBasesTensor);

        at::Tensor outputAnisotropy = tbnnModule.forward(inputs).toTensor();
        auto anisoAccessor = outputAnisotropy.accessor<float, 3>();

        // Ramp-up factor: let baseline RANS converge (first 60 iterations), then smoothly introduce TBNN
        scalar alphaTBNN = min(1.0, max(0.0, (runTime.timeIndex() - 60.0) / 40.0));

        // --- 5. Update OpenFOAM Fields and Compute Stabilizing Stress Correction
        for (label cellI = 0; cellI < nCells; ++cellI)
        {
            // Lumley triangle realizability clipping: physical anisotropy is strictly bounded
            scalar axx = std::clamp(static_cast<double>(anisoAccessor[cellI][0][0]), -0.33, 0.66);
            scalar ayy = std::clamp(static_cast<double>(anisoAccessor[cellI][1][1]), -0.33, 0.66);
            scalar azz = std::clamp(static_cast<double>(anisoAccessor[cellI][2][2]), -0.33, 0.66);
            scalar axy = std::clamp(static_cast<double>(anisoAccessor[cellI][0][1]), -0.5, 0.5);
            scalar axz = std::clamp(static_cast<double>(anisoAccessor[cellI][0][2]), -0.5, 0.5);
            scalar ayz = std::clamp(static_cast<double>(anisoAccessor[cellI][1][2]), -0.5, 0.5);

            symmTensor aVal(axx, axy, axz, ayy, ayz, azz);
            aTBNN[cellI] = aVal;

            // Full Reynolds Stress: tauR = 2*k*(a + 1/3*I)
            scalar kVal = k[cellI];
            tauR[cellI] = 2.0 * kVal * (aVal + (1.0/3.0)*symmTensor::I);

            // Pure anisotropic Reynolds stress source: 2*k*a with smooth relaxation (alpha = 0.01)
            symmTensor tauDiff = 2.0 * kVal * aVal;
            tauDelta[cellI] = 0.99 * tauDelta[cellI] + 0.01 * alphaTBNN * tauDiff;
        }

        forAll(tauDelta.boundaryField(), patchI)
        {
            tauDelta.boundaryFieldRef()[patchI] == symmTensor::zero;
        }
        tauR.correctBoundaryConditions();
        aTBNN.correctBoundaryConditions();

        // --- 6. Momentum Predictor with Semi-Implicit Stress Splitting
        tmp<fvVectorMatrix> tUEqn
        (
            fvm::div(phi, U)
          + turbulence->divDevReff(U)
          + fvc::div(tauDelta) // Stabilizing non-linear correction
         ==
            fvOptions(U)
        );
        fvVectorMatrix& UEqn = tUEqn.ref();

        UEqn.relax();
        fvOptions.constrain(UEqn);

        if (simple.momentumPredictor())
        {
            solve(UEqn == -fvc::grad(p));
            fvOptions.correct(U);
        }

        // --- 7. Pressure Equation (SIMPLE Algorithm)
        {
            volScalarField rAU(1.0/UEqn.A());
            volVectorField HbyA(constrainHbyA(rAU*UEqn.H(), U, p));
            surfaceScalarField phiHbyA("phiHbyA", fvc::flux(HbyA));

            MRF.makeRelative(phiHbyA);
            adjustPhi(phiHbyA, U, p);

            tmp<volScalarField> rAtU(rAU);

            if (simple.consistent())
            {
                rAtU = 1.0/(1.0/rAU - UEqn.H1());
                phiHbyA +=
                    fvc::interpolate(rAtU() - rAU)*fvc::snGrad(p)*mesh.magSf();
                HbyA -= (rAU - rAtU())*fvc::grad(p);
            }

            // Non-orthogonal pressure corrector loop
            while (simple.correctNonOrthogonal())
            {
                fvScalarMatrix pEqn
                (
                    fvm::laplacian(rAtU(), p) == fvc::div(phiHbyA)
                );

                pEqn.setReference(pRefCell, pRefValue);

                pEqn.solve();

                if (simple.finalNonOrthogonalIter())
                {
                    phi = phiHbyA - pEqn.flux();
                }
            }

            #include "continuityErrs.H"

            // Explicitly relax pressure for momentum corrector
            p.relax();

            // Correct velocity
            U = HbyA - rAtU()*fvc::grad(p);
            U.correctBoundaryConditions();
            fvOptions.correct(U);
        }

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    Info<< "End of tbnnFoam execution\n" << endl;

    return 0;
}

// ************************************************************************* //
