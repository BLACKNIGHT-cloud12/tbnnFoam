#include <torch/script.h>

#include "fvCFD.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "simpleControl.H"
#include "fvOptions.H"

int main(int argc, char *argv[])
{
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"

    simpleControl simple(mesh);

    #include "createFields.H"

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

    Info<< "\nStarting SIMPLE time loop\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;

        turbulence->correct();
        volScalarField k = max(turbulence->k(), dimensionedScalar("kFloor", dimArea/sqr(dimTime), 1e-8));
        volScalarField nut = max(turbulence->nut(), dimensionedScalar("nutFloor", dimArea/dimTime, 1e-8));
        volScalarField omega = max(turbulence->omega(), dimensionedScalar("omegaFloor", dimless/dimTime, 1e-4));

        volScalarField tau = min(1.0 / omega, dimensionedScalar("tauMax", dimTime, 5.0));

        volTensorField gradU(fvc::grad(U));
        volSymmTensorField S(symm(gradU));
        volTensorField Omega(skew(gradU));

        const label nCells = mesh.nCells();

        auto invariantsTensor = torch::zeros({nCells, 5}, torch::kFloat32);
        auto tensorBasesTensor = torch::zeros({nCells, 10, 3, 3}, torch::kFloat32);

        auto invAccessor = invariantsTensor.accessor<float, 2>();
        auto basisAccessor = tensorBasesTensor.accessor<float, 4>();

        for (label cellI = 0; cellI < nCells; ++cellI)
        {
            scalar tVal = tau[cellI];
            symmTensor sMat = tVal * S[cellI];
            tensor wMat = tVal * Omega[cellI];
            tensor wTensor = wMat;

            tensor sTensor(sMat.xx(), sMat.xy(), sMat.xz(),
                           sMat.xy(), sMat.yy(), sMat.yz(),
                           sMat.xz(), sMat.yz(), sMat.zz());

            tensor s2 = sTensor & sTensor;
            tensor w2 = wMat & wMat;
            tensor sw = sTensor & wMat;
            tensor ws = wMat & sTensor;

            scalar tr_s2 = tr(s2);
            scalar tr_w2 = tr(w2);
            scalar tr_s3 = tr(s2 & sTensor);
            scalar tr_w2s = tr(w2 & sTensor);
            scalar tr_w2s2 = tr(w2 & s2);

            invAccessor[cellI][0] = std::clamp(static_cast<float>(tr_s2), -50.0f, 50.0f);
            invAccessor[cellI][1] = std::clamp(static_cast<float>(tr_w2), -50.0f, 50.0f);
            invAccessor[cellI][2] = std::clamp(static_cast<float>(tr_s3), -50.0f, 50.0f);
            invAccessor[cellI][3] = std::clamp(static_cast<float>(tr_w2s), -50.0f, 50.0f);
            invAccessor[cellI][4] = std::clamp(static_cast<float>(tr_w2s2), -50.0f, 50.0f);

            tensor T1 = sTensor;
            tensor T2 = sw - ws;
            tensor T3 = s2 - (1.0/3.0)*tr_s2*tensor::I;
            tensor T4 = w2 - (1.0/3.0)*tr_w2*tensor::I;
            tensor T5 = (wTensor & s2) - (s2 & wTensor);
            tensor T6 = (w2 & sTensor) + (sTensor & w2) - (2.0/3.0)*tr_w2s*tensor::I;
            tensor T7 = ((wTensor & sTensor) & w2) - ((w2 & sTensor) & wTensor);
            tensor T8 = ((sTensor & wTensor) & s2) - ((s2 & wTensor) & sTensor);
            tensor T9 = (w2 & s2) + (s2 & w2) - (2.0/3.0)*tr_w2s2*tensor::I;
            tensor T10 = ((wTensor & s2) & w2) - ((w2 & s2) & wTensor);

            tensor allT[10] = {T1, T2, T3, T4, T5, T6, T7, T8, T9, T10};
            for (int b = 0; b < 10; ++b)
            {
                for (int r = 0; r < 3; ++r)
                {
                    for (int c = 0; c < 3; ++c)
                    {
                        basisAccessor[cellI][b][r][c] = static_cast<float>(allT[b][r*3 + c]);
                    }
                }
            }
        }

        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(invariantsTensor);
        inputs.push_back(tensorBasesTensor);

        at::Tensor outputAnisotropy = tbnnModule.forward(inputs).toTensor();
        auto anisoAccessor = outputAnisotropy.accessor<float, 3>();

        // TBNN source activation ramp: injected stress grows linearly from
        // zero at iteration 60 to full strength at iteration 100. Keyed to
        // the absolute time index, so this is only valid for runs starting
        // at iteration 0 - restarts reset the ramp. The standalone solver
        // (tbnnSimpleFoam) instead uses a fixed 0.3 blend factor.
        scalar alphaTBNN = min(1.0, max(0.0, (runTime.timeIndex() - 60.0) / 40.0));

        for (label cellI = 0; cellI < nCells; ++cellI)
        {
            symmTensor aRaw
            (
                anisoAccessor[cellI][0][0],
                0.5 * (anisoAccessor[cellI][0][1] + anisoAccessor[cellI][1][0]),
                0.5 * (anisoAccessor[cellI][0][2] + anisoAccessor[cellI][2][0]),
                anisoAccessor[cellI][1][1],
                0.5 * (anisoAccessor[cellI][1][2] + anisoAccessor[cellI][2][1]),
                anisoAccessor[cellI][2][2]
            );
            symmTensor aDev = dev(aRaw);

            scalar pVal = 0.5 * (sqr(aDev.xx()) + sqr(aDev.yy()) + sqr(aDev.zz())
                               + 2.0 * (sqr(aDev.xy()) + sqr(aDev.xz()) + sqr(aDev.yz())));
            scalar qVal = det(aDev);

            scalar scale = 1.0;
            if (pVal > 1e-14)
            {
                scalar mVal = 2.0 * std::sqrt(pVal / 3.0);
                scalar arg = std::clamp(3.0 * qVal / (pVal * mVal), -1.0, 1.0);
                scalar theta = (1.0 / 3.0) * std::acos(arg);
                scalar l1 = mVal * std::cos(theta);
                scalar l2 = mVal * std::cos(theta - 2.0 * M_PI / 3.0);
                scalar l3 = mVal * std::cos(theta + 2.0 * M_PI / 3.0);

                scalar lMax = max(l1, max(l2, l3));
                scalar lMin = min(l1, min(l2, l3));

                if (lMin < -1.0 / 3.0)
                {
                    scale = min(scale, (1.0 / 3.0) / std::abs(lMin));
                }
                if (lMax > 2.0 / 3.0)
                {
                    scale = min(scale, (2.0 / 3.0) / lMax);
                }
            }

            symmTensor aVal = scale * aDev;
            aTBNN[cellI] = aVal;

            scalar kVal = k[cellI];
            tauR[cellI] = 2.0 * kVal * (aVal + (1.0/3.0)*symmTensor::I);

            scalar nutVal = nut[cellI];
            symmTensor SVal = S[cellI];
            symmTensor tauDiff = 2.0 * kVal * aVal + 2.0 * nutVal * SVal;
            // Heavy EMA under-relaxation (gain 0.04) damps the explicit
            // source; unlike the ramp above it never fully deactivates, so
            // the injected stress approaches the instantaneous tauDiff only
            // asymptotically.
            tauDelta[cellI] = 0.96 * tauDelta[cellI] + 0.04 * alphaTBNN * tauDiff;
        }

        forAll(tauDelta.boundaryField(), patchI)
        {
            tauDelta.boundaryFieldRef()[patchI] == symmTensor::zero;
        }
        tauR.correctBoundaryConditions();
        aTBNN.correctBoundaryConditions();

        tmp<fvVectorMatrix> tUEqn
        (
            fvm::div(phi, U)
          + turbulence->divDevReff(U)
          + fvc::div(tauDelta)
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

            p.relax();

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
