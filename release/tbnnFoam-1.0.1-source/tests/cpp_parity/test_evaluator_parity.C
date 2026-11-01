#include "fvCFD.H"
#include "tbnnEvaluator.H"
#include <iostream>
#include <iomanip>
#include <vector>

void printCase(
    const Foam::tensor& gradU,
    const Foam::scalar omega,
    const Foam::tbnnEvaluator& evaluator,
    bool isLast
)
{
    Foam::scalar I[5];
    Foam::tensor T[10];
    Foam::tbnnEvaluator::calcBasis(gradU, omega, I, T);

    Foam::scalar g[10];
    evaluator.predict_g(I, g);

    Foam::symmTensor a = Foam::tbnnEvaluator::calcAnisotropy(g, T);

    std::cout << "    {\n";
    std::cout << "      \"omega\": " << omega << ",\n";
    std::cout << "      \"gradU\": [\n";
    std::cout << "        [" << gradU.xx() << ", " << gradU.xy() << ", " << gradU.xz() << "],\n";
    std::cout << "        [" << gradU.yx() << ", " << gradU.yy() << ", " << gradU.yz() << "],\n";
    std::cout << "        [" << gradU.zx() << ", " << gradU.zy() << ", " << gradU.zz() << "]\n";
    std::cout << "      ],\n";

    std::cout << "      \"invariants\": [";
    for (int i = 0; i < 5; ++i)
    {
        std::cout << I[i] << (i < 4 ? ", " : "");
    }
    std::cout << "],\n";

    std::cout << "      \"basis\": [\n";
    for (int n = 0; n < 10; ++n)
    {
        std::cout << "        [\n";
        std::cout << "          [" << T[n].xx() << ", " << T[n].xy() << ", " << T[n].xz() << "],\n";
        std::cout << "          [" << T[n].yx() << ", " << T[n].yy() << ", " << T[n].yz() << "],\n";
        std::cout << "          [" << T[n].zx() << ", " << T[n].zy() << ", " << T[n].zz() << "]\n";
        std::cout << "        ]" << (n < 9 ? ",\n" : "\n");
    }
    std::cout << "      ],\n";

    std::cout << "      \"g\": [";
    for (int i = 0; i < 10; ++i)
    {
        std::cout << g[i] << (i < 9 ? ", " : "");
    }
    std::cout << "],\n";

    std::cout << "      \"a\": [\n";
    std::cout << "        [" << a.xx() << ", " << a.xy() << ", " << a.xz() << "],\n";
    std::cout << "        [" << a.xy() << ", " << a.yy() << ", " << a.yz() << "],\n";
    std::cout << "        [" << a.xz() << ", " << a.yz() << ", " << a.zz() << "]\n";
    std::cout << "      ]\n";
    std::cout << "    }" << (isLast ? "\n" : ",\n");
}

int main(int argc, char *argv[])
{
    std::cout << std::setprecision(16);

    Foam::fileName weightsFile = "export_weights/weights.json";
    if (argc > 1)
    {
        weightsFile = argv[1];
    }

    Foam::tbnnEvaluator evaluator;
    if (!evaluator.loadWeights(weightsFile))
    {
        std::cerr << "{\"status\": \"error\", \"message\": \"Failed to load weights\"}\n";
        return 1;
    }

    std::cout << "{\n";
    std::cout << "  \"status\": \"success\",\n";
    std::cout << "  \"cases\": [\n";

    if (argc >= 12)
    {
        Foam::tensor gradU
        (
            std::stod(argv[2]), std::stod(argv[3]), std::stod(argv[4]),
            std::stod(argv[5]), std::stod(argv[6]), std::stod(argv[7]),
            std::stod(argv[8]), std::stod(argv[9]), std::stod(argv[10])
        );
        Foam::scalar omega = std::stod(argv[11]);
        printCase(gradU, omega, evaluator, true);
    }
    else
    {
        std::vector<std::pair<Foam::tensor, Foam::scalar>> testCases;

        testCases.push_back
        ({
            Foam::tensor(0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            20.0
        });

        testCases.push_back
        ({
            Foam::tensor(0.5, 1.2, -0.3, -0.8, -0.2, 0.4, 0.1, -0.5, -0.3),
            15.0
        });

        testCases.push_back
        ({
            Foam::tensor(2.0, 25.0, -5.0, 5.0, -1.0, 8.0, -3.0, -2.0, -1.0),
            5.0
        });

        for (size_t i = 0; i < testCases.size(); ++i)
        {
            printCase(testCases[i].first, testCases[i].second, evaluator, i == testCases.size() - 1);
        }
    }

    std::cout << "  ]\n";
    std::cout << "}\n";

    return 0;
}
