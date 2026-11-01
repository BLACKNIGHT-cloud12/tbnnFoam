#ifndef tbnnEvaluator_H
#define tbnnEvaluator_H

#include "fvCFD.H"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

namespace Foam
{

class tbnnEvaluator
{
private:
    bool initialized_;

    std::vector<scalar> inv_mean_;
    std::vector<scalar> inv_std_;

    std::vector<std::vector<scalar>> W0_, W1_, W2_, W3_;
    std::vector<scalar> b0_, b1_, b2_, b3_;

    static scalar relu(scalar x)
    {
        return x > 0.0 ? x : 0.0;
    }

    static bool parse1D(const std::string& src, const std::string& key, std::vector<scalar>& vec)
    {
        size_t pos = src.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        size_t openBracket = src.find('[', pos);
        if (openBracket == std::string::npos) return false;
        size_t closeBracket = src.find(']', openBracket);
        if (closeBracket == std::string::npos) return false;
        std::string sub = src.substr(openBracket + 1, closeBracket - openBracket - 1);
        std::stringstream ss(sub);
        std::string token;
        vec.clear();
        while (std::getline(ss, token, ','))
        {
            std::stringstream num(token);
            scalar v;
            if (num >> v) vec.push_back(v);
        }
        return !vec.empty();
    }

    static bool parse2D(const std::string& src, const std::string& key, std::vector<std::vector<scalar>>& mat)
    {
        size_t pos = src.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        size_t openOuter = src.find('[', pos);
        if (openOuter == std::string::npos) return false;
        int depth = 0;
        size_t closeOuter = std::string::npos;
        for (size_t i = openOuter; i < src.size(); ++i)
        {
            if (src[i] == '[') depth++;
            else if (src[i] == ']')
            {
                depth--;
                if (depth == 0)
                {
                    closeOuter = i;
                    break;
                }
            }
        }
        if (closeOuter == std::string::npos) return false;
        mat.clear();
        size_t curr = openOuter + 1;
        while (curr < closeOuter)
        {
            size_t openInner = src.find('[', curr);
            if (openInner == std::string::npos || openInner >= closeOuter) break;
            size_t closeInner = src.find(']', openInner);
            if (closeInner == std::string::npos || closeInner >= closeOuter) break;
            std::string rowStr = src.substr(openInner + 1, closeInner - openInner - 1);
            std::stringstream ss(rowStr);
            std::string token;
            std::vector<scalar> row;
            while (std::getline(ss, token, ','))
            {
                std::stringstream num(token);
                scalar v;
                if (num >> v) row.push_back(v);
            }
            if (!row.empty()) mat.push_back(row);
            curr = closeInner + 1;
        }
        return !mat.empty();
    }

public:
    tbnnEvaluator()
    :
        initialized_(false),
        inv_mean_(5, 0.0),
        inv_std_(5, 1.0)
    {}

    bool loadWeights(const fileName& jsonPath)
    {
        std::ifstream file(jsonPath.c_str());
        if (!file.is_open())
        {
            Info << "tbnnEvaluator: Cannot open weights file " << jsonPath << endl;
            return false;
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
        file.close();

        bool ok = true;
        ok &= parse2D(content, "0.weight", W0_);
        ok &= parse1D(content, "0.bias", b0_);
        ok &= parse2D(content, "2.weight", W1_);
        ok &= parse1D(content, "2.bias", b1_);
        ok &= parse2D(content, "4.weight", W2_);
        ok &= parse1D(content, "4.bias", b2_);
        ok &= parse2D(content, "6.weight", W3_);
        ok &= parse1D(content, "6.bias", b3_);
        parse1D(content, "inv_mean", inv_mean_);
        parse1D(content, "inv_std", inv_std_);

        if (!ok || W0_.empty() || W1_.empty() || W2_.empty() || W3_.empty()
            || W0_[0].size() != 5
            || W1_[0].size() != W0_.size()
            || W2_[0].size() != W1_.size()
            || W3_[0].size() != W2_.size()
            || W3_.size() != 10
            || b0_.size() != W0_.size()
            || b1_.size() != W1_.size()
            || b2_.size() != W2_.size()
            || b3_.size() != 10)
        {
            Info << "tbnnEvaluator: Error parsing weights from " << jsonPath << ". Invalid format or dimensions." << endl;
            initialized_ = false;
            return false;
        }

        initialized_ = true;
        Info << "tbnnEvaluator: Successfully loaded network architecture (5 -> "
             << W0_.size() << " -> " << W1_.size() << " -> " << W2_.size() << " -> 10)." << endl;
        return true;
    }

    void predict_g(const scalar I[5], scalar g[10]) const
    {
        if (!initialized_)
        {
            // Weights not loaded: fall back to g = (-1, 0, ..., 0), i.e.
            // a = -T1 = -tau*S. This reproduces a Boussinesq-like linear
            // response under the eddy-viscosity consistency nut ~ k/omega,
            // but it is a diagnostic placeholder, not the trained model -
            // loadWeights() failure is reported upstream and the case
            // should not be used for quantitative results.
            for (int i = 0; i < 10; ++i) g[i] = (i == 0 ? -1.0 : 0.0);
            return;
        }

        scalar x0[5];
        for (int i = 0; i < 5; ++i)
        {
            scalar mean = (inv_mean_.size() == 5 ? inv_mean_[i] : 0.0);
            scalar std = (inv_std_.size() == 5 ? inv_std_[i] : 1.0);
            x0[i] = std::max(-50.0, std::min((I[i] - mean) / (std + 1e-6), 50.0)); // C++11-compatible clamp
        }

        const size_t H0 = W0_.size();
        const size_t H1 = W1_.size();
        const size_t H2 = W2_.size();

        std::vector<scalar> h0(H0, 0.0);
        for (size_t i = 0; i < H0; ++i)
        {
            scalar sum = b0_[i];
            for (int j = 0; j < 5; ++j) sum += W0_[i][j] * x0[j];
            h0[i] = relu(sum);
        }

        std::vector<scalar> h1(H1, 0.0);
        for (size_t i = 0; i < H1; ++i)
        {
            scalar sum = b1_[i];
            for (size_t j = 0; j < H0; ++j) sum += W1_[i][j] * h0[j];
            h1[i] = relu(sum);
        }

        std::vector<scalar> h2(H2, 0.0);
        for (size_t i = 0; i < H2; ++i)
        {
            scalar sum = b2_[i];
            for (size_t j = 0; j < H1; ++j) sum += W2_[i][j] * h1[j];
            h2[i] = relu(sum);
        }

        for (int i = 0; i < 10; ++i)
        {
            scalar sum = b3_[i];
            for (size_t j = 0; j < H2; ++j) sum += W3_[i][j] * h2[j];
            g[i] = sum;
        }
    }

    static void calcBasis(
        const tensor& gradU,
        const scalar omegaVal,
        scalar I[5],
        tensor T[10]
    )
    {
        const scalar tau = min(5.0, 1.0 / max(omegaVal, 1e-4));
        const symmTensor S_symm = symm(gradU);
        const tensor W_skew = skew(gradU);

        const tensor s = tau * tensor(S_symm);
        const tensor w = tau * W_skew;

        const tensor s2 = s & s;
        const tensor w2 = w & w;
        const tensor s3 = s2 & s;
        const tensor w2s = w2 & s;
        const tensor w2s2 = w2 & s2;

        I[0] = tr(s2);
        I[1] = tr(w2);
        I[2] = tr(s3);
        I[3] = tr(w2s);
        I[4] = tr(w2s2);

        const tensor I3 = tensor::I;

        T[0] = s;
        T[1] = (s & w) - (w & s);
        T[2] = s2 - (1.0 / 3.0) * I[0] * I3;
        T[3] = w2 - (1.0 / 3.0) * I[1] * I3;
        T[4] = (w & s2) - (s2 & w);
        T[5] = (w2 & s) + (s & w2) - (2.0 / 3.0) * I[3] * I3;
        T[6] = ((w & s) & w2) - ((w2 & s) & w);
        T[7] = ((s & w) & s2) - ((s2 & w) & s);
        T[8] = (w2 & s2) + (s2 & w2) - (2.0 / 3.0) * I[4] * I3;
        T[9] = ((w & s2) & w2) - ((w2 & s2) & w);
    }

    static symmTensor enforceRealizability(const symmTensor& a)
    {
        symmTensor aDev = dev(a);
        scalar pVal = 0.5 * (sqr(aDev.xx()) + sqr(aDev.yy()) + sqr(aDev.zz())
                           + 2.0 * (sqr(aDev.xy()) + sqr(aDev.xz()) + sqr(aDev.yz())));
        scalar qVal = det(aDev);

        scalar scale = 1.0;
        if (pVal > 1e-14)
        {
            scalar mVal = 2.0 * sqrt(pVal / 3.0);
            scalar arg = std::max(-1.0, std::min(3.0 * qVal / (pVal * mVal), 1.0)); // C++11-compatible clamp
            scalar theta = (1.0 / 3.0) * acos(arg);
            scalar l1 = mVal * cos(theta);
            scalar l2 = mVal * cos(theta - 2.0 * M_PI / 3.0);
            scalar l3 = mVal * cos(theta + 2.0 * M_PI / 3.0);

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
        return scale * aDev;
    }

    static symmTensor calcAnisotropy(const scalar g[10], const tensor T[10])
    {
        tensor a = tensor::zero;
        for (int n = 0; n < 10; ++n)
        {
            a += g[n] * T[n];
        }
        return enforceRealizability(symm(a));
    }
};

}

#endif
