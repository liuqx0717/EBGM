#include "alg.h"
#include "kernels.h"
#include "jet.hpp"
#include "points.hpp"
#include "graph.hpp"

#include <tuple>

#include <opencv2/core.hpp>


#define PI 3.14159265358979323846F

using namespace cv;
using namespace std;

// Generate 40 Garbor kernels.
// Each kernel is a kernelSize*kernelSize matrix.
void genGaborKernels(
    int kernelSize,
    Kernels<40> &result_kernels    // for function overloading
)
{
    assert(kernelSize > 0);

    GarborKernel gk(kernelSize);

    result_kernels.kx.reset(new float[40]);
    result_kernels.ky.reset(new float[40]);
    float *kx_p = result_kernels.kx.get();
    float *ky_p = result_kernels.ky.get();

    #pragma omp parallel for collapse(2) schedule(static)
    for (int nu = 0; nu <= 4; nu++){
        for (int mu = 0; mu <= 7; mu++){
            float k = powf(2.0F, -0.5F*(float)nu) * PI/2.0F;
            float phi = (float)mu * PI/8.0F;

            float kx = k * cosf(phi);
            float ky = k * sinf(phi);

            int j = mu + 8*nu;

            kx_p[j] = kx;
            ky_p[j] = ky;
            tie(result_kernels.re[j], result_kernels.im[j]) = 
                gk.getKernel(2.0F*PI, kx, ky);
        }
    }
}


// return value: dx, dy
std::tuple<float/*dx*/,float/*dy*/>
displacementWithFocus(
    const Jet<40> &jet1, 
    const Jet<40> &jet2, 
    int focus
)
{
    assert(focus >=1 && focus <=5);

    float dx = 0, dy = 0;
    for(int i = 1; i <= focus; i++){
        int startIndex;
        startIndex = 40 - 8*focus;
        tie(dx, dy) = jet1.displacement(jet2, startIndex, 40, dx, dy);
    }

    return make_tuple(dx, dy);
}


