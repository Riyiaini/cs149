#include <math.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>

void sqrtAvx(int N,
                   float initialGuess,
                   float values[],
                   float output[])
{
    static const float kThreshold = 0.00001f;
    int alignedN = N & ~0x7;

    for(int i=0; i<alignedN; i+=8) {
        
        __m256 x_vec = _mm256_loadu_ps(values+i);
        __m256 thresh = _mm256_set1_ps(kThreshold);
        __m256 guess = _mm256_set1_ps(initialGuess);

        __m256 error = _mm256_andnot_ps(
            _mm256_set1_ps(-0.0f),
            _mm256_sub_ps(
                _mm256_mul_ps(_mm256_mul_ps(guess, guess), x_vec),
                _mm256_set1_ps(1.0f)
            )
        );

        while(_mm256_movemask_ps(_mm256_cmp_ps(error, thresh, _CMP_GT_OQ)) != 0) {

            guess = _mm256_mul_ps(
                _mm256_set1_ps(0.5f),
                _mm256_sub_ps(
                    _mm256_mul_ps(_mm256_set1_ps(3.0f), guess),
                    _mm256_mul_ps(x_vec, _mm256_mul_ps(guess, _mm256_mul_ps(guess, guess)))
                )
            );

            error = _mm256_andnot_ps(
                _mm256_set1_ps(-0.0f),
                _mm256_sub_ps(
                    _mm256_mul_ps(_mm256_mul_ps(guess, guess), x_vec),
                    _mm256_set1_ps(1.0f)
                )
            );
        }

        _mm256_storeu_ps(output+i, _mm256_mul_ps(x_vec, guess));
    }

    if(alignedN < N) {

        for (int i=alignedN; i<N; i++) {
            float x = values[i];
            float guess = initialGuess;
            float error = fabs(guess * guess * x - 1.f);

            while (error > kThreshold) {
                guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
                error = fabs(guess * guess * x - 1.f);
            }

            output[i] = x * guess;
        }
    }
}