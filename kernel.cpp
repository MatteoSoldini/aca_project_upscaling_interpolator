#include <stdint.h>

#include <aie_api/aie.hpp>

#define INT_SCALE (1 << 12)

void vector_scalar_mul_aie_scalar(int32_t *a, int32_t *c, int32_t *factor,
                                  int32_t N) {
  for (int i = 0; i < N; i++) {
    c[i] = *factor * a[i];
  }
}

void passthrough(uint8_t *in, uint8_t *out, int32_t N) {
    for (int32_t i = 0; i < N; i++) {
        out[i] = in[i];
    }
}


void write(uint8_t *out, int32_t N) {
    for (int32_t i = 0; i < N; i++) {
        out[i] = 100;
    }
}

// N -> output
void nearest_neightbor_2x(
    uint8_t* in_row, uint8_t *out_row, int32_t out_width
) {
    for (int32_t i = 0; i < out_width; i++) {
        out_row[i] = in_row[i / 2];
    }

//    for (int32_t o_x = 0; o_x < tile_size; o_x++) {
//        float x_norm = (float)o_x / out_w;
//        float y_norm = (float)o_y / out_h;
//
//        int32_t i_x = x_norm * in_w;
//        int32_t i_y = y_norm * in_h;
//        
//        uint8_t *row = i_y == o_y ? row0 : row1; 
//
//        //out[o_x] = row[i_x];
//        out[o_x] = o_y;
//    }
}

void conv2d3k2x(
    uint8_t *in_row_0,  // -1
    uint8_t *in_row_1,  //  0
    uint8_t *in_row_2,  //  1
    uint8_t *out_row, int32_t out_w
) {
    int32_t weight[3] = {4, 2, 4};
    
    for (int32_t out_x = 0; out_x < out_w; out_x++) {
        int32_t in_x = out_x / 2;

        int32_t sum = 0;
        for (int32_t m = 0; m < 3; m++) {
            for (int32_t n = 0; n < 3; n++) {
                int32_t w = weight[m] * weight[n];
                
                // select row
                uint8_t *in_row = 0;
                if (n == 0)      in_row = in_row_0;
                else if (n == 1) in_row = in_row_1;
                else             in_row = in_row_2;
               
                // clamp 
                int32_t k_in_x = in_x + m - 1;
                if (k_in_x < 0) k_in_x = 0;
                if (k_in_x > out_w / 2 - 1) k_in_x = out_w / 2 - 1;
                
                sum += (uint64_t)(in_row[k_in_x] * 1000) / w;
            }
        }

        out_row[out_x] = sum / 1000;
    }
}



extern "C" {
#ifdef SCALAR
void conv2d4k(
    uint8_t *in_row_0,  // -1
    uint8_t *in_row_1,  //  0 
    uint8_t *in_row_2,  //  1
    uint8_t *in_row_3,  //  2
    int16_t *c_mtx_row, int32_t num_c_mtx_row,
    uint8_t *out_row, int32_t out_w
) {
    int32_t scale_factor = num_c_mtx_row;

    for (int32_t out_x = 0; out_x < out_w; out_x++) {
        int32_t in_x = out_x / scale_factor;
        int32_t k_x = out_x % scale_factor;
        
        int32_t pixel = 0;
        int32_t acc = 0;
        for (int32_t m = 0; m < 4; m++) {
            for (int32_t n = 0; n < 4; n++) {
                int32_t idx = 16*k_x + 4*n + m;
                int16_t w = c_mtx_row[idx];
                acc += w;

                // select row
                uint8_t *in_row = 0;
                if (n == 0)      in_row = in_row_0;
                else if (n == 1) in_row = in_row_1;
                else if (n == 2) in_row = in_row_2;
                else if (n == 3) in_row = in_row_3;

                // clamp 
                int32_t k_in_x = in_x + m - 1;
                if (k_in_x < 0) k_in_x = 0;
                if (k_in_x > out_w / 2 - 1) k_in_x = out_w / scale_factor - 1;
              
                pixel += (int32_t)(in_row[k_in_x]) * w;
            }
        }
        
        pixel /= acc;
        
        // clamp
        if (pixel < 0) pixel = 0;
        if (pixel > 255) pixel = 255;

        out_row[out_x] = pixel;
    }
}
#else
void conv2d4k(
    uint8_t *in_row_0,  // -1
    uint8_t *in_row_1,  //  0 
    uint8_t *in_row_2,  //  1
    uint8_t *in_row_3,  //  2
    int16_t *c_mtx_row, int32_t num_c_mtx_row,
    uint8_t *out_row, int32_t out_w
) {
    int32_t num_mul = num_c_mtx_row / 2 + num_c_mtx_row % 2;
    int32_t scale_factor = num_c_mtx_row;    

    for (int32_t i = 0; i < out_w / scale_factor; i++) {
        // fill input vector
        aie::vector<int16_t, 32> in_vec = aie::zeros<int16_t, 32>();
        for (int32_t k_x = 0; k_x < 4; k_x++) {
            
            // clamp
            int32_t in_x = i + k_x - 1;
            if (in_x < 0) in_x = 0;
            if (in_x > out_w/scale_factor - 1) in_x = out_w/scale_factor - 1;
                    
            for (int32_t p = 0; p < 2; p++) {
                in_vec[16*p + k_x]      = in_row_0[in_x];
                in_vec[16*p + 4 + k_x]  = in_row_1[in_x];
                in_vec[16*p + 8 + k_x]  = in_row_2[in_x];
                in_vec[16*p + 12 + k_x] = in_row_3[in_x];
            }
        }

        for (int32_t mul_i = 0; mul_i < num_mul; mul_i++) {
            // fill c_mtx_row vector
            aie::vector<int16_t, 32> c_mtx_vec = aie::load_v<32>(c_mtx_row + 32*mul_i);
            
            // run the two convolutions
            aie::accum<acc32, 32> s = aie::mul(in_vec, c_mtx_vec);
            aie::vector<int32_t, 32> s_vec = aie::to_vector<int32_t>(s);
           
            bool odd_mul = mul_i == num_mul - 1 && num_c_mtx_row % 2;
            int32_t num_p = odd_mul ? 1 : 2;
            for (int32_t p = 0; p < num_p; p++) {
                int32_t pixel = aie::reduce_add(s_vec.extract<16>(p));
                int32_t sum = aie::reduce_add(c_mtx_vec.extract<16>(p));
                
                pixel /= sum;
                if (pixel < 0)   pixel = 0;
                if (pixel > 255) pixel = 255;
                
                out_row[num_c_mtx_row*i + 2*mul_i + p] = pixel;
            }
        } 
    }
}
#endif
}
