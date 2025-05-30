#include <stdint.h>

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

void conv2d5k2x(
    uint8_t *in_row_0,  // -2
    uint8_t *in_row_1,  // -1 
    uint8_t *in_row_2,  //  0
    uint8_t *in_row_3,  //  1
    uint8_t *in_row_4,  //  2
    uint8_t *out_row, int32_t out_w,
    int16_t *kernels    // [2, 5, 5]
) {
    for (int32_t out_x = 0; out_x < out_w; out_x++) {
        int32_t in_x = out_x / 2;
        int32_t k_x = out_x % 2;
        
        int32_t pixel = 0;
        int32_t acc = 0;
        for (int32_t m = 0; m < 5; m++) {
            for (int32_t n = 0; n < 5; n++) {
                int32_t idx = k_x * 25 + m * 5 + n;
                int16_t w = kernels[idx];
                acc += w;

                // select row
                uint8_t *in_row = 0;
                if (n == 0)      in_row = in_row_0;
                else if (n == 1) in_row = in_row_1;
                else if (n == 2) in_row = in_row_2;
                else if (n == 3) in_row = in_row_3;
                else if (n == 4) in_row = in_row_4;

                // clamp 
                int32_t k_in_x = in_x + m - 2;
                if (k_in_x < 0) k_in_x = 0;
                if (k_in_x > out_w / 2 - 1) k_in_x = out_w / 2 - 1;
              
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
