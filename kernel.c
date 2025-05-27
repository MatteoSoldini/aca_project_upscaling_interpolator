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
