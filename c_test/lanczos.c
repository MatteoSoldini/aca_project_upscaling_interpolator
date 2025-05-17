#include <stdio.h>

#define _USE_MATH_DEFINES // for C
#include <math.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

typedef unsigned char u8;
typedef char i8;
typedef unsigned int u32;
typedef int i32;

void neareast(u8 *in_pixels, i32 in_w, i32 in_h, i32 c, u8 *out_pixels, i32 out_w, i32 out_h) {
    printf("x=%d, y=%d, n=%d\n", out_w, out_h, c);

    for (i32 i = 0; i < out_w; i++) {
        for (i32 j = 0; j < out_h; j++) {
            float norm_x = (float)i / out_w;
            float norm_y = (float)j / out_h;
            
            i32 in_x = norm_x * in_w;
            i32 in_y = norm_y * in_h;

            for (i32 k = 0; k < c; k++) {
                out_pixels[c * (i + j * out_w) + k] = in_pixels[c * (in_x + in_y * in_w) + k];
            }
        }
    }
}

double lanczos_kernel(double x, i32 a) {
    if (!(fabs(x) - 0.1f <= a)) {
        printf("x=%f should be smaller than a=%u\n", fabs(x), a);
        exit(-1);
    }

    if (x == 0.0f) return 1.0f;
    return a * sin(M_PI * x) * sin(M_PI * x / a) / pow(x, 2) / pow(M_PI, 2);
}

i32 clamp(i32 in, i32 low, i32 high) {
    if (in < low) return low;
    if (in > high) return high;
    return in;
}

i32 mirror(i32 in, i32 low, i32 high) {
    if (in < low) return low + abs(in - low);
    if (in > high) return high - abs(in - high);
    return in;
}

void lanczos(u8 *in_pixels, i32 in_w, i32 in_h, i32 c, u8 *out_pixels, i32 out_w, i32 out_h, i32 a) {
    if (!(a > 0)) {
        printf("a=%i should be greater than 0\n", a);
        return;
    }

    double ratio_x = (double)in_w / out_w;
    double ratio_y = (double)in_h / out_h;

    double *conv_pixel = malloc(sizeof(float) * c);

    for (i32 i = 0; i < out_w; i++) {
        for (i32 j = 0; j < out_h; j++) {
            i32 in_x = i * ratio_x;
            i32 in_y = j * ratio_y;

            // convolve
            conv_pixel[0] = 0.0f;
            conv_pixel[1] = 0.0f;
            conv_pixel[2] = 0.0f;

            double sum = 0.0f;
            for (i32 m = -a + 1; m <= a; m++) {
                for (i32 n = -a + 1; n <= a; n++) {
                    double x = in_x - i * ratio_x + m;
                    double y = in_y - j * ratio_y + n;
                    
                    double weight = lanczos_kernel(x, a) * lanczos_kernel(y, a);
                    sum += weight;
                    //printf("x=%f, y=%f, weight=%f\n", x, y, weight);
                                        
                    i32 real_in_x = mirror(in_x + m, 0, in_w - 1);
                    i32 real_in_y = mirror(in_y + n, 0, in_h - 1);
                    
                    i32 idx = c * (real_in_x + real_in_y * in_w);
                    for (i32 k = 0; k < c; k++) {
                        conv_pixel[k] += in_pixels[idx + k] * weight;
                    }
                }
            }

            i32 idx = c * (i + j * out_w);
            for (i32 k = 0; k < c; k++) {
                out_pixels[idx + k] = clamp((u32)conv_pixel[k] / sum, 0, 255);
            }
        }
    }

    free(conv_pixel);
}

int main(void) {
    char *in_file = "input.jpg";
    char *out_file = "output.bmp";

    i32 w, h, c;
    u8 *pixels = stbi_load(in_file, &w, &h, &c, 0);

    printf("x=%d, y=%d, c=%d\n", w, h, c);
    
    i32 out_w = 1.5f*w;
    i32 out_h = 1.5f*h;
    u8 *out_pixels = malloc(out_w * out_h * c);
    //neareast(pixels, w, h, c, out_pixels, out_w, out_h);
    lanczos(pixels, w, h, c, out_pixels, out_w, out_h, 2);

    stbi_write_bmp(out_file, out_w, out_h, c, out_pixels);
    
    free(out_pixels);
    stbi_image_free(pixels);
    return 0;
}