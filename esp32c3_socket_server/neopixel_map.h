#ifndef NEOPIXEL_MAP
#define NEOPIXEL_MAP
#include <stdint.h>

/**
      Neopixel Index Map Layout:
| 8   9   10  11  12  13  14  15|
|                               |
|             Front             |
|               A               |
|              /|\              |
|               |               |
|               |               |
|               |               |
|                               |
|                               |
| 7   6   5   4   3   2   1   0 |

This layout represents a 2D arrangement of neopixels where the front is located at indices 8-15,
and the back at indices 0-7. The "A" denotes the center position, with the rest of the pixels positioned
in a sequential pattern around it.
**/

typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} rgb_color_t;

typedef struct {
    rgb_color_t render_color_palette[5][16];
} rgb_palette_t;

// "stop" color palette:
// - Green (0, 200, 0) is assigned to the front LEDs (indices 0-7).
// - Red (200, 0, 0) is assigned to the back LEDs (indices 8-15).
// - Based on the Neopixel index map, the front-facing LEDs (bottom row) are green,
// while the rear-facing LEDs (top row) are red.
rgb_palette_t stop_palette = {
    {
        {{0, 200, 0}, {0, 200, 0}, {0, 200, 0}, {0, 200, 0}, {0, 200, 0}, {0, 200, 0}, {0, 200, 0}, {0, 200, 0},
         {200, 0, 0}, {200, 0, 0}, {200, 0, 0}, {200, 0, 0}, {200, 0, 0}, {200, 0, 0}, {200, 0, 0}, {200, 0, 0}},
        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
         {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
        {{0, 200, 0}, {0, 200, 0}, {0, 200, 0}, {0, 200, 0}, {0, 200, 0}, {0, 200, 0}, {0, 200, 0}, {0, 200, 0},
         {200, 0, 0}, {200, 0, 0}, {200, 0, 0}, {200, 0, 0}, {200, 0, 0}, {200, 0, 0}, {200, 0, 0}, {200, 0, 0}},
        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
         {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
         {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
    }
};

// The "forward" palette uses a grayscale pattern:
// - `{200, 200, 200}` represents white/gray pixels.
// - `{0, 0, 0}` represents black/off pixels.
// - The pattern appears to form a diagonal or scattered light effect
//   rather than a strict front/back color distinction.
rgb_palette_t forward_palette = {
    {
        {{0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0},
         {200, 0, 0}, {200, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 200}, {0, 0, 200}},
        {{0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0},
         {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
        {{0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0},
         {0, 0, 200}, {0, 0, 200}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {200, 0, 0}, {200, 0, 0}},
        {{0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0},
         {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
        {{0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0},
         {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
    }
};

// The "backward" palette uses a grayscale pattern similar to "forward" but with a reversed arrangement:
// - `{200, 200, 200}` represents white/gray pixels.
// - `{0, 0, 0}` represents black/off pixels.
// - The light pattern appears mirrored compared to the "forward" palette.
rgb_palette_t backward_palette = {
    {
        {{200, 0, 0}, {200, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 200}, {0, 0, 200},
         {0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0}},
        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
         {0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0}},
        {{0, 0, 200}, {0, 0, 200}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {200, 0, 0}, {200, 0, 0},
         {0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0}},
        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
         {0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0}},
        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
         {0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0}}
    }
};

// The "left" palette creates a pattern with alternating orange and black sections:
// - `{10, 20, 0}` represents orange pixels, creating visible light or active sections.
// - `{0, 0, 0}` represents black/off pixels, indicating inactive or off sections.
// - The orange pixels are arranged in a staggered pattern, with each row containing alternating sections of orange and black.
//   This arrangement gives the effect of a diagonal or alternating striping pattern from left to right.
rgb_palette_t left_palette = {
    {
        {{0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {200, 0, 0}, {200, 0, 0},
         {0, 0, 200}, {0, 0, 200}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0}},
        {{0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
         {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0}},
        {{0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 200}, {0, 0, 200},
         {200, 0, 0}, {200, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0}},
        {{0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
         {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0}},
        {{0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
         {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0}}
    }
};

// The "right" palette creates a pattern similar to the left palette, with alternating orange and black sections:
// - `{10, 20, 0}` represents orange pixels, creating visible light or active sections.
// - `{0, 0, 0}` represents black/off pixels, indicating inactive or off sections.
// - The pattern alternates the orange pixels in a diagonal or staggered manner from right to left, with each row containing a mix of orange and black.
//   This gives the effect of a mirrored or alternating striping pattern from right to left.
rgb_palette_t right_palette = {
    {
        {{0, 0, 200}, {0, 0, 200}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0},
         {0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {200, 0, 0}, {200, 0, 0}},
        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0},
         {0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
        {{200, 0, 0}, {200, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0},
         {0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 200}, {0, 0, 200}},
        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0},
         {0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
        {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 200, 0}, {0, 200, 0},
         {0, 200, 0}, {0, 200, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
    }
};

#endif // MY_ESP32C3_LIB
