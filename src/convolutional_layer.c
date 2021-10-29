#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "uwnet.h"

// Add bias terms to a matrix
// matrix xw: partially computed output of layer
// matrix b: bias to add in (should only be one row!)
// returns: y = wx + b
matrix forward_convolutional_bias(matrix xw, matrix b)
{
    assert(b.rows == 1);
    assert(xw.cols % b.cols == 0);

    matrix y = copy_matrix(xw);
    int spatial = xw.cols / b.cols;
    int i,j;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            y.data[i*y.cols + j] += b.data[j/spatial];
        }
    }
    return y;
}

// Calculate dL/db from a dL/dy
// matrix dy: derivative of loss wrt xw+b, dL/d(xw+b)
// returns: derivative of loss wrt b, dL/db
matrix backward_convolutional_bias(matrix dy, int n)
{
    assert(dy.cols % n == 0);
    matrix db = make_matrix(1, n);
    int spatial = dy.cols / n;
    int i,j;
    for(i = 0; i < dy.rows; ++i){
        for(j = 0; j < dy.cols; ++j){
            db.data[j/spatial] += dy.data[i*dy.cols + j];
        }
    }
    return db;
}

// Make a column matrix out of an image
// image im: image to process
// int size: kernel size for convolution operation
// int stride: stride for convolution
// returns: column matrix
matrix im2col(image im, int size, int stride)
{

    int outw = (im.w-1)/stride + 1;
    int outh = (im.h-1)/stride + 1;
    int rows = im.c*size*size;
    int cols = outw * outh;
    matrix col = make_matrix(rows, cols);

    // TODO: 5.1
    // Fill in the column matrix with patches from the image
    
    for (int c = 0; c < im.c; c += 1) {

        int curr_pixel = 0;

        // iterate over pixels in image
        for (int i = 0; i < im.h; i += stride) {
            for (int j = 0; j < im.w; j += stride) {

                int index = 0;
                int channel_offset = c * size * size * col.cols;

                // iterate over filter positions
                for (int k = 0; k < size; k++) {
                    for (int l = 0; l < size; l++) {

                        int filter_row = i - (size - 1) / 2 + k; // gives corresponding row in image
                        int filter_col = j - (size - 1) / 2 + l; // gives corresponding col in image

                        float pixel;
                        if (filter_row < 0 || filter_col < 0 || filter_row >= im.h || filter_col >= im.w) { // padding
                            pixel = 0.0;
                        } else {
                            pixel = im.data[filter_col + im.w * (filter_row + im.h * c)];
                        }
                        col.data[index * col.cols + channel_offset + curr_pixel] = pixel;

                        index++;

                    }
                }

                curr_pixel++;

            }
        }
    }

    return col;
}

// The reverse of im2col, add elements back into image
// matrix col: column matrix to put back into image
// int size: kernel size
// int stride: convolution stride
// image im: image to add elements back into
image col2im(int width, int height, int channels, matrix col, int size, int stride)
{

    image im = make_image(width, height, channels);
    int outw = (im.w-1)/stride + 1;
    int rows = im.c*size*size;

    // TODO: 5.2
    // Add values into image im from the column matrix

    for (int c = 0; c < channels; c++) {

        int curr_pixel = 0;

        for (int i = 0; i < height; i += stride) {
            for (int j = 0; j < width; j += stride) {

                int index = 0;
                int channel_offset = c * size * size * col.cols;

                // iterate over filter positions
                for (int k = 0; k < size; k++) {
                    for (int l = 0; l < size; l++) {

                        int filter_row = i - (size - 1) / 2 + k; // gives corresponding row in image
                        int filter_col = j - (size - 1) / 2 + l; // gives corresponding col in image

                        if (filter_row >= 0 && filter_col >= 0 && filter_row < im.h && filter_col < im.w) { // only valid positions
                            float pixel = col.data[index * col.cols + channel_offset + curr_pixel];
                            im.data[filter_col + im.w * (filter_row + im.h * c)] += pixel;
                        }

                        index++;

                    }
                }

                curr_pixel++;
            }
        }
    }

    return im;
}

// Run a convolutional layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_convolutional_layer(layer l, matrix in)
{
    assert(in.cols == l.width*l.height*l.channels);
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int i, j;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.filters);
    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);
        matrix x = im2col(example, l.size, l.stride);
        matrix wx = matmul(l.w, x);
        for(j = 0; j < wx.rows*wx.cols; ++j){
            out.data[i*out.cols + j] = wx.data[j];
        }
        free_matrix(x);
        free_matrix(wx);
    }
    matrix y = forward_convolutional_bias(out, l.b);
    free_matrix(out);

    return y;
}

// Run a convolutional layer backward
// layer l: layer to run
// matrix dy: dL/dy for this layer
// returns: dL/dx for this layer
matrix backward_convolutional_layer(layer l, matrix dy)
{
    matrix in = *l.x;
    assert(in.cols == l.width*l.height*l.channels);

    int i;
    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;


    matrix db = backward_convolutional_bias(dy, l.db.cols);
    axpy_matrix(1, db, l.db);
    free_matrix(db);


    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);
    matrix wt = transpose_matrix(l.w);

    for(i = 0; i < in.rows; ++i){
        image example = float_to_image(in.data + i*in.cols, l.width, l.height, l.channels);

        dy.rows = l.filters;
        dy.cols = outw*outh;

        matrix x = im2col(example, l.size, l.stride);
        matrix xt = transpose_matrix(x);
        matrix dw = matmul(dy, xt);
        axpy_matrix(1, dw, l.dw);

        matrix col = matmul(wt, dy);
        image dxi = col2im(l.width, l.height, l.channels, col, l.size, l.stride);
        memcpy(dx.data + i*dx.cols, dxi.data, dx.cols * sizeof(float));
        free_matrix(col);

        free_matrix(x);
        free_matrix(xt);
        free_matrix(dw);
        free_image(dxi);

        dy.data = dy.data + dy.rows*dy.cols;
    }
    free_matrix(wt);
    return dx;

}

// Update convolutional layer
// layer l: layer to update
// float rate: learning rate
// float momentum: momentum term
// float decay: l2 regularization term
void update_convolutional_layer(layer l, float rate, float momentum, float decay)
{
    // TODO: 5.3
    axpy_matrix(decay, l.w, l.dw);
    axpy_matrix(-rate, l.dw, l.w);
    axpy_matrix(-rate, l.db, l.b);
    scal_matrix(momentum, l.dw);
    scal_matrix(momentum, l.db);
}

// Make a new convolutional layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of convolutional filter to apply
// int stride: stride of operation
layer make_convolutional_layer(int w, int h, int c, int filters, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.filters = filters;
    l.size = size;
    l.stride = stride;
    l.w  = random_matrix(filters, size*size*c, sqrtf(2.f/(size*size*c)));
    l.dw = make_matrix(filters, size*size*c);
    l.b  = make_matrix(1, filters);
    l.db = make_matrix(1, filters);
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update   = update_convolutional_layer;
    return l;
}

