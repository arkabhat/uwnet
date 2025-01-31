#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    for (int im = 0; im < in.rows; im++) {
        // Get image from current row
        matrix currImage;
        currImage.rows = l.height * l.channels;
        currImage.cols = l.width;
        // Array pointer arithmetic means the below works for assigning data to currImage
        currImage.data = in.data + im * in.cols;

        // Create output matrix
        matrix pooledImage;
        pooledImage.rows = outh * l.channels;
        pooledImage.cols = outw;
        // Assign pooledImage to point to appropriate place in output
        pooledImage.data = out.data + im * out.cols;

        // We now have a single image from the dataset, lets pool for it.
        for (int c = 0; c < l.channels; c++) {
            // Channel offset for pooled image
            int channelOffsetPooled = outw * outh * c;

            // Channel offset for original image
            int channelOffsetOriginal = l.width * l.height * c;

            // counter for position
            int currConv = 0;

            for (int i = 0; i < l.height; i += l.stride) {
                for (int j = 0; j < l.width; j += l.stride) {
                    float max = -FLT_MAX;
                    for (int k = 0; k < l.size; k++) {
                        for (int m = 0; m < l.size; m++) {
                            int filter_row = i - (l.size - 1) / 2 + k; // gives corresponding row in matrix
                            int filter_col = j - (l.size - 1) / 2 + m; // gives corresponding col in matrix
                            if (filter_row >= 0 && filter_col >= 0 && filter_row < currImage.rows && filter_col < currImage.cols) { // only valid positions
                                if (currImage.data[filter_col + currImage.cols * filter_row + channelOffsetOriginal] > max) {
                                    max = currImage.data[filter_col + currImage.cols * filter_row + channelOffsetOriginal];
                                }
                            }
                        }
                    }
                    
                    // We have the max for this sizexsize filter region. Store in out
                    pooledImage.data[channelOffsetPooled + currConv] = max;
                    currConv++;
                }
            }
        }
    }
    
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.

    // Each row in dy -> error for one image from input in
    // For each row in dy, we want to populate dx with the equivalent of an image
    // So, i.e. in[1] <- in[i] + dy[1] where we apply dy[1] to only the max val in 
    // every pooling filter region
    for (int im = 0; im < in.rows; im++) {

        // Get image from data of previous layer
        matrix prevImage;// = make_matrix(l.height * l.channels, l.width);
        prevImage.rows = l.height * l.channels;
        prevImage.cols = l.width;
        // Array pointer arithmetic means the below works for assigning data to currImage
        prevImage.data = in.data + im * in.cols;

        // Create dy Matrix for current image
        matrix currDelta;// = make_matrix(l.channels * outh, outw);
        currDelta.rows = outh * l.channels;
        currDelta.cols = outw;
        // Assign currDelta correctly with pointer arithmetic
        currDelta.data = dy.data + im * dy.cols;

        // Create matrix pointing to dx
        matrix out;// = make_matrix(l.height * l.channels, l.width);
        out.rows = l.height * l.channels;
        out.cols = l.width;
        // Assign to proper location of dx
        out.data = dx.data + im * dx.cols;
        
        // We now have a single image from the dataset, lets pool for it.
        for (int c = 0; c < l.channels; c++) {
            // Channel offset for delta image
            int channelOffsetDelta = outw * outh * c;

            // Channel offset for previous image
            int channelOffsetPrev = l.width * l.height * c;

            // counter for position
            int currConv = 0;
            
            for (int i = 0; i < l.height; i += l.stride) {
                for (int j = 0; j < l.width; j += l.stride) {
                    float max = -FLT_MAX;
                    int m_row = 0;
                    int m_col = 0;

                    for (int k = 0; k < l.size; k++) {
                        for (int m = 0; m < l.size; m++) {
                            int filter_row = i - (l.size - 1) / 2 + k; // gives corresponding row in matrix
                            int filter_col = j - (l.size - 1) / 2 + m; // gives corresponding col in matrix
                            if (filter_row >= 0 && filter_col >= 0 && filter_row < prevImage.rows && filter_col < prevImage.cols) { // only valid positions
                                if (prevImage.data[filter_col + prevImage.cols * filter_row + channelOffsetPrev] > max) {
                                    max = prevImage.data[filter_col + prevImage.cols * filter_row + channelOffsetPrev];
                                    m_row = filter_row;
                                    m_col = filter_col;
                                }
                            }
                        }
                    }
                    // We have the max for this sizexsize filter region and its related delta, update prevImage
                    out.data[m_row * out.cols + m_col + channelOffsetPrev] += currDelta.data[channelOffsetDelta + currConv];
                    currConv++;
                }
            }
        }
    }
    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

