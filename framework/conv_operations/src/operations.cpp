#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<cmath>
#include<iostream>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

inline int min(int a, int b){
    return a<b? a:b;
}

inline int max(int a, int b){
    return a<b? b:a;
}

void print_4darray(const py::array_t<float> arr)
{
    py::buffer_info arr_info = arr.request();
    size_t bs = arr_info.shape[0];
    size_t inns = arr_info.shape[1];
    size_t h = arr_info.shape[2];
    size_t w = arr_info.shape[3];

    auto arr_data = arr.unchecked<4>();
    for(int b = 0; b < bs; ++b) {
        for(int i = 0; i < inns; ++i) {
            for(int hh = 0; hh < h; ++hh) {
                for(int ww = 0; ww < w; ++ww)
                    std::cout<<arr_data(b, i, hh, ww)<<" ";
                std::cout<<std::endl;
            }
        }
        std::cout<<std::endl;
    }
}

float multiply(float * kernel, float * data, size_t dh, size_t dw, size_t kh, size_t kw, size_t input_channels) 
{
    float sum = 0;
    int kernel_inter = kh*kw;
    int data_inter = dh*dw;
    for(int i = 0; i < input_channels; ++i) {
        for(int h = 0; h < kh; ++h) {
            for(int w = 0; w < kw; ++w) {
                sum += kernel[i*kernel_inter+h*kw+w]*data[i*data_inter+h*dw+w];
            }
        }
    }

    return sum;
}

py::array_t<float> conv2d_forward(const py::array_t<float>& feat_input, const py::array_t<float>& kernel, const py::array_t<float>& bias, py::array_t<float>& feat_output, int stride, int padding) 
{
    py::buffer_info feat_input_buf = feat_input.request();
    py::buffer_info kernel_buf = kernel.request();
    py::buffer_info bias_buf = bias.request();

    if (feat_input_buf.shape[1] != kernel_buf.shape[1])  // input features' channels should be equal to kernel's input channels
    {
        throw std::runtime_error("feature input channels should be equal to kernel's input channels!");
    }

    if (bias_buf.shape[0] != kernel_buf.shape[0])        // bias' size should be equal to kernel's output channels
    {
        throw std::runtime_error("bias' size should be equal to kernel's output channels!");
    }

    size_t output_channels = kernel_buf.shape[0];
    size_t input_channels = kernel_buf.shape[1];
    size_t kh = kernel_buf.shape[2];
    size_t kw = kernel_buf.shape[3];

    size_t db = feat_input_buf.shape[0];
    //inns
    size_t dh = feat_input_buf.shape[2];
    size_t dw = feat_input_buf.shape[3];
    size_t feat_h = dh;
    size_t feat_w = dw;

    // handle padding
    float* kernel_ptr = (float*)kernel_buf.ptr;
    float* feat_input_ptr = (float*)feat_input_buf.ptr;
    
    if(padding > 0) {
        feat_h = dh+padding*2;
        feat_w = dw+padding*2;
        float * new_input_ptr = new float[db*input_channels*feat_h*feat_w]();

        size_t padding_1bs = input_channels*feat_h*feat_w;
        size_t pading_img_size = feat_h*feat_w;
        size_t feat_1bs = input_channels*dh*dw;
        size_t img_size = dh*dw;
        size_t padding_zero = padding*feat_w;
        for(int i = 0; i < db; ++i)
            for(int j = 0; j < input_channels; ++j) {
                for(int k = 0; k < dh; ++k) {
                    memcpy(new_input_ptr+i*padding_1bs+j*pading_img_size+padding_zero+k*feat_w+padding, \
                           feat_input_ptr+i*feat_1bs+j*img_size+k*dw, dw*sizeof(float));
                }
            }

        feat_input_ptr = new_input_ptr;
    }

    auto output = feat_output.mutable_unchecked<4>();
    size_t one_bs_data = input_channels*feat_h*feat_w;
    size_t one_bs_kernel = input_channels*kh*kw;
    for(int b = 0; b < db; ++b) {
        for(int out = 0; out < output_channels; ++out) {
            for(int i = 0; i < dh+padding*2-kh+1; i += stride) {
                for(int j = 0; j < dw+padding*2-kw+1; j += stride) {
                    float * input_data = &(feat_input_ptr[b*one_bs_data+i*feat_w+j]);
                    float * kernel_data = &(kernel_ptr[out*one_bs_kernel]);
                    output(b, out, i/stride, j/stride) = multiply(kernel_data, input_data, feat_h, feat_w, kh, kw, input_channels)+bias.at(out);
                }
            }
        }
    }

    if(padding > 0) {
        auto padding_feat = py::array_t<float>(db*input_channels*feat_h*feat_w);
        py::buffer_info padding_feat_buf = padding_feat.request();
        float * padding = (float*)padding_feat_buf.ptr;
        memcpy(padding, feat_input_ptr, db*input_channels*feat_h*feat_w*sizeof(float));
        delete [] feat_input_ptr;
        padding_feat.resize({db,input_channels,feat_h,feat_w});
        return padding_feat;
    } 

    return py::array_t<float>(1);   // dummy
    
}

py::array_t<float> conv2d_forward_withbias(const py::array_t<float>& feat_input, const py::array_t<float>& kernel, const py::array_t<float>& bias, py::array_t<float>& feat_output, int stride, int padding) 
{
    return conv2d_forward(feat_input, kernel, bias, feat_output, stride, padding);
}

py::array_t<float> conv2d_forward_nobias(const py::array_t<float>& feat_input, const py::array_t<float>& kernel, py::array_t<float>& feat_output, int stride, int padding) 
{
    py::buffer_info kernel_buf = kernel.request();
	auto bias = py::array_t<float>(kernel_buf.shape[0]);   // output channels size
    py::buffer_info bias_buf = bias.request();
    float * bias_ptr = (float*)bias_buf.ptr;
    memset(bias_ptr, 0, kernel_buf.shape[0]*sizeof(float));
    return conv2d_forward(feat_input, kernel, bias, feat_output, stride, padding);
}

void calc_grad_kernel(const py::array_t<float>& feat_input, py::array_t<float>& grad_kernel, float grad_number, \
                      int out_channel_indx, int bs_ind, int input_channels, int kh, int kw, int dh_start, int dw_start)
{
     auto grad = grad_kernel.mutable_unchecked<4>();
     auto feat_input_arr = feat_input.unchecked<4>();
     for(int i = 0; i < input_channels; ++i) {
         for(int h = 0; h < kh; ++h){
             for(int w = 0; w < kw; ++w)
                 grad(out_channel_indx, i, h, w) += grad_number*feat_input_arr(bs_ind, i, dh_start+h, dw_start+w);
         }
     }
}

void calc_grad_input(const py::array_t<float>& kernel, py::array_t<float>& grad_input, float grad_number, \
                      int output_channels_ind, int bs_ind, int input_channels, int kh, int kw, int dh_start, int dw_start)
{
     auto grad = grad_input.mutable_unchecked<4>();
     auto kernel_data =  kernel.unchecked<4>();
     for(int i = 0; i < input_channels; ++i) {
         for(int h = 0; h < kh; ++h)
             for(int w = 0; w < kw; ++w)
                 grad(bs_ind, i, h+dh_start, w+dw_start) += grad_number*kernel_data(output_channels_ind, i, h, w);
     }
}   

py::array_t<float> conv2d_backward(const py::array_t<float>& grad_output, const py::array_t<float>& feat_input, const py::array_t<float>& kernel, const py::array_t<float>& bias, \
                     py::array_t<float>& kernel_grad, py::array_t<float>& bias_grad, int stride, int padding)
{
    py::buffer_info grad_output_buf = grad_output.request();
    py::buffer_info feat_input_buf = feat_input.request();
    py::buffer_info kernel_buf = kernel.request();
    
    size_t bs = feat_input_buf.shape[0];
    size_t input_channels = feat_input_buf.shape[1];
    size_t dh = feat_input_buf.shape[2];
    size_t dw = feat_input_buf.shape[3];

    size_t output_channels = kernel_buf.shape[0];
    size_t ki = kernel_buf.shape[1];
    size_t kh = kernel_buf.shape[2];
    size_t kw = kernel_buf.shape[3];

    if (input_channels != ki)        // input feature's input channels should be equal to kernel's input channels
    {
        throw std::runtime_error("bias' size should be equal to kernel's output channels!");
    }

    // grad for bias
    size_t gh = grad_output_buf.shape[2];
    size_t gw = grad_output_buf.shape[3];
    auto grad_output_arr = grad_output.unchecked<4>();
    auto bias_grad_arr = bias_grad.mutable_unchecked<1>();
    for(size_t b = 0; b < bs; ++b) {
        for(size_t i = 0; i < output_channels; ++i) {
            float grad_sum = 0;
            for(size_t j = 0; j < gh; ++j)
                for(size_t k = 0; k < gw; ++k)
                    grad_sum += grad_output_arr(b, i, j, k);
            bias_grad_arr(i) += grad_sum;
        }
    }

    // allocate holder for input grad
    auto input_grad = py::array_t<float>(bs*input_channels*dh*dw);
    py::buffer_info input_grad_buf = input_grad.request();
    float * input_grad_ptr = (float*)input_grad_buf.ptr;
    memset(input_grad_ptr, 0, bs*input_channels*dh*dw*sizeof(float));
    input_grad.resize({bs,input_channels,dh,dw});


    // grad for kernel & input data
    for(size_t b = 0; b < bs; ++b) {
        for(size_t out = 0; out < output_channels; ++out)
            for(size_t i = 0; i < dh-kh+1; i += stride)
                for(size_t j = 0; j < dw-kw+1; j += stride) {
                    float grad_number = grad_output_arr(b, out, i/stride, j/stride);
                    calc_grad_kernel(feat_input, kernel_grad, grad_number, out, b, input_channels, kh, kw, i, j);
                    calc_grad_input(kernel, input_grad, grad_number, out, b, input_channels, kh, kw, i, j);
                }
    }
    
    if(padding > 0) {
        size_t grad_h = dh-2*padding;
        size_t grad_w = dw-2*padding;
        auto new_grad = py::array_t<float>(bs*input_channels*grad_h*grad_w);
        new_grad.resize({bs,input_channels,grad_h,grad_w});
        py::buffer_info new_grad_buf = new_grad.request();
        float * new_grad_ptr = (float*)new_grad_buf.ptr;
        size_t grad_img_size = grad_h*grad_w;
        size_t input_grad_img_size = dh*dw;
        size_t padding_size = padding*dw;
        for(int i = 0; i < bs; ++i)
            for(int j = 0; j < input_channels; ++j) {
                for(int k = 0; k < grad_h; ++k) {
                    memcpy(new_grad_ptr+i*input_channels*grad_img_size+j*grad_img_size+k*grad_w, 
                    input_grad_ptr+i*input_channels*input_grad_img_size+j*input_grad_img_size+padding_size+k*dw+padding, grad_w*sizeof(float));
                }
            }
        return new_grad;
    }

    return input_grad;
}


py::array_t<float> conv2d_backward_withbias(const py::array_t<float>& grad_output, const py::array_t<float>& feat_input, const py::array_t<float>& kernel, const py::array_t<float>& bias, \
                     py::array_t<float>& kernel_grad, py::array_t<float>& bias_grad, int stride, int padding) 
{
    return conv2d_backward(grad_output, feat_input, kernel, bias, kernel_grad, bias_grad, stride, padding);
}

py::array_t<float> conv2d_backward_nobias(const py::array_t<float>& grad_output, const py::array_t<float>& feat_input, const py::array_t<float>& kernel,  \
                     py::array_t<float>& kernel_grad, py::array_t<float>& bias_grad, int stride, int padding) 
{
    py::buffer_info kernel_buf = kernel.request();
	auto bias = py::array_t<float>(kernel_buf.shape[0]);   // output channels size
    return conv2d_backward(grad_output, feat_input, kernel, bias, kernel_grad, bias_grad, stride, padding);
}

py::array_t<int> maxpool2d_forward(const py::array_t<float>& feat_input, py::array_t<float>& output_max, int kernel_size, int stride, int padding)
{
    py::buffer_info feat_input_buf = feat_input.request();
    
    size_t bs = feat_input_buf.shape[0];
    size_t input_channels = feat_input_buf.shape[1];
    size_t h = feat_input_buf.shape[2];
    size_t w = feat_input_buf.shape[3];

    size_t dh = (h-kernel_size)/stride+1;
    size_t dw = (w-kernel_size)/stride+1;

    auto feat = feat_input.unchecked<4>();
    auto output = output_max.mutable_unchecked<4>();

    // allocate holder for max pool position
    auto output_max_inds = py::array_t<int>(bs*input_channels*dh*dw*2);
    output_max_inds.resize({bs,input_channels,dh,dw*2});
    auto output_max_pos = output_max_inds.mutable_unchecked<4>();
    for(size_t b = 0; b < bs; ++b)
        for(size_t inns = 0; inns < input_channels; ++inns) {
            for(size_t i = 0; i < h-kernel_size+1; i += stride) {
                for(size_t j = 0; j < w-kernel_size+1; j += stride) {
                        
                        size_t pos_h = i/stride;
                        size_t pos_w = j/stride;
                        float max_val = -1000000.0f;
                        int max_pos_h = -1;
                        int max_pos_w = -1;
                        for(int kh = i; kh < i+kernel_size; ++kh)
                            for(int kw = j; kw < j+kernel_size; ++kw)
                                if(max_val < feat(b, inns, kh, kw)) {
                                    max_val = feat(b, inns, kh, kw);
                                    max_pos_h = kh;
                                    max_pos_w = kw;
                                }
                        output(b,inns,pos_h,pos_w) = max_val;
                        output_max_pos(b,inns,pos_h,pos_w*2) = max_pos_h;
                        output_max_pos(b,inns,pos_h,pos_w*2+1) = max_pos_w;
                }
            }
        }

    return output_max_inds;
}

py::array_t<float> maxpool2d_backward(const py::array_t<float>& grad_output, const py::array_t<int>& output_max_inds, size_t dh, size_t dw, int kernel_size, int stride, int padding)
{
    py::buffer_info grad_output_buf = grad_output.request();
    
    size_t bs = grad_output_buf.shape[0];
    size_t input_channels = grad_output_buf.shape[1];
    size_t gh = grad_output_buf.shape[2];
    size_t gw = grad_output_buf.shape[3];
    
    auto grad_input = py::array_t<float>(bs*input_channels*dh*dw);
    py::buffer_info grad_input_buf = grad_input.request();
    float * grad_input_ptr = (float*)grad_input_buf.ptr;
    memset(grad_input_ptr, 0, bs*input_channels*dh*dw*sizeof(float));

    grad_input.resize({bs,input_channels,dh,dw});
    auto new_grad = grad_input.mutable_unchecked<4>();

    auto grad = grad_output.unchecked<4>();
    auto output_max_pos = output_max_inds.unchecked<4>();
    for(int b = 0; b < bs; ++b)
        for(int inns = 0; inns < input_channels; ++inns) {
            for(int i = 0; i < gh; ++i)
                for(int j = 0; j < gw; ++j) {
                    int pos_h = output_max_pos(b, inns, i, j*2);
                    int pos_w = output_max_pos(b, inns, i, j*2+1);
                    new_grad(b, inns, pos_h, pos_w) = grad(b, inns, i, j);
                }
        }

    return grad_input;
}

PYBIND11_MODULE(conv_operations, m) {
    m.doc() = "conv2d, maxpool2d forward & backward with pybind11";

	m.attr("__version__") = "0.0.1";
    m.def("add", &add, "for test");
    m.def("print_4darray", &print_4darray, "print 4d array for debug");
    m.def("conv2d_forward_withbias", &conv2d_forward_withbias, "A function do conv2d forward(with bias) operation according to input features & kernel");
    m.def("conv2d_forward_nobias", &conv2d_forward_nobias, "A function do conv2d forward(without bias) operation according to input features & kernel");
    m.def("conv2d_backward_withbias", &conv2d_backward_withbias, "A function do conv2d backward(with bias) operation according to grad output");
    m.def("conv2d_backward_nobias", &conv2d_backward_nobias, "A function do conv2d backward(without bias) operation according to grad output");
    m.def("maxpool2d_forward", &maxpool2d_forward, "A function do maxpool2d forward operation according to input features");
    m.def("maxpool2d_backward", &maxpool2d_backward, "A function do maxpool2d backward operation according to grad output");

}
