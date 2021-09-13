#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<cmath>

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

void conv2d_forward(const py::array_t<float>& feat_input, const py::array_t<float>& kernel, const py::array_t<float>& bias, py::array_t<float>& padding_feat, py::array_t<float>& feat_output, int stride, int padding) 
{
    py::buffer_info feat_input_buf = feat_input.request();
    py::buffer_info kernel_buf = kernel.request();
    py::buffer_info bias_buf = bias.request();
    py::buffer_info feat_output_buf = feat_output.request();

    if (feat_input_buf.shape[1] != kernel_buf.shape[1])  // input features' channels should be equal to kernel's input channels
    {
        throw std::runtime_error("feature input channels should be equal to kernel's input channels!");
    }

    if (bias_buf.shape[0] != kernel_buf.shape[0])       // bias' size should be equal to kernel's output channels
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
        py::buffer_info padding_feat_buf = padding_feat.request();
        float * new_input_ptr = (float*)padding_feat_buf.ptr;
        size_t feat_h = dh+padding*2;
        size_t feat_w = dw+padding*2;
        size_t feat_img_size = feat_h*feat_w;
        size_t padding_zero = padding*feat_w;
        for(int i = 0; i < db; ++i)
            for(int j = 0; j < input_channels; ++j) {
                for(int k = 0; k < dh; ++k) {
                    //                           base        padding     done                      base    done
                    memcpy(new_input_ptr+i*j*feat_img_size+padding_zero+k*feat_w, feat_input_ptr+i*j*dh*dw+k*dw, dw*sizeof(float));
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
}

void conv2d_forward_withbias(const py::array_t<float>& feat_input, const py::array_t<float>& kernel, const py::array_t<float>& bias, py::array_t<float>& padding_feat, py::array_t<float>& feat_output, int stride, int padding) 
{
    conv2d_forward(feat_input, kernel, bias, padding_feat, feat_output, stride, padding);
}

void conv2d_forward_nobias(const py::array_t<float>& feat_input, const py::array_t<float>& kernel, py::array_t<float>& padding_feat, py::array_t<float>& feat_output, int stride, int padding) 
{
    py::buffer_info kernel_buf = kernel.request();
	auto bias = py::array_t<float>(kernel_buf.shape[0]);   // output channels size
    conv2d_forward(feat_input, kernel, bias, padding_feat, feat_output, stride, padding);
}

void calc_grad_kernel(const float * input_data, py::array_t<float>& grad_kernel, float grad_number, \
                      int out_channel_indx, int input_channles, int kh, int kw, int dh, int dw)
{
     auto grad = grad_kernel.mutable_unchecked<4>();
     for(int i = 0; i < input_channles; ++i) {
         for(int h = 0; h < kh; ++h){
             for(int w = 0; w < kw; ++w)
                 grad(out_channel_indx, i, h, w) = grad_number*input_data[i*dh*dw+h*dw+w];
         }
     }
}

void calc_grad_input(const float * kernel_data, py::array_t<float>& grad_input, float grad_number, \
                      int out_channel_indx, int input_channles, int kh, int kw, int dh_start, int dw_start)
{
     auto grad = grad_input.mutable_unchecked<4>();
     for(int i = 0; i < input_channles; ++i) {
         for(int h = dh_start; h < dh_start+kh; ++h){
             for(int w = dw_start; w < dw_start+kw; ++w)
                 grad(out_channel_indx, i, h, w) = grad_number*kernel_data[i*kh*kw+h*kw+w-dw_start];
         }
     }
}   

void conv2d_backward(const py::array_t<float>& grad_output, const py::array_t<int>& feat_input, const py::array_t<int>& kernel, const py::array_t<int>& bias, \
                     py::array_t<float>& input_grad, py::array_t<float>& kernel_grad, py::array_t<float>& bias_grad, int stride, int padding)
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

    // grad for kernel & input data
    float * input_feat = (float*)feat_input_buf.ptr;
    float * kernel_feat = (float*)kernel_buf.ptr;
    for(size_t b = 0; b < bs; ++b) {
        for(size_t out = 0; out < output_channels; ++out)
            for(size_t i = 0; i < dh+padding*2-kh+1; i += stride)
                for(size_t j = 0; j < dw+padding*2-kw+1; j += stride) {
                    float * input_data = input_feat+b*input_channels*dh*dw+i*dw+j;
                    float * kernel_data = kernel_feat+out*input_channels*kh*kw;
                    calc_grad_kernel(input_data, kernel_grad, grad_output_arr(b, out, i/stride, j/stride), out, input_channels, kh, kw, dh, dw);
                    calc_grad_input(kernel_data, input_grad, grad_output_arr(b, out, i/stride, j/stride), out, input_channels, kh, kw, i, j);
                }
    }
}

PYBIND11_MODULE(conv_operations, m) {
    m.doc() = "conv2d forward & backward with pybind11";

	m.attr("__version__") = "0.0.1";
    m.def("add", &add, "for test");
    m.def("conv2d_forward_withbias", &conv2d_forward_withbias, "A function do conv2d forward(with bias) operation according to input features & kernel");
    m.def("conv2d_forward_nobias", &conv2d_forward_nobias, "A function do conv2d forward(without bias) operation according to input features & kernel");
    m.def("conv2d_backward", &conv2d_backward, "A function do conv2d backward operation according to grad output");
}