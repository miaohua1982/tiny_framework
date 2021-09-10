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

float multiply(float * kernel, float * data, dh, dw, kh, kw, input_channels) 
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
    size_t output_width = (dw+padding*2-kw)/stride+1;
    size_t output_height = (dh+padding*2-kh)/stride+1;
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
                    memcpy(new_input_ptr+i*j*feat_img_size+padding_zero+k*feat_w, feat_input_ptr+i*j*dh*dw+k*dw, dw*sizeof(float))
                }
            }

        feat_input_ptr = new_input_ptr;
    }

    auto output = feat_output.unchecked<4>();
    size_t one_bs_data = input_channels*feat_h*feat_w;
    size_t one_bs_kernel = input_channels*kh*kw;
    for(int b = 0; i < db; ++b) {
        for(int out = 0; out < output_channels; ++out) {
            for(int i = 0; i < dh+padding*2-kh+1; i += stride) {
                for(int j = 0; j < dw+padding*2-kw+1; j += stride) {
                    input_data = &(feat_input_ptr[b*one_bs_data+i*feat_w+j])
                    kernel_data = &(kernel_ptr[out*one_bs_kernel]);
                    output(b, out, i/stride, j/stride) = multiply(kernel_data, input_data, feat_h, feat_w, kh, kw, input_channels)+bias[out];
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

                bs, input_channels, dh, dw = self.creator[0].shape
                output_channels, ki, kh, kw = self.creator[1].shape
                
                # grad for bias
                if self.has_bias:
                    grad_bias = self.grad.data.sum(axis=(0,2,3))
                    self.creator[2].backward(Tensor(grad_bias), self)

                # grad for kernel
                input_data = self.creator[0].data
                padding = self.padding
                if padding>0:
                    input_data = self.padding_input_data
                grad_kernel = np.zeros((output_channels, input_channels, kh, kw))
                for b in range(bs):
                    for out in range(output_channels):
                        for i in range(0, dh+padding*2-kh+1, self.stride):
                            for j in range(0, dw+padding*2-kw+1, self.stride):
                                input = input_data[b, :, i:i+kh, j:j+kw]  # input_channels*kh*kw
                                grad_kernel[out] += input*self.grad.data[b, out, i//self.stride, j//self.stride]
                if padding>0:
                    del self.padding_input_data
                self.creator[1].backward(Tensor(grad_kernel), self)

                # grad for input data
                kernel = self.creator[1].data
                grad_input = np.zeros((bs, input_channels, dh+padding*2, dw+padding*2))
                for b in range(bs):
                    for out in range(output_channels):
                        for i in range(0, dh+padding*2-kh+1, self.stride):
                            for j in range(0, dw+padding*2-kw+1, self.stride):
                                grad_input[b, :, i:i+kh, j:j+kw] += kernel[out]*self.grad.data[b, out, i//self.stride, j//self.stride]
                self.creator[0].backward(Tensor(grad_input[:,:,padding:dh+padding,padding:dw+padding]), self)


void conv2d_backward(const py::array_t<float>& grad_output, const py::array_t<int>& feat_input, const py::array_t<int>& kernel, const py::array_t<int>& bias, \
                     py::array_t<float>& input_grad, py::array_t<float>& kernel_grad, py::array_t<float>& bias_grad, int stride, int padding)
{
    py::buffer_info grad_output_buf = grad_output.request();
    py::buffer_info feat_input_buf = feat_input.request();
    py::buffer_info kernel_buf = kernel.request();
    py::buffer_info bias_buf = bias.request();
    
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

  
    size_t gh = grad_output_buf.shape[2];
    size_t gw = grad_output_buf.shape[3];
    auto output = grad_output.unchecked<4>();
    for(int b = 0; b < bs; ++b) {
        for(int i = 0; i < output_channels; ++i) {
            float grad_sum = 0;
            for(int j = 0; j < gh; ++j)
                for(int k = 0; k < gw; ++k)
                    grad_sum += output(b, i, j, k);
            grad_bias[i] += grad_sum;
        }
    }
    
    grad_bias = self.grad.data.sum(axis=(0,2,3))
    self.creator[2].backward(Tensor(grad_bias), self)

    // the buffer info for array
    py::buffer_info grad_output_buf = grad_output.request();
    py::buffer_info grad_input_buf = grad_input.request();
    // the channel & height & width for grad input buf, the same shape with feat_x in forward function
    if (grad_input_buf.shape[0] != 1)  // input grad's shape is assumed to be 1*c*h*w, a typical value is 1*512*37*50
    {
        throw std::runtime_error("we only support batch==1 right now!");
    }

    auto grad_output_cache = grad_output.unchecked<4>();
    auto feat_pos_cache = feat_pos.unchecked<4>();
    auto grad_input_cache = grad_input.mutable_unchecked<4>();

    // the batch size of grad output from upstream
    size_t bs_grad_output = grad_output_buf.shape[0];
    // the channel number of grad input
    size_t c = grad_input_buf.shape[1];

    // the base offset for buffers
    int num_feat_map = roi_size*roi_size;
    // iterator the grad one by one 
    for(int i = 0; i < bs_grad_output; ++i)
    {
        for(int ch = 0; ch < c; ++ch) {
            for(int idx = 0; idx < num_feat_map; ++idx) {
                int y = idx/roi_size;
                int x = idx%roi_size;
            
                // the feat pos has recorded the position where the max value from during forwarding procedure
                int max_feat_pos_y = feat_pos_cache(i, ch, idx, 0);
                int max_feat_pos_x = feat_pos_cache(i, ch, idx, 1);
                // add grad from upstream
                grad_input_cache(0, ch, max_feat_pos_y, max_feat_pos_x) += \
                grad_output_cache(i, ch, y, x);
            }
        }
    }
}

PYBIND11_MODULE(conv_operations, m) {
    m.doc() = "conv2d forward & backward with pybind11";

	m.attr("__version__") = "0.0.1";
    m.def("add", &add, "for test");
    m.def("conv2d_forward_withbias", &conv2d_forward_withbias, "A function do conv2d forward(with bias) operation according to input features & kernel");
    m.def("conv2d_forward_nobias", &conv2d_forward_nobias, "A function do conv2d forward(without bias) operation according to input features & kernel");
    m.def("roi_pooling_backward", &roi_pooling_backward, "A function do roi pooling backward operation according to grad output");
}
