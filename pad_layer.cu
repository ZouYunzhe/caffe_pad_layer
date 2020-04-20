#include <vector>

#include "caffe/layers/pad_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void PadForward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int pad_l, const int pad_r,const int pad_t,const int pad_b) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index;  // Preserve the original value
    int height_out = height_in + pad_t + pad_b;
    int width_out = width_in + pad_l + pad_r;
    int w = i % width_in;
    i /= width_in;
    int h = i % height_in;
    i /= height_in;
    int c = i % channel;
    i /= channel;

    out[((i * channel + c) * height_out + h + pad_t) * width_out + pad_l + w] =
        in[index];
  }
}

template <typename Dtype>
__global__ void PadForwardPadZero(const int count, Dtype* out,
    const int num, const int channel, const int height_out, const int width_out,
    const int pad_l, const int pad_r,const int pad_t,const int pad_b) {
  CUDA_KERNEL_LOOP(index, count) {
    int w = index % width_out;
    int h = (index / width_out) % height_out;
    if (h < pad_t || h > height_out-1-pad_b || w < pad_l || w > width_out-1-pad_r) {
      out[index] = Dtype(0);
    }
  }
}

// No matching PadBackwardPadZero, since no gradient propagates
// through zero padding

template <typename Dtype>
void PadLayer<Dtype>::Forward_gpu(const std::vector<Blob<Dtype>*>& bottom,
    const std::vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int bcount = bottom[0]->count();
  const int tcount = top[0]->count();
  // First, set all data to be zero for the boundary pixels
  // CUDA_CHECK(cudaMemset(top_data, 0, sizeof(Dtype) * tcount));
  // Copy the main body (not yet setting the padding)
  // NOLINT_NEXT_LINE(whitespace/operators)
  PadForward<Dtype><<<CAFFE_GET_BLOCKS(bcount), CAFFE_CUDA_NUM_THREADS>>>(
      bcount, bottom_data, top_data, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
      PAD_L_,PAD_R_,PAD_T_,PAD_B_);
  CUDA_POST_KERNEL_CHECK;

  // Padding
  switch (PAD_TYPE_) {
  case PadParameter::ZERO:
    // NOLINT_NEXT_LINE(whitespace/operators)
    PadForwardPadZero<Dtype><<<CAFFE_GET_BLOCKS(tcount),
                               CAFFE_CUDA_NUM_THREADS>>>(
        tcount, top_data, NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_,
        PAD_L_,PAD_R_,PAD_T_,PAD_B_);
    break;
  
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void PadBackward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int pad_l, const int pad_r,const int pad_t,const int pad_b) {
  CUDA_KERNEL_LOOP(index, count) {
    int i = index;  // Preserve original value
    int height_out = height_in + pad_t + pad_b;
    int width_out = width_in + pad_l + pad_r;
    int w = i % width_in;
    i /= width_in;
    int h = i % height_in;
    i /= height_in;
    int c = i % channel;
    i /= channel;
    out[index] = in[((i * channel + c) * height_out + h + pad_t) *
                    width_out + pad_l + w];
  }
}

template <typename Dtype>
void PadLayer<Dtype>::Backward_gpu(const std::vector<Blob<Dtype>*>& top,
    const std::vector<bool>& propagate_down,
    const std::vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    Dtype* top_diff = top[0]->mutable_gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bcount = bottom[0]->count();
    const int tcount = top[0]->count();
    caffe_gpu_set(bcount, static_cast<Dtype>(0), bottom_diff);
    // In reverse order from Forward_gpu, so ...
    // Padding first. Operate within top to set the gradient in the
    // part to be copied to bottom.
    switch (PAD_TYPE_) {
    case PadParameter::ZERO:
      break;  // No gradient in the padding; it's constant
    }
    // Copy into place
    // NOLINT_NEXT_LINE(whitespace/operators)
    PadBackward<Dtype><<<CAFFE_GET_BLOCKS(bcount), CAFFE_CUDA_NUM_THREADS>>>(
        bcount, top_diff, bottom_diff, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
        PAD_L_,PAD_R_,PAD_T_,PAD_B_);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(PadLayer);

}  // namespace caffe