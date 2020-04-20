#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/pad_layer.hpp"
#include "caffe/net.hpp"


namespace caffe {

	template <typename Dtype>
	void PadLayer<Dtype>::LayerSetUp(const std::vector<Blob<Dtype>*>& bottom,
		const std::vector<Blob<Dtype>*>& top) {
		// LayerSetup() handles the number of dimensions; Reshape() handles the sizes.
		// bottom[0] supplies the data
		const PadParameter& param = this->layer_param_.pad_param();

		PAD_TYPE_ = param.type();
		PAD_L_ = param.pad_l();
		PAD_R_ = param.pad_r();
		PAD_T_ = param.pad_t();
		PAD_B_ = param.pad_b();
		CHECK_EQ(bottom.size(), 1) << "Pad Layer takes a single blob as input.";
		CHECK_EQ(top.size(), 1) << "Pad Layer takes a single blob as output.";
		CHECK_EQ(bottom[0]->num_axes(), 4) << "Pad Layer must have four axes.";
		NUM_ = bottom[0]->num();
		CHANNEL_ = bottom[0]->channels();
		HEIGHT_IN_ = bottom[0]->height();
		WIDTH_IN_ = bottom[0]->width();
		HEIGHT_OUT_ = HEIGHT_IN_ + PAD_T_ + PAD_B_;
		WIDTH_OUT_ = WIDTH_IN_ + PAD_L_ + PAD_R_;
	}

	template <typename Dtype>
	void PadLayer<Dtype>::Reshape(const std::vector<Blob<Dtype>*>& bottom,
		const std::vector<Blob<Dtype>*>& top) {
		std::vector<int> shape(4, 0);
		shape[0] = NUM_;
		shape[1] = CHANNEL_;
		shape[2] = HEIGHT_OUT_;
		shape[3] = WIDTH_OUT_;
		top[0]->Reshape(shape);
	}


	template <typename Dtype>
	void PadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* bottom_data = bottom[0]->cpu_data();

		for (int n = 0; n < NUM_; ++n) {
			for (int c = 0; c < CHANNEL_; ++c) {
				// First copy the main body into place
				for (int h = 0; h < HEIGHT_IN_; ++h) {
					// copy the width part
					caffe_copy(WIDTH_IN_,
						bottom_data + bottom[0]->offset(n, c, h, 0),
						top_data + top[0]->offset(n, c, h + PAD_T_, PAD_L_));
				}

				// Now pad, first width, then height. This order may affect the
				// corners
				switch (PAD_TYPE_) {
				case PadParameter::ZERO:
				{
					// Left and right. Loop over the rows not in the vertical padding
					for (int h = PAD_T_; h < HEIGHT_OUT_ - PAD_B_; ++h) {
						// Offset to current row start (in padding of this row)
						int off = top[0]->offset(n, c, h, 0);
						// Left pad
						for (int wdst = 0; wdst < PAD_L_; ++wdst) {
							*(top_data + off + wdst) = static_cast<Dtype>(0);
						}
						// Right
						for (int wdst = WIDTH_OUT_ - PAD_R_; wdst < WIDTH_OUT_; ++wdst) {
							*(top_data + off + wdst) = static_cast<Dtype>(0);
						}
					}
					// Top
					for (int h = 0; h < PAD_T_; ++h) {
						int off = top[0]->offset(n, c, h, 0);
						std::fill(top_data + off, top_data + off + WIDTH_OUT_,
							static_cast<Dtype>(0));
					}
					// Bottom
					for (int h = HEIGHT_OUT_ - PAD_B_; h < HEIGHT_OUT_; ++h) {
						int off = top[0]->offset(n, c, h, 0);
						std::fill(top_data + off, top_data + off + WIDTH_OUT_,
							static_cast<Dtype>(0));
					}
				}
				break;
				}
			}
		}
	}


	template <typename Dtype>
	void PadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		Dtype* top_diff = top[0]->mutable_cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		// Very similar to Forward, except reverse the order.
		for (int n = 0; n < NUM_; ++n) {
			for (int c = 0; c < CHANNEL_; ++c) {
				// First do the padding. We need to reverse the order, and
				// actually need to manipulate the diffs in top, before copying
				// to bottom. First height, then width. This order may affect
				// the corners
				switch (PAD_TYPE_) {
				case PadParameter::ZERO:
					// There's no information in the padding, since it's constant.
					break;
				}  // switch over types

				// Now copy the main body into place
				for (int h = 0; h < HEIGHT_IN_; ++h) {
					// copy the width part
					caffe_copy(WIDTH_IN_,
						top_diff + top[0]->offset(n, c, h + PAD_T_, PAD_L_),
						bottom_diff + bottom[0]->offset(n, c, h, 0));
				}
			}  // c
		}  // n
	}

#ifdef CPU_ONLY
	STUB_GPU(PadLayer);
#endif

	INSTANTIATE_CLASS(PadLayer);
	REGISTER_LAYER_CLASS(Pad);

}  // namespace caffe