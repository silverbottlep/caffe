#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConsilienceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
void ConsilienceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  num_ = bottom[0]->num();
  channels1_ = bottom[0]->channels();
  channels2_ = bottom[1]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
	h_time_w_ = height_ * width_;
  
	//(*top)[0]->Reshape(num, channels*channels, height, width);
	(*top)[0]->Reshape(num_, channels1_*channels2_, 1, 1);
}

template <typename Dtype>
void ConsilienceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
}

template <typename Dtype>
void ConsilienceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(ConsilienceLayer);
#endif

INSTANTIATE_CLASS(ConsilienceLayer);

}  // namespace caffe
