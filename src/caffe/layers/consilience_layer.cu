#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConsilienceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const Dtype* bottom_data1 = bottom[0]->gpu_data();
  const Dtype* bottom_data2 = bottom[1]->gpu_data();
  for (int n = 0; n < num_; ++n) {
		int M = channels1_;
		int N = channels2_;
		int K = h_time_w_;
		const Dtype* A = bottom_data1 + bottom[0]->offset(n);
		const Dtype* B = bottom_data2 + bottom[1]->offset(n);
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M, N, K,
				(Dtype)1., A, B, (Dtype)0., top_data + (*top)[0]->offset(n));
	}
}

template <typename Dtype>
void ConsilienceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < bottom->size(); ++i) {
    if (propagate_down[i]) {
      caffe_gpu_set((*bottom)[i]->count(), Dtype(0),
                    (*bottom)[i]->mutable_gpu_data());
    }
  }
}


INSTANTIATE_CLASS(ConsilienceLayer);

}  // namespace caffe
