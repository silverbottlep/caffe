#include <algorithm>
#include <cfloat>
#include <vector>
#include <npp.h>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* squared_data = squared_.mutable_gpu_data();
  int n = bottom[0]->num();
  int d = bottom[0]->count() / n;
  //caffe_gpu_powx(n*d, bottom_data, Dtype(2), squared_data);
  for (int i=0; i<n; ++i) {
    //caffe_gpu_asum<Dtype>(d, squared_data+i*d, &normsqr);
    //caffe_gpu_scale<Dtype>(d, pow(10,0.5)*pow(normsqr, -0.5), bottom_data+i*d, top_data+i*d);
    caffe_gpu_scale<Dtype>(d, Dtype(1.0), bottom_data+i*d, top_data+i*d);
    caffe_gpu_add_scalar<Dtype>(d, Dtype(1.0), top_data+i*d);
  }

//  for (int i=0; i<n; ++i) {
//    caffe_gpu_scale<Dtype>(d, Dtype(1.0/50.0), bottom_data+i*d, top_data+i*d);
//    caffe_gpu_add_scalar<Dtype>(d, Dtype(1.0/512.0), top_data+i*d);
//  }
}

//template <typename Dtype>
//void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//    vector<Blob<Dtype>*>* top) {
//  const Dtype* bottom_data = bottom[0]->gpu_data();
//  Dtype* top_data = (*top)[0]->mutable_gpu_data();
//  Dtype* squared_data = squared_.mutable_gpu_data();
//  Dtype normsqr;
//  int n = bottom[0]->num();
//  int d = bottom[0]->count() / n;
//	Npp32f min, max;
//	int bytes;
//  for (int i=0; i<n; ++i) {
//	nppsMinMaxGetBufferSize_32f(d,&bytes);
//	Npp8u * scratch = nppsMalloc_8u(bytes*4);
//		nppsMinMax_32f((Npp32f*)(bottom_data + i*d), d, &min, &max, scratch);
//    caffe_gpu_scale<Dtype>(d, Dtype(1.0/50.0), bottom_data+i*d, top_data+i*d);
//    caffe_gpu_add_scalar<Dtype>(d, Dtype(1.0/512.0), top_data+i*d);
//		LOG(INFO) << "Min: " << (float)min << " Max: " << (float)max;
//	nppsFree(scratch);
//  }
//
////  for (int i=0; i<n; ++i) {
////    caffe_gpu_scale<Dtype>(d, Dtype(1.0/50.0), bottom_data+i*d, top_data+i*d);
////    caffe_gpu_add_scalar<Dtype>(d, Dtype(1.0/512.0), top_data+i*d);
////  }
////  caffe_gpu_powx(n*d, bottom_data, Dtype(2), squared_data);
////  for (int i=0; i<n; ++i) {
////    caffe_gpu_asum<Dtype>(d, squared_data+i*d, &normsqr);
////    caffe_gpu_scale<Dtype>(d, pow(normsqr, -0.5), bottom_data+i*d, top_data+i*d);
////    caffe_gpu_add_scalar<Dtype>(d, 1, top_data+i*d);
////  }
//}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  //const Dtype* top_data = top[0]->gpu_data();
  //const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  //int n = top[0]->num();
  //int d = top[0]->count() / n;
  //Dtype a;
  //for (int i=0; i<n; ++i) {
    //caffe_gpu_dot(d, top_data+i*d, top_diff+i*d, &a);
    //caffe_gpu_scale(d, Dtype(1.0), top_data+i*d, bottom_diff+i*d);
    //caffe_gpu_sub(d, top_diff+i*d, bottom_diff+i*d, bottom_diff+i*d);
    //caffe_gpu_dot(d, bottom_data+i*d, bottom_data+i*d, &a);
    //caffe_gpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff+i*d, bottom_diff+i*d);
  //}
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
}

INSTANTIATE_CLASS(NormalizeLayer);


}  // namespace caffe
