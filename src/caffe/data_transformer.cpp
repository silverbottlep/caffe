#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

float rgb_mean_value[3] = {104, 117, 123};
template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data,
																			 struct transform_param *t_param) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

	float *mean_value;
	if (channels==3){
		mean_value = rgb_mean_value;
	}
	t_param->h_off = 0;
	t_param->w_off = 0;
	t_param->mirrored = 0;
	if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
		t_param->h_off = h_off;
		t_param->w_off = w_off;
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean_value[c]) * scale;
                //(datum_element - mean[data_index]) * scale;
          }
        }
      }
			t_param->mirrored = 1;
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean_value[c]) * scale;
                //(datum_element - mean[data_index]) * scale;
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::Transform2(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data,
																			 struct transform_param &t_param) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

	float *mean_value;
	if (channels==3){
		mean_value = rgb_mean_value;
	}
	if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off = t_param.h_off;
		int w_off = t_param.w_off;
    if (mirror && t_param.mirrored) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean_value[c]) * scale;
                //(datum_element - mean[data_index]) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
            transformed_data[top_index] =
                (datum_element - mean_value[c]) * scale;
                //(datum_element - mean[data_index]) * scale;
          }
        }
      }
    }
  } else {
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]));
        transformed_data[j + batch_item_id * size] =
            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::ConsilienceTransform(const int batch_item_id, 
					const Datum& flow_datum, const Dtype* mean, Dtype* transformed_flow_data, 
					struct transform_param& t_param) {
  const string& flow_data = flow_datum.data();
  const int channels = flow_datum.channels();
  const int height = flow_datum.height();
  const int width = flow_datum.width();
  const int size = flow_datum.channels() * flow_datum.height() * flow_datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();

	float *mean_value;
	if (channels==3){
		mean_value = rgb_mean_value;
	}
	if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size) {
    CHECK(flow_data.size()) << "Image cropping only support uint8 data";
    int h_off = t_param.h_off;
		int w_off = t_param.w_off;
    if (mirror && t_param.mirrored) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int mean_index = ((c%2) * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            transformed_flow_data[top_index] =
                static_cast<Dtype>(static_cast<uint8_t>
										(flow_data[data_index] - mean_value[c]));
										//(flow_data[data_index] - mean[mean_index]));
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int mean_index = ((c%2) * height + h + h_off) * width + w + w_off;
            transformed_flow_data[top_index] =
                static_cast<Dtype>(static_cast<uint8_t>(
										flow_data[data_index] - mean_value[c]));
											//flow_data[data_index] - mean[mean_index]));
          }
        }
      }
    }
  } else {
    // NOTICE!! WE SHOULD CONSIDER MEAN INDEX HERE LATER
    // we will prefer to use data() first, and then try float_data()
    if (flow_data.size()) {
      for (int j = 0; j < size; ++j) {
        transformed_flow_data[j + batch_item_id * size] =
            static_cast<Dtype>(static_cast<uint8_t>(
									flow_data[j]) - mean[j]);
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_flow_data[j + batch_item_id * size] =
            flow_datum.float_data(j) - mean[j];
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::ConsilienceRescaleTransform(const int batch_item_id, 
					const Datum& flow_datum, float min, float max, const Dtype* mean, 
					Dtype* transformed_flow_data, struct transform_param& t_param) {
  const string& flow_data = flow_datum.data();
  const int channels = flow_datum.channels();
  const int height = flow_datum.height();
  const int width = flow_datum.width();
  const int size = flow_datum.channels() * flow_datum.height() * flow_datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();
	Dtype flow_scale = (max - min)/255;

	if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }
  if (crop_size) {
    CHECK(flow_data.size()) << "Image cropping only support uint8 data";
    int h_off = t_param.h_off;
		int w_off = t_param.w_off;
    if (mirror && t_param.mirrored) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            //int mean_index = ((c%2) * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(flow_data[data_index]))*flow_scale + min;
            transformed_flow_data[top_index] =
                (datum_element) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            //int mean_index = ((c%2) * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(flow_data[data_index]))*flow_scale + min;
						transformed_flow_data[top_index] =
                (datum_element) * scale;
          }
        }
      }
    }
  } else {
    // NOTICE!! WE SHOULD CONSIDER MEAN INDEX HERE LATER
    // we will prefer to use data() first, and then try float_data()
    if (flow_data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(flow_data[j]))*flow_scale + min;
        transformed_flow_data[j + batch_item_id * size] =
            (datum_element) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_flow_data[j + batch_item_id * size] =
            (flow_datum.float_data(j)*flow_scale + min) * scale;
      }
    }
  }
}

template<typename Dtype>
void DataTransformer<Dtype>::FlowTransform(const int batch_item_id,
                                       const Datum& datum,
																			 float min, float max,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();
	Dtype flow_scale = (max - min)/255;

	if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
               << "set at the same time.";
  }

  if (crop_size) {
    CHECK(data.size()) << "Image cropping only support uint8 data";
    int h_off, w_off;
    // We only do random crop when we do training.
    if (phase_ == Caffe::TRAIN) {
      h_off = Rand() % (height - crop_size);
      w_off = Rand() % (width - crop_size);
    } else {
      h_off = (height - crop_size) / 2;
      w_off = (width - crop_size) / 2;
    }
    if (mirror && Rand() % 2) {
      // Copy mirrored version
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int mean_index = ((c%2) * height + h + h_off) * width + w + w_off;
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + (crop_size - 1 - w);
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]))*flow_scale + min;
            transformed_data[top_index] =
                (datum_element - mean[mean_index]) * scale;
          }
        }
      }
    } else {
      // Normal copy
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            int top_index = ((batch_item_id * channels + c) * crop_size + h)
                * crop_size + w;
            int data_index = (c * height + h + h_off) * width + w + w_off;
            int mean_index = ((c%2) * height + h + h_off) * width + w + w_off;
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[data_index]))*flow_scale + min;
						transformed_data[top_index] =
                (datum_element - mean[mean_index]) * scale;
          }
        }
      }
    }
  } else {
    // NOTICE!! WE SHOULD CONSIDER MEAN INDEX HERE LATER
    // we will prefer to use data() first, and then try float_data()
    if (data.size()) {
      for (int j = 0; j < size; ++j) {
        Dtype datum_element =
            static_cast<Dtype>(static_cast<uint8_t>(data[j]))*flow_scale + min;
        transformed_data[j + batch_item_id * size] =
            (datum_element) * scale;
//        transformed_data[j + batch_item_id * size] =
//            (datum_element - mean[j]) * scale;
      }
    } else {
      for (int j = 0; j < size; ++j) {
        transformed_data[j + batch_item_id * size] =
            (datum.float_data(j)*flow_scale + min) * scale;
//        transformed_data[j + batch_item_id * size] =
//            (datum.float_data(j) - mean[j]) * scale;
      }
    }
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
