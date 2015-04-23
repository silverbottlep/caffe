#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <sys/stat.h>
#include <boost/filesystem.hpp>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

using boost::filesystem::path;

	template <typename Dtype>
	VideoDataLayer<Dtype>::~VideoDataLayer<Dtype>() {
		this->JoinPrefetchThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  VideoDataParameter video_data_param = this->layer_param_.video_data_param();
  const int new_height = video_data_param.new_height();
  const int new_width  = video_data_param.new_width();
  const int num_channels = video_data_param.num_channels();
	string root_dir = video_data_param.root_dir();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames, labels and nframes
  const string& source = this->layer_param_.video_data_param().source();
  CHECK_GT(source.size(), 0);
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good())
      << "Could not open image list (filename: \""+ source + "\")";
  string filename;
  int label, nframes;
	float min, max;
	struct data_item video_item;
  while (infile >> filename >> label >> nframes >> min >> max) {
		video_item.filename = filename;
		video_item.label = label;
		video_item.nframes = nframes;
		video_item.min = min;
		video_item.max = max;
    lines_.push_back(video_item);
  }

  if (this->layer_param_.video_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.video_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.video_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  CHECK(!lines_.empty())
      << "Image list is empty (filename: \"" + source + "\")";
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
	if (video_data_param.input_type() == caffe::VideoDataParameter_InputType_IMAGE) {
		path framename(root_dir);
		framename /= lines_[lines_id_].filename;
		framename /= lines_[lines_id_].filename + "_f0001.jpg";
		CHECK(ReadImageToDatum(framename.string(), lines_[lines_id_].label, 
					new_height, new_width, &datum));
	}
	else {
		CHECK(ReadFlowToDatum(root_dir, lines_[lines_id_].filename, 
			lines_[lines_id_].label, 1, num_channels, 
			new_height, new_width, &datum));
	}

  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.video_data_param().batch_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size,
                                 crop_size);
  } else {
    (*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
                       datum.width());
    this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(),
        datum.width());
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void VideoDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  VideoDataParameter video_data_param = this->layer_param_.video_data_param();
  const int batch_size = video_data_param.batch_size();
  const int new_height = video_data_param.new_height();
  const int new_width = video_data_param.new_width();
  const int num_channels = video_data_param.num_channels();
	string root_dir = video_data_param.root_dir();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK_GT(lines_size, lines_id_);
		int nframes = lines_[lines_id_].nframes;
		int start_frame = (rand()%(nframes-num_channels-1))+1;
		
		if (video_data_param.input_type() == caffe::VideoDataParameter_InputType_IMAGE) {
			path framename(root_dir);
			char numstr[7]={0};
			sprintf(numstr,"_f%04d",start_frame);
			string numstr_string(numstr);
			framename /= lines_[lines_id_].filename;
			framename /= lines_[lines_id_].filename + numstr_string + ".jpg";
			CHECK(ReadImageToDatum(framename.string(), lines_[lines_id_].label, 
						new_height, new_width, &datum));
			
			struct transform_param t_param;
			this->data_transformer_.Transform(item_id, datum, this->mean_, top_data, &t_param);
		}
		else {
			if (!ReadFlowToDatum(root_dir,lines_[lines_id_].filename, 
						lines_[lines_id_].label, start_frame, num_channels, 
						new_height, new_width, &datum)){
				continue;
			}
			this->data_transformer_.FlowTransform(item_id, datum, lines_[lines_id_].min, lines_[lines_id_].max, this->mean_, top_data);
		}

	//int height = this->prefetch_data_.height();
	//int width = this->prefetch_data_.width();

//		for (int h = 100; h < 113; ++h) {
//			for (int w = 100; w < 113; ++w) {
//				int top_index = (item_id*height + h) * width + w;
//				int data_index = h*width + w;
//				//top_data[top_index] =
//					//static_cast<Dtype>(static_cast<uint8_t>(flow_data[data_index]));
//			}
//			if (item_id == 0){
//				LOG(INFO) << top_data[(item_id*height + h) * width + 0]
//					<< " " << top_data[(item_id*height + h) * width + 1]
//					<< " " << top_data[(item_id*height + h) * width + 2]
//					<< " " << top_data[(item_id*height + h) * width + 3]
//					<< " " << top_data[(item_id*height + h) * width + 4]
//					<< " " << top_data[(item_id*height + h) * width + 5]
//					<< " " << top_data[(item_id*height + h) * width + 6]
//					<< " " << top_data[(item_id*height + h) * width + 7]
//					<< " " << top_data[(item_id*height + h) * width + 8]
//					<< " " << top_data[(item_id*height + h) * width + 9]
//					<< " " << top_data[(item_id*height + h) * width + 10]
//					<< " " << top_data[(item_id*height + h) * width + 11]
//					<< " " << top_data[(item_id*height + h) * width + 12];
//			}
//		}
    top_label[item_id] = datum.label();
		//LOG(INFO) << lines_[lines_id_].filename << " label:" << top_label[item_id] << " nframes: " << nframes << " start_frame: " << start_frame << " min, max: " << lines_[lines_id_].min << ", " << lines_[lines_id_].max;
		
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.video_data_param().shuffle()) {
        ShuffleImages();
      }
    }
	}
}

INSTANTIATE_CLASS(VideoDataLayer);

}  // namespace caffe
