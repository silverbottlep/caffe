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
	ConsilienceDataLayer<Dtype>::~ConsilienceDataLayer<Dtype>() {
		this->JoinPrefetchThread();
}

template <typename Dtype>
void ConsilienceDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  ConsilienceDataParameter consilience_data_param = this->layer_param_.consilience_data_param();
  const int new_height = consilience_data_param.new_height();
  const int new_width  = consilience_data_param.new_width();
	string image_dir = consilience_data_param.image_dir();
	string flow_dir = consilience_data_param.flow_dir();
	const int flow_size  = consilience_data_param.flow_size();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames, labels and nframes
  const string& source = this->layer_param_.consilience_data_param().source();
  CHECK_GT(source.size(), 0);
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  CHECK(infile.good())
      << "Could not open image list (filename: \""+ source + "\")";
  string filename;
  int label, nframes;
  while (infile >> filename >> label >> nframes) {
    lines_.push_back(std::make_pair(std::make_pair(filename, label), nframes));
    //lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.consilience_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.consilience_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.consilience_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  CHECK(!lines_.empty())
      << "Image list is empty (filename: \"" + source + "\")";
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  Datum flow_datum;
	struct transform_param t_param;
	t_param.h_off = 0;
	t_param.w_off = 0;
	t_param.mirrored = 0;
	path framename(image_dir);
	framename /= lines_[lines_id_].first.first; 
	framename /= lines_[lines_id_].first.first + "_f0001.jpg";
	CHECK(ReadImageToDatum(framename.string(), lines_[lines_id_].first.second, 
				new_height, new_width, &datum));
		
	CHECK(ReadFlowMagnitude(flow_dir, lines_[lines_id_].first.first, 
		1, new_height, new_width, &flow_datum, &t_param, flow_size));

  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.consilience_data_param().batch_size();
  if (crop_size > 0) {
    (*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size,
                                 crop_size);
		// flow_data
    (*top)[2]->Reshape(batch_size, 1, flow_size, flow_size);
    this->prefetch_data2_.Reshape(batch_size, 1, flow_size, flow_size);
  } else {
    (*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
                       datum.width());
    this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(),
        datum.width());
    (*top)[2]->Reshape(batch_size, 1, flow_size, flow_size);
    this->prefetch_data2_.Reshape(batch_size, 1, flow_size, flow_size);
  }
  LOG(INFO) << "output data1 size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  LOG(INFO) << "output data2 size: " << (*top)[2]->num() << ","
      << (*top)[2]->channels() << "," << (*top)[2]->height() << ","
      << (*top)[2]->width();
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
void ConsilienceDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ConsilienceDataLayer<Dtype>::InternalThreadEntry() {
  Datum datum;
  Datum flow_datum;
  CHECK(this->prefetch_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_data2 = this->prefetch_data2_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  ConsilienceDataParameter consilience_data_param = this->layer_param_.consilience_data_param();
  const int batch_size = consilience_data_param.batch_size();
  const int new_height = consilience_data_param.new_height();
  const int new_width = consilience_data_param.new_width();
  const int num_channels = consilience_data_param.num_channels();
	string image_dir= consilience_data_param.image_dir();
	string flow_dir= consilience_data_param.flow_dir();
	const int flow_height = this->prefetch_data2_.height();
	const int flow_width = this->prefetch_data2_.width();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK_GT(lines_size, lines_id_);
		int nframes = lines_[lines_id_].second;
		int start_frame = (rand()%(nframes-num_channels-1))+1;
		struct transform_param t_param;
		
		path framename(image_dir);
		char numstr[7]={0};
		sprintf(numstr,"_f%04d",start_frame);
		string numstr_string(numstr);
		framename /= lines_[lines_id_].first.first; 
		framename /= lines_[lines_id_].first.first + numstr_string + ".jpg";
		CHECK(ReadImageToDatum(framename.string(), lines_[lines_id_].first.second, 
					new_height, new_width, &datum));
		this->data_transformer_.Transform(item_id, datum, this->mean_, top_data, &t_param);

		// read optical flow image, crop it, mirroring, resize it to 13,13(conv5)
		CHECK(ReadFlowMagnitude(flow_dir, lines_[lines_id_].first.first, 
					start_frame, new_height, new_width, &flow_datum, &t_param, flow_height));
		const string& flow_data = flow_datum.data();
//		if (item_id == 0){
//			LOG(INFO) << lines_[lines_id_].first.first << " label:" << datum.label() << " nframes: " << nframes << " start_frame: " << start_frame << " h_off:" << t_param.h_off << " w_off:" << t_param.w_off << " mirrored:" << t_param.mirrored << " flow_size: " << flow_height;
//		}
		for (int h = 0; h < flow_height; ++h) {
			for (int w = 0; w < flow_width; ++w) {
				int top_index = (item_id*flow_height + h) * flow_width + w;
				int data_index = h*flow_width + w;
				top_data2[top_index] =
					static_cast<Dtype>(static_cast<uint8_t>(flow_data[data_index]));
			}
//			if (item_id == 0){
//				LOG(INFO) << top_data2[(item_id*flow_height + h) * flow_width + 0]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 1]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 2]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 3]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 4]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 5]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 6]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 7]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 8]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 9]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 10]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 11]
//					<< " " << top_data2[(item_id*flow_height + h) * flow_width + 12];
//			}
		}
//		this->data_transformer_.ConsilienceTransform(item_id, flow_datum, 
//						top_data2, t_param);
//		LOG(INFO) << lines_[lines_id_].first.first << " label:" << datum.label() << " nframes: " << nframes << " start_frame: " << start_frame << " h_off:" << t_param.h_off
//			<< " w_off:" << t_param.w_off << " mirrored:" << t_param.mirrored;
    top_label[item_id] = datum.label();
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.consilience_data_param().shuffle()) {
        ShuffleImages();
      }
    }
	}
}

INSTANTIATE_CLASS(ConsilienceDataLayer);

}  // namespace caffe
