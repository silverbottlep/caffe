#include <glog/logging.h>
#include <leveldb/db.h>
#include <lmdb.h>
#include <stdint.h>
#include <vector>
#include <sys/stat.h>
#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::Datum;
using caffe::BlobProto;
using std::string;
using std::max;
using std::vector;
using boost::filesystem::path;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 4 || argc > 5) {
    LOG(ERROR) << "Usage: compute_image_mean type[0=rgb 1=flow] root_dir filelist output_file ";
    return 1;
  }

	cv::Mat mean_img;
	cv::Mat img;
	BlobProto mean_blob;
	bool is_color;
	bool is_flow;
	int data_size;
	int cv_read_flag;
	int num_channels;
 
	if (atoi(argv[1]) == 0){
		is_color = true;
		is_flow = false;
		cv_read_flag = CV_LOAD_IMAGE_COLOR;
		num_channels = 3;
	}
	else if (atoi(argv[1]) == 1){
		is_color = false;
		is_flow = true;
		cv_read_flag = CV_LOAD_IMAGE_GRAYSCALE;
		num_channels = 1;
	}

	string root_dir(argv[2]);
	std::ifstream infile(argv[3]);
  vector< std::pair< std::pair<std::string, int>, int> > lines_;
  string filename;
  int label, nframes;
	int total_images=0;
  fprintf(stderr, "START!\n");
  while (infile >> filename >> label >> nframes) {
    lines_.push_back(std::make_pair(std::make_pair(filename, label), nframes));
		total_images += nframes;
  }
  int lines_size = lines_.size();
  fprintf(stderr, "%d files %d images\n", lines_size, total_images);
  //LOG(INFO) << "total: " << lines_size << " files " << total_images << " images";

	int image_count=0;
	for (int item_id = 0; item_id < lines_size; ++item_id) {
	//for (int item_id = 0; item_id < 5000; ++item_id) {
		int nframes = lines_[item_id].second;
		for (int frame_id = 0; frame_id < nframes-1; ++frame_id) {
		//for (int frame_id = 0; frame_id < 10; ++frame_id) {
			path framename(root_dir);
			char numstr[7]={0};
			sprintf(numstr,"_f%04d",frame_id+1);
			string numstr_string(numstr);
			framename /= lines_[item_id].first.first; 
			if (!is_flow)
				framename /= lines_[item_id].first.first + numstr_string + ".jpg";
			else
				framename /= lines_[item_id].first.first + numstr_string + "_opt.jpg";
			
			img = cv::imread(framename.string(), cv_read_flag);
			if (item_id==0 && frame_id==0){
				mean_blob.set_num(1);
				mean_blob.set_channels(num_channels);
				mean_blob.set_height(img.rows);
				mean_blob.set_width(img.cols);

				data_size = num_channels * img.rows * img.cols;
				for (int i = 0; i < data_size; ++i) {
					mean_blob.add_data(0.);
				}
				fprintf(stderr, "initialize mean image (%d,%d,%d)\n", 
						num_channels, img.rows, img.cols);
			}

			int blob_idx = 0;
			for (int c = 0; c < num_channels; ++c) {
				for (int h = 0; h < img.rows; ++h) {
					for (int w = 0; w < img.cols; ++w) {
						mean_blob.set_data(blob_idx, mean_blob.data(blob_idx) +
								static_cast<float>(img.at<cv::Vec3b>(h,w)[c]));
						blob_idx++;
					}
				}
			}
			image_count++;
			//fprintf(stderr, "%s\n", framename.c_str());
		}
		fprintf(stderr, "%s(%d/%d)\n", 
				lines_[item_id].first.first.c_str(), image_count, total_images);
	}

	mean_img = img.clone();
	int idx = 0;
	for (int c = 0; c < num_channels; ++c) {
		for (int h = 0; h < mean_img.rows; ++h) {
			for (int w = 0; w < mean_img.cols; ++w) {
				mean_blob.set_data(idx, mean_blob.data(idx) / image_count);
				mean_img.at<cv::Vec3b>(h,w)[c] = static_cast<uchar>(mean_blob.data(idx));
				idx++;
			}
		}
	}
	string jpgname(argv[4]);
	cv::imwrite(jpgname+".jpg",mean_img);

  LOG(INFO) << "Write to " << argv[4];
  WriteProtoToBinaryFile(mean_blob, argv[4]);


  return 0;
}
