#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <leveldb/db.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdint.h>
#include <sys/stat.h>
#include <boost/filesystem.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

namespace caffe {

using boost::filesystem::path;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(1073741824, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

//bool ReadFlowTarToDatum(const string& root_dir, const string& filename, const int label, const int start_frame, const int nchannels, const int height, const int width, Datum* datum) {
//  cv::Mat cv_img_x;
//  cv::Mat cv_img_y;
//  int cv_read_flag = CV_LOAD_IMAGE_GRAYSCALE;
//	int num_channels = 2*nchannels;
//	string temp_directory = "/tmp/caffe_readflow/" + filename + "/";
//	string tar_name = root_dir + filename + ".tar";
//
//	int res = mkdir(temp_directory.c_str(), S_IRWXU|S_IRWXG);
//	if(res){
//			LOG(ERROR) << "Could not open or find directory " << temp_directory;
//			return false;
//	}
//
//	string command = "tar xf " + tar_name + " -C " + temp_directory;
//	res = system(command.c_str());
//	if(res){
//			LOG(ERROR) << "Could not extract tarball to " << temp_directory;
//			return false;
//	}
//
//	datum->set_channels(num_channels);
//	for (int i=start_frame; i<start_frame+nchannels; i++){
//		string* datum_string;
//		char numstr[7]={0};
//		sprintf(numstr,"_f%04d",i);
//		string numstr_string(numstr);
//		string framename_x = temp_directory + filename + numstr_string + "_optx.jpg";
//		string framename_y = temp_directory + filename + numstr_string + "_opty.jpg";
//
//		cv::Mat cv_img_origin_x = cv::imread(framename_x, cv_read_flag);
//		cv::Mat cv_img_origin_y = cv::imread(framename_y, cv_read_flag);
//		if (!cv_img_origin_x.data || !cv_img_origin_y.data) {
//			LOG(ERROR) << "Could not open or find file " << framename_x;
//			return false;
//		}
//		if (height > 0 && width > 0) {
//			cv::resize(cv_img_origin_x, cv_img_x, cv::Size(width, height));
//			cv::resize(cv_img_origin_y, cv_img_y, cv::Size(width, height));
//		} else {
//			cv_img_x = cv_img_origin_x;
//			cv_img_y = cv_img_origin_y;
//		}
//		
//		if (i==start_frame){
//			datum->set_height(cv_img_x.rows);
//			datum->set_width(cv_img_x.cols);
//			datum->set_label(label);
//			datum->clear_data();
//			datum->clear_float_data();
//			datum_string = datum->mutable_data();
//		}
//
//		for (int h = 0; h < cv_img_x.rows; ++h) {
//			for (int w = 0; w < cv_img_x.cols; ++w) {
//				datum_string->push_back(
//						static_cast<char>(cv_img_x.at<uchar>(h, w)));
//			}
//		}
//		for (int h = 0; h < cv_img_y.rows; ++h) {
//			for (int w = 0; w < cv_img_y.cols; ++w) {
//				datum_string->push_back(
//						static_cast<char>(cv_img_y.at<uchar>(h, w)));
//			}
//		}
//
//	}
//
//	command = "rm -rf " + temp_directory;
//	res = system(command.c_str());
//	if(res){
//			LOG(ERROR) << "Could not delete files in " << temp_directory;
//			return false;
//	}
//
//  return true;
//}

bool ReadFlowToDatum(const string& root_dir, const string& filename, 
		const int label, const int start_frame, const int nchannels, 
		const int height, const int width, Datum* datum) {
                                       
  cv::Mat cv_img;
  int cv_read_flag = CV_LOAD_IMAGE_GRAYSCALE;
	int num_stacks = 2*nchannels;
	datum->set_channels(num_stacks);

	for (int i=start_frame; i<start_frame+nchannels; i++){
		string* datum_string;
		char numstr[7]={0};
		sprintf(numstr,"_f%04d",i);
		string numstr_string(numstr);
		path framename(root_dir);
		framename /= filename; framename /= filename + numstr_string + "_opt.jpg";

		cv::Mat cv_img_origin = cv::imread(framename.string(), cv_read_flag);
		if (!cv_img_origin.data) {
			LOG(ERROR) << "Could not open or find file " << framename.string();
			return false;
		}
		if (height > 0 && width > 0) {
			cv::resize(cv_img_origin, cv_img, cv::Size(width, height*2));
		} else {
			cv_img = cv_img_origin;
		}
		
		if (i==start_frame){
			datum->set_height(cv_img.rows/2);
			datum->set_width(cv_img.cols);
			datum->set_label(label);
			datum->clear_data();
			datum->clear_float_data();
			datum_string = datum->mutable_data();
		}

		for (int h = 0; h < cv_img.rows; ++h) {
			for (int w = 0; w < cv_img.cols; ++w) {
				datum_string->push_back(
						static_cast<char>(cv_img.at<uchar>(h, w)));
			}
		}
	}

  return true;
}

//bool ReadFlowToDatum(const string& root_dir, const string& filename, 
//		const int label, const int start_frame, const int nchannels, 
//		const int height, const int width, Datum* datum) {
//  cv::Mat cv_img_x;
//  cv::Mat cv_img_y;
//  int cv_read_flag = CV_LOAD_IMAGE_GRAYSCALE;
//	int num_stacks= 2*nchannels;
//	datum->set_channels(num_stacks);
//
//	for (int i=start_frame; i<start_frame+nchannels; i++){
//		string* datum_string;
//		char numstr[7]={0};
//		sprintf(numstr,"_f%04d",i);
//		string numstr_string(numstr);
//		path framename_x(root_dir);
//		path framename_y(root_dir);
//		framename_x /= filename; framename_x /= filename + numstr_string + "_optx.jpg";
//		framename_y /= filename; framename_y /= filename + numstr_string + "_opty.jpg";
//
//		cv::Mat cv_img_origin_x = cv::imread(framename_x.string(), cv_read_flag);
//		cv::Mat cv_img_origin_y = cv::imread(framename_y.string(), cv_read_flag);
//		if (!cv_img_origin_x.data || !cv_img_origin_y.data) {
//			LOG(ERROR) << "Could not open or find file " << framename_x.string();
//			return false;
//		}
//		if (height > 0 && width > 0) {
//			cv::resize(cv_img_origin_x, cv_img_x, cv::Size(width, height));
//			cv::resize(cv_img_origin_y, cv_img_y, cv::Size(width, height));
//		} else {
//			cv_img_x = cv_img_origin_x;
//			cv_img_y = cv_img_origin_y;
//		}
//		
//		if (i==start_frame){
//			datum->set_height(cv_img_x.rows);
//			datum->set_width(cv_img_x.cols);
//			datum->set_label(label);
//			datum->clear_data();
//			datum->clear_float_data();
//			datum_string = datum->mutable_data();
//		}
//
//		for (int h = 0; h < cv_img_x.rows; ++h) {
//			for (int w = 0; w < cv_img_x.cols; ++w) {
//				datum_string->push_back(
//						static_cast<char>(cv_img_x.at<uchar>(h, w)));
//			}
//		}
//		for (int h = 0; h < cv_img_y.rows; ++h) {
//			for (int w = 0; w < cv_img_y.cols; ++w) {
//				datum_string->push_back(
//						static_cast<char>(cv_img_y.at<uchar>(h, w)));
//			}
//		}
//
//	}
//
//  return true;
//}

bool ReadVideoToDatum(const string& filename, const int label, const int nframes,
    const int height, const int width, const bool is_color, Datum* datum) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
	int num_channels = (is_color ? 3 : 1);

	for (int i=0; i<nframes; i++){
		string* datum_string;
		char numstr[7]={0};
		sprintf(numstr,"_f%04d",i+1);
		string numstr_string(numstr);
		string framename = filename + numstr_string + ".jpg";

		cv::Mat cv_img_origin = cv::imread(framename, cv_read_flag);
		if (!cv_img_origin.data) {
			LOG(ERROR) << "Could not open or find file " << framename;
			return false;
		}
		if (height > 0 && width > 0) {
			cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
		} else {
			cv_img = cv_img_origin;
		}
		
		if (i==0){
			datum->set_channels(num_channels);
			datum->set_height(cv_img.rows);
			datum->set_width(cv_img.cols);
			datum->set_label(label);
			datum->clear_data();
			datum->clear_float_data();
			datum_string = datum->mutable_data();
		}

		if (is_color) {
			for (int c = 0; c < num_channels; ++c) {
				for (int h = 0; h < cv_img.rows; ++h) {
					for (int w = 0; w < cv_img.cols; ++w) {
						datum_string->push_back(
								static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
					}
				}
			}
		} else {  // Faster than repeatedly testing is_color for each pixel w/i loop
			for (int h = 0; h < cv_img.rows; ++h) {
				for (int w = 0; w < cv_img.cols; ++w) {
					datum_string->push_back(
							static_cast<char>(cv_img.at<uchar>(h, w)));
				}
			}
		}

	}

  return true;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);

  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }

  int num_channels = (is_color ? 3 : 1);
  datum->set_channels(num_channels);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  if (is_color) {
    for (int c = 0; c < num_channels; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          datum_string->push_back(
            static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
        }
      }
    }
  } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(
          static_cast<char>(cv_img.at<uchar>(h, w)));
        }
      }
  }
  return true;
}

leveldb::Options GetLevelDBOptions() {
  // In default, we will return the leveldb option and set the max open files
  // in order to avoid using up the operating system's limit.
  leveldb::Options options;
  options.max_open_files = 100;
  return options;
}

// Verifies format of data stored in HDF5 file and reshapes blob accordingly.
template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob) {
  // Verify that the number of dimensions is in the accepted range.
  herr_t status;
  int ndims;
  status = H5LTget_dataset_ndims(file_id, dataset_name_, &ndims);
  CHECK_GE(status, 0) << "Failed to get dataset ndims for " << dataset_name_;
  CHECK_GE(ndims, min_dim);
  CHECK_LE(ndims, max_dim);

  // Verify that the data format is what we expect: float or double.
  std::vector<hsize_t> dims(ndims);
  H5T_class_t class_;
  status = H5LTget_dataset_info(
      file_id, dataset_name_, dims.data(), &class_, NULL);
  CHECK_GE(status, 0) << "Failed to get dataset info for " << dataset_name_;
  CHECK_EQ(class_, H5T_FLOAT) << "Expected float or double data";

  blob->Reshape(
    dims[0],
    (dims.size() > 1) ? dims[1] : 1,
    (dims.size() > 2) ? dims[2] : 1,
    (dims.size() > 3) ? dims[3] : 1);
}

template <>
void hdf5_load_nd_dataset<float>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<float>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_float(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read float dataset " << dataset_name_;
}

template <>
void hdf5_load_nd_dataset<double>(hid_t file_id, const char* dataset_name_,
        int min_dim, int max_dim, Blob<double>* blob) {
  hdf5_load_nd_dataset_helper(file_id, dataset_name_, min_dim, max_dim, blob);
  herr_t status = H5LTread_dataset_double(
    file_id, dataset_name_, blob->mutable_cpu_data());
  CHECK_GE(status, 0) << "Failed to read double dataset " << dataset_name_;
}

template <>
void hdf5_save_nd_dataset<float>(
    const hid_t file_id, const string dataset_name, const Blob<float>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_float(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make float dataset " << dataset_name;
}

template <>
void hdf5_save_nd_dataset<double>(
    const hid_t file_id, const string dataset_name, const Blob<double>& blob) {
  hsize_t dims[HDF5_NUM_DIMS];
  dims[0] = blob.num();
  dims[1] = blob.channels();
  dims[2] = blob.height();
  dims[3] = blob.width();
  herr_t status = H5LTmake_dataset_double(
      file_id, dataset_name.c_str(), HDF5_NUM_DIMS, dims, blob.cpu_data());
  CHECK_GE(status, 0) << "Failed to make double dataset " << dataset_name;
}

}  // namespace caffe
