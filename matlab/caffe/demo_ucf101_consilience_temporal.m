rgb_dir = '../../data/ucf101/ucf101_image/';
flow_dir = '../../data/ucf101/ucf101_opt/';
list_file = '../../data/ucf101/test1.txt';
%list_file = '../../data/ucf101/test2.txt';
%list_file = '../../data/ucf101/test3.txt';

%model_def_file = '../../examples/consilience/consilience_temporal_deploy.prototxt';
%model_file = '../../examples/consilience/snapshot/consilience_temporal_iter_40000.caffemodel';
%model_def_file = '../../examples/consilience/consilience_temporal_nonorm_deploy.prototxt';
%model_file = '../../examples/consilience/snapshot/consilience_temporal_nonorm_ft_iter_40000.caffemodel';
model_def_file = '../../examples/consilience/vgg19_consilience_temporal_nonorm_deploy.prototxt';
model_file = '../../examples/consilience/snapshot/vgg19_consilience_temporal_nonorm_ft_iter_40000.caffemodel';
use_gpu = true;
matcaffe_init(use_gpu, model_def_file, model_file);
caffe('set_device',0);

flow_mean = imread('../../data/ucf101/ucf101_flow_mean.binaryproto.jpg');
FLOW_MEAN = single(flow_mean);
image_mean = imread('../../data/ucf101/ucf101_rgb_mean.binaryproto.jpg');
IMAGE_MEAN = single(image_mean(:,:,[3 2 1]));

list_fid = fopen(list_file);
line = fgetl(list_fid);
num_item = 0;
while ischar(line)
	num_item = num_item + 1;
	itemlist{num_item} = textscan(line,'%s %d %d %f %f\n');
	item_name{num_item} = itemlist{num_item}{1};
	item_label(num_item) = itemlist{num_item}{2};
	nframes(num_item) = itemlist{num_item}{3};
	line = fgetl(list_fid);
end
fclose(list_fid);

accuracy = 0;
CROPPED_DIM = 224;
nsamples = 25;
nchannels = 10;
ncrops = 10;
rgb_input = zeros(CROPPED_DIM, CROPPED_DIM, 3, ncrops, 'single');
flow_input = zeros(CROPPED_DIM, CROPPED_DIM, nchannels*2, ncrops, 'single');
scores = single(zeros(101,nsamples*ncrops));

for i=1:num_item
	item = char(item_name{i});
	tic;
	gap = floor(single(nframes(i)-nchannels) / single(nsamples));
	for j=1:nsamples
		frame_num = (j-1)*gap+1;
	
		flow = [];
		for k=1:nchannels
			flow_filename = strcat(flow_dir, item, '/', item, '_f', ...
				num2str(frame_num+k-1,'%04u'), '_opt.jpg');
			im = imread(char(flow_filename));
			im = single(im);
			im = im - FLOW_MEAN;
			flow_x = im(1:size(im,1)/2,:);
			flow_y = im(size(im,1)/2+1:end,:);
			flow(:,:,(k-1)*2+1) = flow_x;
			flow(:,:,(k-1)*2+2) = flow_y;

			if k==1
				rgb_filename = strcat(rgb_dir, item, '/', item, '_f', ...
				num2str(frame_num,'%04u'), '.jpg');
				im = imread(char(rgb_filename));
				im = single(im);
				if size(im,3) == 1
						im = cat(3,im,im,im);
				end
				im = im(:,:,[3 2 1]) - IMAGE_MEAN;
				rgb_input = prepare_image_ucf101(im);
			end
		end
		flow_input = prepare_image_ucf101(flow);

		output_data = caffe('forward', {single(rgb_input); single(flow_input)} );
		scores(:,(j-1)*ncrops+1:(j-1)*ncrops+ncrops) = squeeze(output_data{1});
	end

	s = sum(scores,2)/size(scores,2);
	[max_s, idx] = max(s);
	output_label(i) = idx-1;
	output_score(:,i) = s;

	if item_label(i) == output_label(i)
		accuracy = accuracy+1;
	end

	consilience_temporal_result(i).feat = scores;
	consilience_temporal_result(i).item_name = item;
	consilience_temporal_result(i).nframes = nframes(i);
	consilience_temporal_result(i).label = item_label(i);
	consilience_temporal_result(i).output_label = output_label(i);

	fprintf('processing %s(%d/%d) accuracy: %.2f%% output:(%d/%f) gt:%d \n', ...
			item, i, num_item, (accuracy/i)*100, output_label(i), max_s, item_label(i));
	toc;
end

%save('consilience_temporal_result.mat', 'consilience_temporal_result');
save('vgg19_consilience_temporal_result.mat', 'consilience_temporal_result');
fprintf('total accuracy:%s\n',accuracy/num_item);
