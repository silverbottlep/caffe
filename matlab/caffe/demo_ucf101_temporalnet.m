src_dir = '../../data/ucf101/ucf101_opt/';
list_file = '../../data/ucf101/test1.txt';

model_def_file = '../../examples/twostream/temporalnet_deploy.prototxt';
model_file = '../../examples/twostream/snapshot/temporalnet_iter_180000.caffemodel';
use_gpu = true;
matcaffe_init(use_gpu, model_def_file, model_file, 2);

flow_mean = imread('../../data/ucf101/ucf101_flow_mean.binaryproto.jpg');
FLOW_MEAN = single(flow_mean);

list_fid = fopen(list_file);
line = fgetl(list_fid);
num_item = 0;
while ischar(line)
	num_item = num_item + 1;
	itemlist{num_item} = textscan(line,'%s %d %d\n');
	item_name{num_item} = itemlist{num_item}{1};
	item_label(num_item) = itemlist{num_item}{2};
	nframes(num_item) = itemlist{num_item}{3};
	line = fgetl(list_fid);
end
fclose(list_fid);

accuracy = 0;
CROPPED_DIM = 224;
IMAGE_DIM = 256;
nsamples = 25;
nchannels = 10;
input = zeros(CROPPED_DIM, CROPPED_DIM, nchannels*2, nsamples*10, 'single');

for i=1:num_item
	item = char(item_name{i});

	gap = floor(single(nframes(i)-nchannels) / single(nsamples));
	for j=1:nsamples
		frame_num = (j-1)*gap+1;
	
		flow = [];
		for k=1:nchannels
			filename = strcat(src_dir, item, '/', item, '_f', ...
				num2str(frame_num+k-1,'%04u'), '_opt.jpg');
			im = imread(char(filename));
			im = single(im);
			im = im - FLOW_MEAN;
			flow_x = im(1:size(im,1)/2,:);
			flow_y = im(size(im,1)/2+1:end,:);
			flow(:,:,(k-1)*2+1) = flow_x;
			flow(:,:,(k-1)*2+2) = flow_y;

		end
%		center_x = floor( (size(flow,2)-IMAGE_DIM+1) / 2)+1;
%		flow = flow(:,center_x:center_x+IMAGE_DIM-1,:);
		images{j} = prepare_image_ucf101(flow);

%		temp = prepare_image_ucf101(flow);
%		images(:,:,:,(j-1)*10+1:(j-1)*10+10) = temp;
%		for k=1:10
%			subplot(3,4,k); imagesc(temp(:,:,(k-1)*2+1,5)); colorbar;
%		end
%		pause;
	end
	
	for j=1:nsamples
		input(:,:,:,(j-1)*10+1:(j-1)*10+10) = images{j};
	end

	output_data = caffe('forward', {input});
	scores = squeeze(output_data{1});
	s = sum(scores,2)/size(scores,2);
	[max_s, idx] = max(s);
	output_label(i) = idx-1;
	output_score(:,i) = s;

	if item_label(i) == output_label(i)
		accuracy = accuracy+1;
	end

	flow_result(i).feat = scores;
	flow_result(i).item_name = item;
	flow_result(i).nframes = nframes(i);
	flow_result(i).label = item_label(i);
	flow_result(i).output_label = output_label(i);

	fprintf('processing %s(%d/%d) accuracy: %.2f%% output:(%d/%f) gt:%d \n', ...
			item, i, num_item, (accuracy/i)*100, output_label(i), max_s, item_label(i));
end

save('flow_result.mat', 'flow_result');
fprintf('total accuracy:%s\n',accuracy/num_item);
