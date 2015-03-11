src_dir = '../../data/ucf101/ucf101_image/';
list_file = '../../data/ucf101/test1.txt';

model_def_file = '../../examples/twostream/spatialnet_ft_deploy.prototxt';
model_file = '../../examples/twostream/snapshot/spatialnet_ft_iter_35000.caffemodel';
use_gpu = true;
matcaffe_init(use_gpu, model_def_file, model_file);

image_mean = imread('../../data/ucf101/ucf101_rgb_mean.binaryproto.jpg');
IMAGE_MEAN = single(image_mean(:,:,[3 2 1]));

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
nsamples = 25;
for i=1:num_item
	item = char(item_name{i});

	images = zeros(CROPPED_DIM, CROPPED_DIM, 3, nsamples*10, 'single');
	gap = floor(single(nframes(i)) / single(nsamples));
	for j=1:nsamples
		frame_num = (j-1)*gap+1;
		filename = strcat(src_dir, item, '/', item, '_f', ...
			num2str(frame_num,'%04u'), '.jpg');
		
		im = imread(char(filename));
		im = single(im);
		if size(im,3) == 1
				im = cat(3,im,im,im);
		end
		im = im(:,:,[3 2 1]) - IMAGE_MEAN;
		images(:,:,:,(j-1)*10+1:(j-1)*10+10) = prepare_image_ucf101(im);
	end

	output_data = caffe('forward', {images});
	scores = squeeze(output_data{1});
	s = sum(scores,2)/size(scores,2);
	[max_s, idx] = max(s);
	output_label(i) = idx-1;
	output_score(:,i) = s;

	if item_label(i) == output_label(i)
		accuracy = accuracy+1;
	end

	fprintf('processing %s(%d/%d) accuracy: %.2f%% output:(%d/%f) gt:%d \n', ...
			item, i, num_item, (accuracy/i)*100, output_label(i), max_s, item_label(i));
end

fprintf('total accuracy:%s\n',accuracy/num_item);
%save('result.mat','output_label','output_score','accuracy');
