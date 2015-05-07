%load('rgb_consilience_result.mat');
%load('rgb_result.mat');
load('rgb_vgg19_consilience_result.mat');
load('rgb_vgg19_result.mat');
load('flow_result.mat');

alpha = 0.1:0.1:0.9;
accuracy = zeros(length(alpha),1);
rgb_accuracy = 0;
flow_accuracy = 0;
total_item = length(rgb_consilience_result);
for i=1:total_item
	label = rgb_consilience_result(i).label;
	rgb_feat = rgb_consilience_result(i).feat;
	%rgb_feat = rgb_result(i).feat;
	
	rgb_feat = sum(rgb_feat,2)/size(rgb_feat,2);
	[max_s, idx] = max(rgb_feat);
	rgb_output_label = idx-1;
	if label == rgb_output_label
		rgb_accuracy = rgb_accuracy + 1;
	end

	flow_feat = flow_result(i).feat;
	flow_feat = sum(flow_feat,2)/size(flow_feat,2);
	[max_s, idx] = max(flow_feat);
	flow_output_label = idx-1;
	if label == flow_output_label
		flow_accuracy = flow_accuracy + 1;
	end

	for j=1:length(alpha)
		feat = rgb_feat.*alpha(j) + flow_feat.*(1-alpha(j));
		[max_s, idx] = max(feat);
		two_output_label(j) = idx-1;
		if label == two_output_label(j)
			accuracy(j) = accuracy(j)+1;
		end
	end
	fprintf('processing %s gt:%d (rgb %d, %.2f) (flow %d, %.2f) (two %d, %.2f)\n', ...
						rgb_consilience_result(i).item_name, label, rgb_output_label, 100*rgb_accuracy/i, ...
						flow_output_label, 100*flow_accuracy/i, two_output_label(5), 100*accuracy(5)/i);
end

fprintf('rgb accuracy = %f\n', rgb_accuracy/total_item);
fprintf('flow accuracy = %f\n', flow_accuracy/total_item);
for i=1:length(alpha)
	fprintf('accuracy(alpha:%f) = %f\n',alpha(i), accuracy(i)/length(rgb_consilience_result));
end
