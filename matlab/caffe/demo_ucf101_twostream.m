load('rgb_result.mat');
load('flow_result.mat');

alpha = 0.1:0.1:0.9;
accuracy = zeros(length(alpha),1);
for i=1:length(rgb_result)
	label = rgb_result(i).label;
	rgb_feat = rgb_result(i).feat_prob;
	rgb_feat = sum(rgb_feat,2)/size(rgb_feat,2);
	flow_feat = flow_result(i).feat;
	flow_feat = sum(flow_feat,2)/size(flow_feat,2);
	for j=1:length(alpha)
		feat = rgb_feat.*alpha(j) + flow_feat.*(1-alpha(j));
		[max_s, idx] = max(feat);
		output_label = idx-1;
		if label == output_label
			accuracy(j) = accuracy(j)+1;
		end
	end
	fprintf('processing %s \n',rgb_result(i).item_name);
end

for i=1:length(alpha)
	fprintf('accuracy(alpha:%f) = %f\n',alpha(i), accuracy(i)/length(rgb_result));
end

