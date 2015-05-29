load('flow_result.mat');
load('rgb_vgg19_result.mat');
%load('rgb_vgg19_consilience_result.mat');
load('vgg19_consilience_d1024_result.mat');
%load('rgb_result.mat');
%load('rgb_consilience_result.mat');
%load('consilience_result_d2048.mat');

triple_accuracy  = zeros(11,11,11);
total_item = length(consilience_result);
for i=1:total_item
	label = consilience_result(i).label;
	fprintf('processing %s gt:%d\n', consilience_result(i).item_name, label);

	%rgb_feat = rgb_consilience_result(i).feat; rgb_feat = sum(rgb_feat,2)/size(rgb_feat,2);
	rgb_feat = rgb_result(i).feat; rgb_feat = sum(rgb_feat,2)/size(rgb_feat,2);
	flow_feat = flow_result(i).feat; flow_feat = sum(flow_feat,2)/size(flow_feat,2);
	consilience_feat = consilience_result(i).feat; cons_feat = sum(consilience_feat,2)/250;

	rgb_feat = rgb_feat/norm(rgb_feat);
	flow_feat = flow_feat/norm(flow_feat);
	cons_feat = cons_feat/norm(cons_feat);

  for r=0:10
  for f=0:10
  for c=0:10
    triple_feat = rgb_feat*r + flow_feat*f + cons_feat*c;
    [triple_max_s, idx] = max(triple_feat);
    triple_output_label = idx-1;
    if label == triple_output_label
      triple_accuracy(r+1,f+1,c+1) = triple_accuracy(r+1,f+1,c+1)+1;
      %triple_accuracy(r,f,c) = triple_accuracy(r,f,c)+1;
    end
  end
  end
  end

end
triple_accuracy = triple_accuracy./total_item;
