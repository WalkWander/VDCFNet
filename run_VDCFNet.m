function res = run_VDCFNet(subS, rp, bSaveImage, param)
init_rect = subS.init_rect;
img_files = subS.s_frames;
num_frame = numel(img_files);
result = repmat(init_rect,[num_frame, 1]);
if nargin < 4
    param = {};
end
addpath(genpath('/path/to/matconvnet-master'));
vl_setupnn();
im = vl_imreadjpeg(img_files);

param.lambda = 1e-4;
[state, ~] = VDCFNet_initialize(im{1}, init_rect, param);
tic;
for frame = 2:num_frame
    [state, region] = VDCFNet_update(state, im{frame}, frame, result);
    result(frame,:) = region;
end
time = toc;
res.type = 'rect';
res.res = result;
res.fps = num_frame / time;
end


function [state, location] = VDCFNet_initialize(I, region, param)
state.gpu = true;
state.visual = true;

state.lambda = 1e-4;
state.padding = 1.5;
state.output_sigma_factor = 0.1;
state.interp_factor = 0.002;

state.output=121;
state.upsize=2;

state.num_scale = 3;
state.scale_step = 1.03;
state.min_scale_factor = 0.2;
state.max_scale_factor = 5;
state.scale_penalty = 0.98;
state.net_name = './VDCFNet-2018-6-01';

state = vl_argparse(state, param);

net_name = [state.net_name,'.mat'];
net = load(net_name);
net = vl_simplenn_tidy(net.net);
state.net = net;
state.net.layers{1,1}.pad=0;
state.net.layers{1,4}.pad=0;
%readers can check our vconv filters by
% ' >>size(state.net.layers{1,1}.weights{1,1}); '

state.scale_factor = state.scale_step.^((1:state.num_scale)-ceil(state.num_scale/2));
state.scale_penalties = ones(1,state.num_scale);
state.scale_penalties((1:state.num_scale)~=ceil(state.num_scale/2)) = state.scale_penalty;

state.net_input_size = state.net.meta.normalization.imageSize(1:2);
state.net_average_image = reshape(single([106,102,94]),[1,1,3]);

output_sigma = sqrt(prod([state.output,state.output]./(1+state.padding)))*state.output_sigma_factor;
state.yf = single(fft2(gaussian_shaped_labels(output_sigma, [state.output,state.output])));
state.cos_window = single(hann(size(state.yf,1)) * hann(size(state.yf,2))');
cos_window1 = single(hann(201) * hann(201)');
state.cos_window1 = cos_window1(41:161,41:161);% Losser cosine window

yi = linspace(-1, 1, state.net_input_size(1));
xi = linspace(-1, 1, state.net_input_size(2));
[xx,yy] = meshgrid(xi,yi);
state.yyxx = single([yy(:), xx(:)]') ;

if state.gpu %gpuSupport
    state.yyxx = gpuArray(state.yyxx);
    state.net = vl_simplenn_move(state.net, 'gpu');
    I = gpuArray(I);
    state.yf = gpuArray(state.yf);
    state.cos_window = gpuArray(state.cos_window);
end

state.pos = region([2,1])+region([4,3])/2;
state.target_sz = region([4,3])';
state.min_sz = max(4,state.min_scale_factor.*state.target_sz);
state.max_sz = state.max_scale_factor.*state.target_sz;

window_sz = state.target_sz*(1+state.padding);
patch = imcrop_multiscale(I, state.pos, window_sz, state.net_input_size, state.yyxx);
target = bsxfun(@minus, patch, state.net_average_image);
res = vl_simplenn(state.net, target);

xf = fft2(bsxfun(@times, res(end).x, state.cos_window));
state.numel_xf = numel(xf);
kf = sum(xf.*conj(xf),3)/state.numel_xf;
state.model_alphaf = state.yf ./ (kf + state.lambda);
state.model_xf = xf;

location = region;
if state.visual
    imshow(uint8(I));
    rectangle('Position',location,'EdgeColor','g');
    drawnow;
end

end

function [state, location] = VDCFNet_update(state, I, frame,result,varargin)
if state.gpu, I = gpuArray(I);end
state.padding1 = 1.5;
%% 1st search
window_sz1 = state.target_sz*(1+1.5);
patch_crop1 = imcrop_multiscale(I, state.pos, window_sz1, state.net_input_size, state.yyxx);
search1 = bsxfun(@minus, patch_crop1, state.net_average_image);
res1 = vl_simplenn(state.net, search1);

zf1 = fft2(bsxfun(@times, res1(end).x, state.cos_window));
kzf1 = sum(bsxfun(@times, zf1, conj(state.model_xf)),3)/state.numel_xf;

response1 = squeeze(real(ifft2(bsxfun(@times, state.model_alphaf, kzf1))));
Peak = max(response1(:));
[v_delta, h_delta] = find(response1 == Peak, 1);

if v_delta > size(response1,1) / 2  %wrap around to negative half-space of vertical axis
    v_delta = v_delta - size(response1,1);
end
if h_delta > size(response1,2) / 2  %same for horizontal axis
    h_delta = h_delta - size(response1,2);
end
state.pos1 = state.pos + [v_delta , h_delta ].*window_sz1'./state.net_input_size;
if frame==2
   state.Peak = Peak;
end
%% 2nd search
window_sz = bsxfun(@times, state.target_sz, state.scale_factor)*(1+state.padding1);
if Peak > state.Peak/7
    patch_crop = imcrop_multiscale(I, state.pos1, window_sz, state.net_input_size, state.yyxx);
else
    region_past = result(frame-2,:);
    state.pos1 = region_past([2,1])+region_past([4,3])/2;
    patch_crop = imcrop_multiscale(I, state.pos1, window_sz, state.net_input_size, state.yyxx);
end
patch_crop = imcrop_multiscale(I, state.pos1, window_sz, state.net_input_size, state.yyxx);
search = bsxfun(@minus, patch_crop, state.net_average_image);
res = vl_simplenn(state.net, search);
zf = fft2(bsxfun(@times, res(end).x, state.cos_window1));
kzf = sum(bsxfun(@times, zf, conj(state.model_xf)),3)/state.numel_xf;
response = squeeze(real(ifft2(bsxfun(@times, state.model_alphaf, kzf))));
currentScaleID = ceil(state.num_scale/2);
scale_delta = currentScaleID;
bestPeak = -Inf;
for s=1:state.num_scale
    responseUP(:,:,s) = imresize(response(:,:,s), state.upsize, 'bicubic');
    thisResponse = responseUP(:,:,s);
    if s~=currentScaleID, thisResponse = thisResponse * state.scale_penalties(s); end
    thisPeak = max(thisResponse(:));
    if thisPeak > bestPeak, bestPeak = thisPeak; scale_delta = s; end
end
responseMap = responseUP(:,:,scale_delta);
[vert_delta, horiz_delta] = find(responseMap == max(responseMap(:)), 1);

if vert_delta > size(responseUP,1) / 2  %wrap around to negative half-space of vertical axis
    vert_delta = vert_delta - size(responseUP,1);
end
if horiz_delta > size(responseUP,2) / 2  %same for horizontal axis
    horiz_delta = horiz_delta - size(responseUP,2);
end
window_sz = window_sz(:,scale_delta);
state.pos = state.pos1 + [vert_delta , horiz_delta ]/state.upsize .*window_sz'./state.net_input_size;
state.target_sz = min(max(window_sz./(1+state.padding1), state.min_sz), state.max_sz);

%% update 
if bestPeak>state.Peak/3
    patch = imcrop_multiscale(I, state.pos, window_sz, state.net_input_size, state.yyxx);
    target = bsxfun(@minus, patch, state.net_average_image);

    res = vl_simplenn(state.net, target);
    xf = fft2(bsxfun(@times, res(end).x, state.cos_window));
    kf = sum(xf .* conj(xf), 3) / numel(xf);
    alphaf = state.yf ./ (kf + state.lambda);   %equation for fast training

    state.model_alphaf = (1 - state.interp_factor) * state.model_alphaf + state.interp_factor * alphaf;
    state.model_xf = (1 - state.interp_factor) * state.model_xf + state.interp_factor * xf;   
end
box = [state.pos([2,1]) - state.target_sz([2,1])'/2, state.target_sz([2,1])'];
location = double(gather(box));

id = sprintf('%03d',frame);

if state.visual
    imshow(uint8(I),'border','tight','initialmagnification','fit');
    rectangle('Position',location,'EdgeColor','g');
%     text(round(location(1)), round(location(2))-20, sprintf('%05d',Peak), 'Color','g', 'FontWeight','bold', 'FontSize',12);
    text(10, 15, ['#' id], 'Color','r', 'FontWeight','bold', 'FontSize',24);
%     text(10, 45, sprintf('%05d',state.Peak), 'Color','r', 'FontWeight','bold', 'FontSize',24);
    drawnow;
end

end

function labels = gaussian_shaped_labels(sigma, sz)
[rs, cs] = ndgrid((1:sz(1)) - floor(sz(1)/2), (1:sz(2)) - floor(sz(2)/2));
labels = exp(-0.5 / sigma^2 * (rs.^2 + cs.^2));
labels = circshift(labels, -floor(sz(1:2) / 2) + 1);
assert(labels(1,1) == 1)
end

function img_crop = imcrop_multiscale(img, pos, sz, output_sz, yyxx)
[im_h,im_w,im_c,~] = size(img);

if im_c == 1
    img = repmat(img,[1,1,3,1]);
end

pos = gather(pos);
sz = gather(sz);
im_h = gather(im_h);
im_w = gather(im_w);

cy_t = (pos(1)*2/(im_h-1))-1;
cx_t = (pos(2)*2/(im_w-1))-1;

h_s = sz(1,:)/(im_h-1);
w_s = sz(2,:)/(im_w-1);

s = reshape([h_s;w_s], 2,1,[]); % x,y scaling
t = [cy_t;cx_t]; % translation

g = bsxfun(@times, yyxx, s); % scale
g = bsxfun(@plus, g, t); % translate
g = reshape(g, 2, output_sz(1), output_sz(2), []);

img_crop = vl_nnbilinearsampler(img, g);
end