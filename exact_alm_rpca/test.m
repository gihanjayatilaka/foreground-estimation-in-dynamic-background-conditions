v = VideoReader('test3.avi');
i = 1;
vid = [];
st_frame = 1000;
n_frame = 200;
%while hasFrame(v)
for i = 1:st_frame
    video = readFrame(v);
end
for i = st_frame:st_frame+n_frame
    video = readFrame(v);
    video_g = rgb2gray(video);
    [m, n] = size(video_g);
    vid = [vid ; double(reshape(video_g, [1,n*m]))]; 
end

[n_frame, k] = size(vid);
[L, S] = exact_alm_rpca(vid');

amin = 0;
amax = 255;

out = zeros(m, n, n_frame);

for i = 1:n_frame
    A = S(:,i);
    I = mat2gray(A, [amin amax]);
    out(:, :, i) = reshape(I, m, n);
end

implay(out);