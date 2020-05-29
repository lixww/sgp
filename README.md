# sgp
Trying to enhance the undertext in the [SGP dataset](http://openn.library.upenn.edu/)

### Environment
Python 3.7.4 <br />
Pytorch 1.4.0
<br />


### Results preview (on cropped Tiff * rescale-0.25)
Original image of 024r_029v, LDA version, NN-enhanced version, Conv-1d, Conv-2d, Conv-3d:
<br />
<img src='/figs/results_preview/scale25/024r_029v_orig_eval.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_lda.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_ae.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_conv1d.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_conv2d_eval.png' width='100'>
<img src='/figs/results_preview/scale25/024r_029v_conv3d_eval.png' width='100'>
<br />

Original image of 102v_107r, LDA version, NN-enhanced version, Conv-1d, Conv-2d, Conv-3d:
<br />
<img src='/figs/results_preview/scale25/102v_107r_orig_eval.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_lda.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_ae.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_conv1d.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_conv2d_eval.png' width='100'>
<img src='/figs/results_preview/scale25/102v_107r_conv3d_eval.png' width='100'>
<br />

Original image of 214v_221r, LDA version, NN-enhanced version, Conv-1d, Conv-2d, Conv-3d:
<br />
<img src='/figs/results_preview/scale25/214v_221r_orig_eval.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_lda.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_ae.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_conv1d.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_conv2d_eval.png' width='100'>
<img src='/figs/results_preview/scale25/214v_221r_conv3d_eval.png' width='100'>
<br />


### Network Reference
Denoise Autoencoder [[1]](#xing2015stacked)
<br />
<img src='/figs/networks/dae.png' width='360'>
<br />

Stacked Denoise Autoencoder [[1]](#xing2015stacked)
<br />
<img src='/figs/networks/sdae.png' width='900'>
<br />


### Reference
<a id="xing2015stacked">[1]</a> 
C. Xing, L. Ma, and X. Yang. Stacked denoise autoencoder based feature extraction and classification for hyperspectral images. Journal of Sensors, 2016:e3632943, 2015.
