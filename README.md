# sgp
Trying to enhance the undertext in the [SGP dataset](http://openn.library.upenn.edu/)

### Used Modules
python 3.7.4 <br />
- network building: <br />
torch 1.4.0 <br />
- data structures: <br />
pandas 0.25.1 <br />
numpy 1.17.2 <br />
- evaluation metrics: <br />
sklearn 0.23.2 <br />
- images operation & curves drawing: <br />
skimage 0.16.2 <br />
matlpotlib 3.1.1 <br />
torchvision 0.5.0 <br />

### To run & test: 
- `networks/models.py`: classes of all networks. <br />
- `networks/xxx_classify.py`: training of a network (xxx indicates the type of network) <br />
- `networks/xxx_classify_test_roi.py`: testing of a network, outputs enhancement reconstruction of a test image (NOTE: please run training before testing) <br />
- training data can be put under `networks/data/sgp/xxx.csv` (for pixel data) and `networks/data/sgp/{folio_id}/cropped_roi/` (for cropped image patches)
- intermediate folders created during training: `networks/training_log/`, `networks/model/`, `networks/reconstructed_xxx/`


### Results preview (on cropped Tiff * rescale-0.25)
Images of 024r_029v
<br />
Original version, LDA version, AE-enhanced version, 1DConvNet, 2DFConvNet, 3DConvNet-hyb, Conv-hybrid:
<br />
<img src='/figs/results_preview/024r_029v_orig_eval.png' width='100'>
<img src='/figs/results_preview/024r_029v_lda.png' width='100'>
<img src='/figs/results_preview/024r_029v_ae.png' width='100'>
<img src='/figs/results_preview/024r_029v_conv1d.png' width='100'>
<img src='/figs/results_preview/024r_029v_fconv2d_eval_model_102v_107r.png' width='100'>
<img src='/figs/results_preview/024r_029v_conv3d_hyb_eval_model_102v_107r.png' width='100'>
<img src='/figs/results_preview/024r_029v_conv_hybrid.png' width='100'>
<br />

Images of 102v_107r
<br />
Original version, LDA version, AE-enhanced version, 1DConvNet, 2DFConvNet, 3DConvNet-hyb, Conv-hybrid:
<br />
<img src='/figs/results_preview/102v_107r_orig_eval.png' width='100'>
<img src='/figs/results_preview/102v_107r_lda.png' width='100'>
<img src='/figs/results_preview/102v_107r_ae.png' width='100'>
<img src='/figs/results_preview/102v_107r_conv1d.png' width='100'>
<img src='/figs/results_preview/102v_107r_fconv2d_eval_model_102v_107r.png' width='100'>
<img src='/figs/results_preview/102v_107r_conv3d_hyb_eval_model_102v_107r.png' width='100'>
<img src='/figs/results_preview/102v_107r_conv_hybrid.png' width='100'>
<br />

Images of 214v_221r
<br />
Original version, LDA version, AE-enhanced version, 1DConvNet, 2DFConvNet, 3DConvNet-hyb, Conv-hybrid:
<br />
<img src='/figs/results_preview/214v_221r_orig_eval.png' width='100'>
<img src='/figs/results_preview/214v_221r_lda.png' width='100'>
<img src='/figs/results_preview/214v_221r_ae.png' width='100'>
<img src='/figs/results_preview/214v_221r_conv1d.png' width='100'>
<img src='/figs/results_preview/214v_221r_fconv2d_eval_model_102v_107r.png' width='100'>
<img src='/figs/results_preview/214v_221r_conv3d_hyb_eval_model_102v_107r.png' width='100'>
<img src='/figs/results_preview/214v_221r_conv_hybrid.png' width='100'>
<br />


### Network Architecture
Stacked Autoencoder [[1]](#xing2015stacked)
<br />
<img src='/figs/network_architecture/sae.png' width='800'>
<br />

1DConvNet [[2]](#hu2015deep)
<br />
<img src='/figs/network_architecture/1dconv_basic.png' width='680'>
<br />

2DFConvNet
<br />
<img src='/figs/network_architecture/f2dconv.png' width='800'>
<br />

2DConvNet-hyb
<br />
<img src='/figs/network_architecture/2dconv_hyb.png' width='680'>
<br />

Hybrid Convnet [[3]](#lee2016contextual)
<br />
<img src='/figs/network_architecture/baseline_leecnn.png' width='800'>
<br />


### Reference
<a id="xing2015stacked">[1]</a> 
C. Xing, L. Ma, and X. Yang. Stacked denoise autoencoder based feature extraction and classification for hyperspectral images. Journal of Sensors, 2016:e3632943, 2015.

<a id="hu2015deep">[2]</a> 
Hu, W., Huang, Y., Wei, L., Zhang, F. and Li, H., 2015. Deep convolutional neural networks for hyperspectral image classification. Journal of Sensors, 2015.

<a id="lee2016contextual">[3]</a> 
Lee, H. and Kwon, H., 2016, July. Contextual deep CNN based hyperspectral classification. In 2016 IEEE International Geoscience and Remote Sensing Symposium (IGARSS) (pp. 3322-3325). IEEE.
