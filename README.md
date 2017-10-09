# 2017_coco_stuff
http://cocodataset.org/#stuff-challenge2017

# Models
- Encode-decoder (ae)
- Stacked Hourglass (stack)
- Squeeze-and-Excitation (se)
- Label Hierarchy (multi)

### Encode-decoder
<img src="src/ae.png" width="400">
- The basic model
- tedt-dev2017 mIoU: 0.1234

### Label Hierarchy
<img src="src/multi.png">
- According to [Label Hierarchy](https://github.com/nightrome/cocostuff#label-hierarchy)
- From left to right [image, gt1, gt2, g3, predict1, predict2, predict3]
- Aluxary loss
    1. Hierarchy: first predict indoor or outdoor, then plant, and then tree.
    2. Multi-task: network predict different level labels at the same time.
- tedt-dev2017 mIOU: 0.1049 
    
### Squeeze-and-Excitation
<img src="src/se.png"  width="400">
- The network uses the Squeeze-and-Excitation(https://github.com/hujie-frank/SENet) technique to improve the performance.
- tedt-dev2017 mIOU:  0.1230 

### Stacked Hourglass
<img src="src/stack.png"  width="400">
- The network imitate Stacked Hourglass Networks for Human Pose Estimation(https://arxiv.org/abs/1603.06937)
- two consecutive stack. one
- test-dev2017 mIOU: 0.1245 



# Result
![Encode-decoder](src/ae_loss.png)


- Seems it reach it's upper bound
- The learning matters a lot


## Dilated