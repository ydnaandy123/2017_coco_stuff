# 2017_coco_stuff
http://cocodataset.org/#stuff-challenge2017

# Models
- Encode-decoder (ae)
- Stacked Hourglass (stack)
- Squeeze-and-Excitation (se)
- Label Hierarchy (multi)

### Encode-decoder
<img src="https://github.com/favicon.ico" width="48">
<img src=".src/ae.png" alt="Encode-decoder" width="100px" height="20px">
- The most basic model
- Seems it reach it's upper bound
- The learning matters a lot

### Stacked Hourglass
![Stacked Hourglass](./src/stack.png)
- The network imitate Stacked Hourglass Networks for Human Pose Estimation(https://arxiv.org/abs/1603.06937)
- two consecutive stack. one

### Squeeze-and-Excitation
![Squeeze-and-Excitation](src/se.png)
-  The network uses the Squeeze-and-Excitation(https://github.com/hujie-frank/SENet) technique to improve the performance.


### Label Hierarchy
![Label Hierarchy](src/multi.png)
- According to [Label Hierarchy](https://github.com/nightrome/cocostuff#label-hierarchy)
- From left to right [image, gt1, gt2, g3, predict1, predict2, predict3]
    1. Hierarchy: first predict indoor or outdoor, then plant, and then tree.
    2. Multi-task: network predict different level labels at the same time.


# Result
![Encode-decoder](src/ae_loss.png)


## Dilated