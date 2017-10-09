# 2017_coco_stuff
http://cocodataset.org/#stuff-challenge2017

# Results

## Label Hierarchy
![Label Hierarchy](src/multi.png)
- according to [Label Hierarchy](https://github.com/nightrome/cocostuff#label-hierarchy)
- from left to right [image, gt1, gt2, g3, predict1, predict2, predict3]
- multi-task: network predict different level labels at the same time.
- hierarchy: first predict indoor or outdoor, then plant, and then tree.
