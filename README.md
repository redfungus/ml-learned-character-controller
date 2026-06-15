
# Overview

This repository contains an ML learned character controller trained on data collected from a simulated character controller. For simplicity, the controller only predicts the rotation.

https://github.com/redfungus/ml-learned-character-controller/raw/master/images/unity_best.mp4

## Key Technical Decisions

**Data Generation** is simulated using the data generation script in `OrientationPrediction`

**Data Splitting** is done by taking 80% of the data for training and 10% each for validation and testing. The last 20% of the data is used for validation and testing. In the case of this task, this splitting schemes works as the distribution of the datasets becomes very similar given the uniform inputs. However, for data were there might be meaningful differences between the start and end of the trajectory, this might not work.

**PyTorch Lightning** was chosen for its modular architecture, separating model definition, training logic, and data handling. This enables easy experimentation with different model architectures (MLP, split encoder) and loss functions through a registry pattern. It also enables easy logging for tensorflow through callbacks.

**Quaternion Representation** required careful handling due to the double-cover problem: quaternions q and -q represent identical rotations. I address this with:
- `AntipodalMSELoss` loss or `GeodesicLoss` (`src/losses/quaternion.py`)
- Preprocessing step ensuring quaternion continuity across sequences
- Output normalization to maintain unit quaternions alongside norm regularization before output normalization.

**Multi-Step Rollout Training** with scheduled sampling improves autoregressive stability. Teacher forcing decays exponentially from 100% to 20%, allowing the model to gradually learn from its own predictions during training. (`src/lightning_module.py` in `_compute_loss` function). An alternative approach would be adding noise to the data which would also be much more perfomant.

**Angular Velocity Prediction** alongside delta quaternions provides richer supervision signal and enables velocity-aware motion generation (`src/lightning_module.py` in `_compute_loss` function). We modify the dataset loading so that the angular velocity of the next row will be used for the current step. (`src/data/dataset.py` in `_prepare_data`).

**Features**: The model uses 6 input features: `[pad_x, pad_y, vel_x, vel_y, vel_z, dt]` - control input, angular velocity, and timestep. Notably, the current orientation is *not* an input feature. Since the model predicts delta quaternions (relative orientation changes) rather than absolute orientations, it is frame-agnostic. So we can use the outputs of the model for both local and global rotations. 

## Challenges Encountered

**Teacher Forcing Decay**: Initial linear decay caused training instability. Switching to exponential decay (rate 0.005/epoch) provided smoother curriculum learning.

**Quaternion Discontinuities and Double Cover Issue**: When training neural networks with quaternions we have to be careful about potential discontinuities caused by sudden sign flips around 180 and -180 degrees and also when we do multiple revolutions. To mitigate this we preprocess the data to ensure all quaternions are always on the same hemisphere. Also, when the model outputs a delta quaternion and we apply it to the previous orientaion we ensure the resulting quaternion is in the same hemisphere. This is especially important when doing autoregressive rollouts. In models where we don't predict the angular velocity, we have to also ensure the calculated angular velocity based on the predicted delta quaternion and `dt` is the short way around. Alternatively, we could use 6D representations that do not have these continuity issues. I also use Geodesic loss since it represents the actual angle difference by measuring the arc on a 3D sphere.

**High Angular Velocities**

The data lacks high angular velocity samples since the input is uniform and this makes it very unlikely to contain trajectories where one input is continously used. This created some instabilities in the first models explained and shown below.

## Future Improvements

- **6D Rotation Representation**: Adopt Zhou et al.'s continuous representation for improved gradient flow
- **Production**: Robust dependency management, input normalization statistics, tests and dockerization
- **Split Encoder Architecture** separates state inputs (orientation, angular velocity) from control inputs (pad x/y, dt) through independent encoder pathways before merging. A basic version of this is in the `models` folder but wasn't tested.
  - Allows each encoder to learn specialized representations for its input domain
  - Provides natural inductive bias: physical state vs. user intent are different signals
  - Enables asymmetric capacity allocation (e.g., deeper state encoder for complex quaternion dynamics)
  - Allows the decoder to scale with `dt`
  - Facilitates transfer learning: control encoder could generalize across different state representations

## Results

I evaluate the results by examining both single step and multi step evaluation metrics. Since we are generating trajectories it is crucial to examine the metrics over multi step rollouts.

Both multi-step learning with scheduled teacher forcing and adding angular velocity prediction to the model boosted the performance of the model. The best model uses both of them.

**Angular Error**

| Model                                        | 1-step      | 10-step     | 50-step      | 100-step      |
|:---------------------------------------------|:------------|:------------|:-------------|:--------------|
| Multi step with angular velocity prediction  | 0.01 ± 0.02 | 0.12 ± 0.07 | 0.71 ± 0.30  | 1.48 ± 0.58   |
| Multi step                                   | 0.00 ± 0.01 | 0.23 ± 0.13 | 1.31 ± 0.66  | 2.31 ± 1.27   |
| Single step with angular velocity prediction | 0.01 ± 0.02 | 0.32 ± 0.13 | 4.06 ± 1.00  | 9.60 ± 1.89   |
| Single step                                  | 0.03 ± 0.04 | 1.30 ± 0.70 | 13.85 ± 5.39 | 30.77 ± 10.31 |

**Anti podal MSE**

| Model                                        | 1-step              | 10-step             | 50-step             | 100-step            |
|:---------------------------------------------|:--------------------|:--------------------|:--------------------|:--------------------|
| Multi step with angular velocity prediction  | 0.000000 ± 0.000000 | 0.000002 ± 0.000001 | 0.000045 ± 0.000035 | 0.000191 ± 0.000141 |
| Multi step                                   | 0.000000 ± 0.000000 | 0.000005 ± 0.000006 | 0.000164 ± 0.000148 | 0.000529 ± 0.000598 |
| Single step with angular velocity prediction | 0.000000 ± 0.000000 | 0.000009 ± 0.000007 | 0.001330 ± 0.000675 | 0.007280 ± 0.003071 |
| Single step                                  | 0.000000 ± 0.000000 | 0.000166 ± 0.000172 | 0.016777 ± 0.013445 | 0.079392 ± 0.057349 |

Below I show a sample trajectory the worst and best model:

### Single-step training without angular velocity prediction

<img src="https://github.com/redfungus/ml-learned-character-controller/raw/master/images/drift_first_model.jpg" alt="drawing" width="640"/>

<video width="640" controls>
  <source src="https://github.com/redfungus/ml-learned-character-controller/raw/master/images/trajectory_video_first.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Multi-step training with angular velocity prediction

<img src="https://github.com/redfungus/ml-learned-character-controller/raw/master/images/drift_best_model.jpg" alt="drawing" width="640"/>

<video width="640" controls>
  <source src="https://github.com/redfungus/ml-learned-character-controller/raw/master/images/trajectory_video_best.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Unity Prototype

The project also includes a basic Unity prototype to see the models and physics side-by-side. Below are recordings for the worst and best model. The model to be used can be configured using the ScriptableObject in `Assets/OrientationModel`.
The first few seconds of each video show random inputs from the same distribution that generated the data. Here I show the controller being used in global space. This can be configured in the same ScriptableObject to apply locally (pressing down makes the nose of the plane go down no matter the orientation).  

### Single-step training without angular velocity prediction

<video width="640" controls>
  <source src="[images/unity_first.mp4](https://github.com/redfungus/ml-learned-character-controller/raw/master/images/unity_first.mp4)" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Multi-step training with angular velocity prediction

<video width="640" controls>
  <source src="https://github.com/redfungus/ml-learned-character-controller/raw/master/images/unity_best.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## References

- Zhou et al. (2019) - "[On the Continuity of Rotation Representations in Neural Networks](https://arxiv.org/abs/1812.07035)"
- Pavllo et al. (2018) - "[QuaterNet: A Quaternion-based Recurrent Model for Human Motion](https://arxiv.org/abs/1805.06485)"
- Daniel Holden (theorangeduck.com) - [Unrolling Rotations](https://theorangeduck.com/page/unrolling-rotations)