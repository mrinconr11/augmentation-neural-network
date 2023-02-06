# augmentation-neural-network
This repository contains an example of the augmentation neural networks (NN) used in the published paper: "Augmentation of field fluorescence measures for improved in situ contaminant detection"

A few key flourescence intensities are used as input to an augmentation neural network that predicts the full excitation emission matrix (EEM). The generated EEM is then used as input to predict the concentrations of naphthenic acids, phenol, fluoranthene and pyrene.

Few fluorescence intensities are not able to estimate contaminants concentrations. The overlapping signals affect the linear relationship between fluorescence intensity and the contaminant concentration. The augmentation NN predicts the full EEM to make concentration predictions. Results are as accurate as using the original EEM.

You can find the full paper in: Rincón Remolina, M. C., & Peleato, N. M. (2023). Augmentation of field fluorescence measures for improved in situ contaminant detection. Environmental Monitoring and Assessment, 195(1), 34. 
https://link.springer.com/article/10.1007/s10661-022-10652-1
