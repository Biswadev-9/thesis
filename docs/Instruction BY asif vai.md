# **Step 1: Define the exact classification task** 

The main task must be defined before any coding begins. The recommended primary task is four-class MRI classification: glioma, meningioma, pituitary tumor, and non-tumor. The labels must be kept consistent across training, validation, testing, and external validation. 

- Class 0: glioma. 

- Class 1: meningioma. 

- Class 2: pituitary tumor. 

- Class 3: non-tumor. 

# **Step 2: Select and document datasets** 

Use at least one primary dataset and one external validation dataset. The primary dataset should support the four target classes. External validation should test whether the model generalizes beyond the training source. 

|**Dataset role**|**Required content**|**Use in study**|
|---|---|---|
|Primary dataset|Four-class MRI classification dataset with glioma,<br>meningioma, pituitary tumor, and non-tumor<br>classes.|Main training, validation,<br>and internal testing.|
|External dataset A|Figshare brain tumor MRI dataset with glioma,<br>meningioma, and pituitary tumor classes.|External validation for<br>common tumor classes if no<br>non-tumor class is available.|
|||Robustness or auxiliary|
|External dataset B or<br>optional|BraTS or another clinically relevant MRI dataset,<br>if class mapping is appropriate.|validation. Use only when<br>labels match the research<br>task.|



# **Step 3: Prevent data leakage before training** 

Data leakage can invalidate the entire study. The dataset must be split before augmentation and before any synthetic data generation. 70% training, 15% validation, and 15% internal testing. 

# **Step 4: Perform initial data audit and quality control** 

Before model development, inspect image quality and class balance. This step is essential because MRI datasets often contain noisy, low-contrast, duplicated, or mislabeled images. 

- Check image dimensions, grayscale/RGB status, bit depth, and intensity ranges. 

- Plot class distribution and report imbalance ratio. 

- Inspect representative images from each class. 

- Check for corrupted files, repeated images, unreadable images, and inconsistent labels. 

- Record all exclusions and corrections in a dataset audit table. 

# **Step 5: Standardize image preprocessing** 

All images must pass through the same basic preprocessing pipeline before advanced preprocessing. This ensures fair comparison across baseline and proposed models. 

- Resize all images to a fixed input size, such as 256 × 256 pixels. 

- Convert grayscale images to three channels only when required by pretrained CNN or Transformer backbones. 

- Normalize intensity using min-max scaling or z-score normalization. Use the same method for training and testing. 

- Apply background cropping only if it does not remove tumor regions. 

- Avoid aggressive skull stripping unless segmentation quality is validated. 

# **Step 6: Implement diffusion-based preprocessing** 

The diffusion-based preprocessing module should be edge-preserving. The recommended starting point is anisotropic diffusion filtering because it reduces noise while preserving boundaries. It is also easier to justify and control than generative diffusion for a first implementation. 

Use the following diffusion update as the main preprocessing formulation: 

𝐼(𝑡+ 1) = 𝐼(𝑡) + 𝑙𝑎𝑚𝑏𝑑𝑎 ∗ 𝑑𝑖𝑣(𝑐(||𝑔𝑟𝑎𝑑 𝐼(𝑡)||) 𝑔𝑟𝑎𝑑 𝐼(𝑡)) 

where 𝐼(𝑡) is the image at diffusion step t, lambda is the update step, and 𝑐(. ) is the diffusion coefficient. 2 𝑠 1 Two common choices are 𝑐(𝑠) = exp (− ~~(~~ 𝑘𝑎𝑝𝑝𝑎) )𝑎𝑛𝑑 𝑐(𝑠) = 𝑠 2. 1 + (𝑘𝑎𝑝𝑝𝑎 ~~)~~ 

- Tune the number of diffusion iterations using the validation set only. Start with 5, 10, 15, and 20 iterations. 

- Tune kappa using validation experiments. Small kappa preserves stronger edges but may leave more noise. 

- Keep lambda stable. For a standard explicit scheme, lambda should be small, commonly not greater than 0.25. 

- Compare anisotropic diffusion with no preprocessing, Wiener filtering, adaptive gamma correction, CLAHE, and logarithmic transformation. 

- Do not choose preprocessing based only on visual appearance. Select it using validation performance and boundary/texture preservation checks. 

# **Step 7: Add training-only augmentation** 

Augmentation should improve generalization without changing the diagnostic class. Use conservative transformations because MRI anatomy has clinical meaning. 

- Allowed basic augmentations: small rotation, small translation, horizontal flip only if anatomically acceptable for the dataset, mild zoom, and slight brightness/contrast change. 

- Avoid strong warping, excessive rotation, or transformations that distort tumor morphology. 

- Apply augmentation only to the training set. 

- Report the exact augmentation ranges in the method section. 

If synthetic minority-class generation is used, validate synthetic image quality. Use visual inspection, feature-space comparison, and a metric such as FID or KID when appropriate. Synthetic samples should not be added to validation or test sets. 

# **Step 8: Handle class imbalance** 

Class imbalance must be handled at the training stage. Use more than one strategy only when ablation confirms benefit. 

- Start with stratified splitting and class-weighted cross-entropy. 

- Evaluate focal loss for difficult minority classes. A common form is 

   - 𝐹𝐿 = −𝑎𝑡  (1 − 𝑝𝑡)<sup>𝑔𝑎𝑚𝑚𝑎</sup> 𝑙𝑜𝑔(𝑝𝑡). 

- Use a weighted sampler only for the training loader. 

- Compare class weighting, focal loss, balanced sampler, and augmentation in ablation experiments. 

- Use macro-F1, balanced accuracy, and class-wise recall to judge imbalance handling. Accuracy alone is not sufficient. 

# **Step 9: Build baseline models first** 

Strong baseline models are needed before developing the proposed architecture. Without strong baselines, the study cannot support claims about improvement or quantum advantage. 

|**Baseline**|**Model**|
|---|---|
|Baseline 1|Simple CNN|
|Baseline 2|ResNet50 or DenseNet121|
|Baseline 3|EfficientNetB0/EfficientNetV2|
|Baseline 4|ViT or Swin Transformer|
|Baseline 5|Fixed QCNN|
|Baseline 6|Classical multiscale CNN without quantum branch|



# **Step 10: Implement the classical feature extraction branch** 

The classical branch should extract stable anatomical and texture patterns from MRI images. Use a strong pretrained backbone and fine-tune it carefully. 

- Recommended CNN backbone: EfficientNetV2, EfficientNetB0, DenseNet121, or ConvNeXt-Tiny. 

- Recommended Transformer backbone: Swin Transformer or ViT-Small, depending on compute resources. 

- Extract features from the final convolutional block or global average pooling layer. 

- Use dropout and weight decay to reduce overfitting. 

- Save the intermediate feature embeddings for later feature separability analysis using t-SNE or UMAP. 

# **Step 11: Implement the adaptive multiscale kernel branch** 

This branch directly addresses the fixed-kernel limitation. It should learn which receptive field is more useful for each input image or feature map. 

Recommended design: use parallel convolutional paths with different kernel sizes, such as 3 × 3, 5 × 5, and 7 × 7, or use dilated convolutions with different dilation rates. Then use a scale attention or gating module to weight the paths. 

- Small kernels help capture fine boundary and local texture information. 

- Larger or dilated kernels help capture broader tumor context and heterogeneous regions. 

- The gate should output normalized weights for the multiscale branches. 

- Report the learned scale weights for example images to show whether the model adapts to tumor morphology. 

Ablation is essential: compare fixed 3 × 3 convolution, fixed 5 × 5 convolution, fixed dilated convolution, and the proposed adaptive multiscale module. 

# **Step 12: Implement the adaptive quantum-classical branch** 

The quantum branch should be developed carefully. It should not be presented as automatically superior. Its role is to test whether quantum or quantum-inspired feature transformations improve separability, efficiency, or robustness. 

- Convert classical feature maps into compact feature vectors before quantum encoding. 

- Use dimensionality reduction before quantum encoding. Options include global average pooling, 1 × 1 convolution, PCA on training features, or a learnable projection layer. 

- Use angle encoding for the first implementation because it is simple and stable. 

- Start with 4 qubits for comparison with prior QCNN work. Then test 6 or 8 qubits if simulation resources allow. 

- Use a parameterized quantum circuit with rotation gates and entangling gates. 

- Test at least two circuit depths and at least two entanglement patterns. 

- Use expectation values as quantum features and concatenate them with classical features. 

Recommended quantum experiments: fixed QCNN circuit, adaptive circuit depth, adaptive entanglement topology, and learnable gate-parameter re-uploading. The final model should use only the configuration selected by validation performance and computational feasibility. 

# **Step 13: Design the feature fusion module** 

The fusion module combines the classical, adaptive multiscale, and quantum feature branches. Simple concatenation should be used as the first fusion baseline. Then add attention-based or gated fusion only if it improves validation performance. 

- Fusion baseline 1: concatenate all feature vectors and train a dense classifier. 

- Fusion baseline 2: apply channel attention or squeeze-and-excitation after concatenation. 

- Fusion baseline 3: use gated fusion where the model learns the importance of each branch. 

- Optional: use transformer-style attention over branch tokens if sufficient data are available. 

Report branch contribution through ablation and learned fusion weights. This helps explain whether the quantum branch, diffusion preprocessing, or adaptive kernels contribute meaningfully. 

# **Step 14: Build the final classifier** 

The final classifier should be simple enough to avoid overfitting. Use fully connected layers with normalization, dropout, and a final SoftMax layer for four classes. 

- Input: fused feature vector. 

- Hidden layers: one or two dense layers with ReLU or GELU activation. 

- Regularization: dropout, weight decay, and early stopping. 

- Output: four-class softmax probability vector. 

- Loss: class-weighted cross-entropy or focal loss based on validation results. 

# **Step 15: Training protocol** 

The training protocol must be fixed before final testing. Do not change hyperparameters after seeing test or external validation results. 

- Optimizer: AdamW or Adam. 

- Learning rate: tune using validation data. Start with 1e-4 and 3e-4 for fine-tuning. 

- Batch size: select based on GPU memory, commonly 16 or 32. 

- Epochs: train with early stopping. Use patience of 10 to 15 epochs. 

- Scheduler: cosine annealing or ReduceLROnPlateau. 

- Save the best model using validation macro-F1 or balanced accuracy, not only validation accuracy. 

- Run at least three seeds for the final model and major baselines. 

- Log all hyperparameters, random seeds, split files, and software versions. 

# **Step 16: Internal testing** 

After model selection using the validation set, evaluate the final model once on the internal test set. The test set should remain unseen during training and hyperparameter tuning. 

- Report accuracy, balanced accuracy, macro-precision, macro-recall, macro-F1, weighted-F1, sensitivity, specificity, MCC, and one-vs-rest AUC. 

- Report class-wise precision, recall, F1-score, and support. 

- Include a confusion matrix and analyze which tumor classes are confused. 

- Report calibration using expected calibration error or Brier score if possible. 

# **Step 17: Cross-dataset validation** 

Cross-dataset validation is necessary because benchmark accuracy alone does not prove clinical robustness. Train on the primary dataset and test on an external dataset after matching label definitions. 

- If the external dataset has the same four classes, evaluate the full four-class task. 

- If the external dataset has only glioma, meningioma, and pituitary classes, evaluate a three-class external task using the same trained feature extractor if scientifically valid, or fine-tune only under a clearly stated transfer protocol. 

- Do not mix external data into training unless it is part of a separate multi-source experiment. 

- Report the performance drop from internal testing to external testing. 

- Discuss possible causes of domain shift, such as scanner differences, contrast, resolution, and patient population. 

# **Step 18: Robustness testing** 

Robustness testing checks whether the model is stable under realistic image degradation. This is important for MRI data because noise and contrast variation are common. 

- Add controlled Gaussian noise to test images and measure performance degradation. 

- Test contrast shifts, mild blur, resolution changes, and intensity normalization changes. 

- Compare robustness of the proposed model against strong CNN and Transformer baselines. 

- Report whether diffusion preprocessing improves robustness under noisy inputs. 

# **Step 19: Explainability analysis** 

Explainability must be included as a core part of the proposed model, not as a decorative figure. The model should show which regions and patterns influenced each prediction. 

- Use Grad-CAM or Score-CAM for CNN-based feature maps. 

- Use attention rollout or attention maps for Transformer-based components. 

- Use SHAP or LIME for the final fused feature vector or classifier output. 

- Use uncertainty scores, such as Monte Carlo dropout or ensemble variance, to flag uncertain predictions. 

- Evaluate explanations using sanity checks, such as deletion/insertion tests or comparison with tumor masks when available. 

- Include correct and incorrect prediction examples for each class. 

# **Step 20: Quantum advantage and efficiency analysis** 

Quantum advantage must be treated as an empirical question. The study should test whether the quantum or quantum-inspired branch gives measurable benefit under identical experimental conditions. 

- Compare the proposed model with the same architecture after removing the quantum branch. 

- Compare fixed QCNN with adaptive quantum branch. 

- Report trainable parameters, inference time, training time, memory usage, and performance metrics. 

- Analyze feature separability using UMAP or t-SNE embeddings. 

- Use statistical testing across folds or seeds. Suitable tests include paired bootstrap, Wilcoxon signed-rank test, McNemar test, or confidence intervals. 

If the quantum branch does not outperform strong baselines, report the result honestly. In that case, the contribution may still be useful if it improves parameter efficiency, robustness, or interpretability. 

# **Step 21: Ablation study design** 

|**Ablation ID**|**Configuration**|
|---|---|
|A0|Raw image + baseline CNN|
|A1|Conventional preprocessing + CNN|
|A2|Diffusion preprocessing + CNN|
|A3|Diffusion + adaptive multiscale branch|
|A4|Diffusion + fixed QCNN branch|
|A5|Diffusion + adaptive quantum branch|
|A6|Diffusion + adaptive multiscale + adaptive quantum + fusion|
|A7|Core model + imbalance-aware loss|
|A8|Core model + explainability and uncertainty|



The ablation study should report the same metrics for every configuration. Use macro-F1 and class-wise recall as primary selection metrics because the task is multiclass and may be imbalanced. 

# **Step 22: Research question-to-experiment mapping** 

|**Research question**|**Experiment**|**Evidence**|
|---|---|---|
|RQ1: Improve multiclass classification|Compare<br>proposed<br>model<br>with<br>all<br>baselines on internal and external test sets.|Macro-F1,<br>balanced<br>accuracy,<br>class-wise recall.|
|RQ2: Diffusion preprocessing|Compare no preprocessing, conventional<br>preprocessing,<br>and<br>diffusion<br>preprocessing.|Image<br>quality,<br>macro-F1,<br>robustness under noise.|
|RQ3: Boundary and texture preservation|Assess explanations and performance on<br>difficult cases; use masks if available.|Grad-CAM<br>localization,<br>deletion/insertion tests.|
|RQ4: Adaptive kernels/circuits|Compare<br>fixed<br>kernels/circuits<br>with<br>adaptive modules.|Performance gain and learned<br>scale/circuit weights.|
|RQ5: Tumor variation handling|Analyze<br>performance<br>by<br>tumor<br>size/appearance if metadata or masks are<br>available.|Class-wise metrics and subgroup<br>performance.|
|RQ6: Imbalance strategy|Compare class weights, focal loss, sampler,<br>and augmentation.|Minority-class recall, macro-F1,<br>calibration.|
|RQ7: External dataset performance|Train on primary dataset and test on<br>external dataset.|Performance drop, cross-dataset<br>macro-F1.|
|RQ8: Quantum benefit|Remove or replace quantum branch and<br>compare.|Accuracy, macro-F1, parameter<br>count, time.|
|RQ9: Explainability|Generate<br>and<br>evaluate<br>class-specific<br>explanations.|Clinically relevant heatmaps and<br>sanity checks.|
|RQ10: Component contribution|Full ablation study.|Delta in metrics for each module.|



# **Step 23: Statistical reporting** 

The final paper should include uncertainty estimates. Single-run results are weak for Q1-level reporting. 

- Report mean and standard deviation across folds or random seeds. 

- Use 95% confidence intervals for main metrics. 

- Use McNemar test for paired comparison of classification errors between two models on the same test set. 

- Use Wilcoxon signed-rank test or paired bootstrap for repeated fold/seed comparisons. 

- Report p-values only when the experimental design supports paired testing. 

- Do not overstate minor improvements unless they are statistically and clinically meaningful. 

