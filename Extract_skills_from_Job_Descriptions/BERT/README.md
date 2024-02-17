# Extract_skills_from_job-Description
This model is a fine-tuned version of [bert-base-cased] ([https://openai.com](https://huggingface.co/google-bert/bert-base-cased)https://huggingface.co/google-bert/bert-base-cased)  on the LinkedIn (Data-related Job Posts) dataset

It achieves the following results on the evaluation set:
- Loss: 0.013761
- Precision: 0.930325
- Recall: 0.948412
- F1: 0.939281
- Accuracy: 0.995606

## Demo Deployment:
A demo app with the model results can be found here: [GalalEwida/Skills_Extractor_fromJobDescription_BERT_Base](https://huggingface.co/GalalEwida/Skills_Extractor_fromJobDescription_BERT_Base)

## Model description
The objective of this model is to provide a lightweight yet accurate Language Model (LLM) tailored for custom Named Entity Recognition (NER). Specifically, the model is designed to process technical requirements written in natural language, such as a job post from LinkedIn and Glassdoor, and extract various types of skills and technologies that potential candidates may be expected to possess. The primary focus of this model is to facilitate the identification and extraction of relevant information related to skills and technologies required in job descriptions, aiding in the hiring process and talent acquisition.

## Intended uses & limitations
The current release of this exercise serves as an experimental implementation and should be considered for research and exploration purposes only. It is not intended for deployment in production systems at this stage. As a first iteration, this model has provided valuable insights and learning experiences that were utilised to enhance its performance for the intended purposes in downstream experiments that due to copyright and data privacy restrictions can't be shared publicly.


The primary intention behind this model is to serve as a fundamental component of a larger recommendation system, which specifically focuses on analysing and extracting technical requirements from natural language texts, such as job posts from platforms like LinkedIn.

### Limitations:
As with any experimental release, there are certain limitations that users should be aware of. These include:

**Not Production-Ready:** The model is not yet optimized and validated for production environments. It may lack the robustness and reliability required for real-world, high-stakes applications.
**Ongoing Improvements:** Due to its experimental nature, the model still has certain inaccuracies and limitations in understanding complex natural language variations.
**Narrow Focus:** The model is designed to address a specific use case - extracting technical requirements from job descriptions. It may not perform optimally for other NLP tasks or general language understanding tasks.
**Data Sensitivity:** The model's performance heavily relies on the quality and diversity of the training data it has been exposed to. It may not perform well with data that significantly differs from its training dataset.
**Fairness and Bias:** As with any language model, there may be biases present in the training data, which could affect the model's predictions and recommendations.

*Please exercise caution and conduct further evaluation before considering this model for any critical applications. Feedback and contributions from the community are welcome to enhance the model's capabilities and address its limitations.*

## Training and evaluation data
The dataset used for training and evaluation comprises web-scraped job posts extracted from LinkedIn. The job posts included in the dataset are specifically targeted at data-related roles, such as Data Scientists, Engineers, Analysts, and Machine Learning specialists.
For the training and validation process, a random split of 70% for training, 15% for validation, and 15% for evaluation was employed. This partitioning ensures that the model's performance is assessed on unseen data, enhancing its ability to generalize to new job descriptions and skills beyond those present in the training set.

Please note that the quality and representativeness of the training data play a crucial role in the model's performance. Efforts were made to curate a diverse and relevant dataset, but the model's efficacy may still be influenced by the data's completeness and diversity. Feedback and contributions from the community are valuable in enhancing the dataset and, consequently, the model's performance.

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| 0.0180        | 1.0   | 729  | 0.1679          | 0.9083    | 0.9394 | 0.9236 | 0.9944   |
| 0.0116        | 2.0   | 1458 | 0.0137          | 0.9303    | 0.9484 | 0.9392 | 0.9956   |


