##  Abstract
This task is considered the cornerstone of this project because we rely on accurately extracting skills from the job description to help us build a correct database that represents both the job description and the mentioned skills aspect. For this reason, we utilized more than one method, including Spacy, LSTM, and BERT, and compared them in terms of accuracy. To obtain clean data, it was important for us to explore various recruitment sites such as LinkedIn, Wuzzuf, and Indeed. We used GPT-3 to assist us in acquiring the necessary skills to start with, then refined the results to obtain clean data.

##  Introduction
In this project, we use a dataset scraped from recruitment websites. This is a dataset of job descriptions. Our mission is to create models to extract the skills required from each job description. We take unlabeled data, use the GPT model to label it, and modify it ourselves. We then use this synthetic dataset to train some other models. We implement the distillation concept in this way. We then evaluate our models. We have chosen many models, from less complex to very complex. We have to keep in mind that the task is not completely objective because different people have different definitions of skills and, therefore, it may be difficult to evaluate models based on scores alone.

##  Related Work

In our investigation of relevant literature for our project, we found several noteworthy papers.

- Our primary paper "Skill Extraction from Job Postings Using Weak Supervision" (2022) by Zhang et al. described a skill extraction method that used Semi-Supervised Learning for this task. The paper's References were a key resource in finding other related works.

- One such paper is "Retrieving Skills from Job Descriptions: A Language Model Based Extreme Multi-label Classification Framework" (2020) by Bhola et al. This work proposes an innovative language model-based approach to skill extraction from job descriptions through extreme multi-label classification. The authors fine-tune pre-trained language models on a large dataset of job descriptions and evaluate their method through experiments against baseline methods.



- Furthermore, "Deep Job Understanding at LinkedIn" (2020) by Li et al. presents a comprehensive approach to job recommendation through a Job Understanding system developed by LinkedIn. The system incorporates deep learning models, natural language processing, and graph-based algorithms to provide personalized job recommendations to LinkedIn users.

- Finally, "SpanBERT: Improving Pre-training by Representing and Predicting Spans" (2020) by Joshi et al. presents the SpanBERT language model, which extends the BERT model by capturing span representations in text. The authors introduce a new training objective to encourage the model to predict spans of text rather than individual tokens and evaluate its effectiveness against other state-of-the-art models. The paper presents a novel approach to language modeling and provides valuable insights into the importance of span representations in text.

##  Dataset(s)
* In this project, we used job advertisement data from various employment sites, where the data consists of the job title and job description, and our goal was to extract data from this description.
* We used other data about skills needed for jobs. We used it to confirm model outputs and filtering skills to check data quality

When analyzing the job descriptions, we noticed that the average token length was around 560, indicating that the job descriptions were relatively long. This prompted us to leverage the power of GPT-3 to extract the skills required for each job. The model succeeded in extracting a total of 3,761 unique skills. We used other pre-trained models and compared their output with data on skills to verify them. (The initial primary goal was to create clean data and extract the largest possible amount of skills present in the job description to help us train other models)

However, we encountered challenges with the extracted skills, as the model made several mistakes and included non-skill-related information. Some examples of erroneously extracted skills included terms such as "full-time," "100," and "$". Furthermore, the model occasionally produced skills in incorrect formats, such as "Pandas / Pytorch /" instead of separate skills like "Pandas" and "Pytorch."

To address these issues, our team decided to perform data cleaning on the extracted skills. Each team member was assigned approximately 750 skills for data cleaning purposes. During this process, we focused on removing incorrect skills like "full-time," "100," and "$". Additionally, we ensured that the skills were in the appropriate format by making necessary adjustments. For instance, we transformed "Pandas / Pytorch /" into separate skills, namely "Pandas" and "Pytorch."

After completing the data cleaning phase, we obtained a final set of 1,933 unique skills that accurately represent the required skills from the job descriptions. These skills underwent a meticulous cleaning process, enhancing the overall quality and usability of the extracted data.

##  Methods
[*LSTM model:*] (https://github.com/Galal-pic/GD-Project/tree/main/Extract_skills_from_Job_Descriptions/lstm)
We used a single layer, bidirectional LSTM model with embedding size of 75 and hidden dimension size of 50. We trained the model for 20 epochs as we noticed that the "dev set" macro f1 score reaches a saturation point at 20 epochs. The f1 score for the skill recognition task is 0.70 and the macro f1 score for the tag classification task is 0.73.

[*Spacy:*] (https://github.com/Galal-pic/GD-Project/tree/main/Extract_skills_from_Job_Descriptions/Spacy_model)
The performance of the Spacy model was stronger than LSTM while using a much smaller size of data, and this was noticeable. I relied on training the model en_core_web_sm, which is able to understand the language in general, then I modified it by training it on data in a specific way, and the results were F1 = 0.86

[*BERT model:*] (https://github.com/Galal-pic/GD-Project/tree/main/Extract_skills_from_Job_Descriptions/BERT)
The bert_base_cased model performs better than the Spacy model and the LSTM model. The F1 score is around 0.939281. In the BERT mode, the AutoTokenizer.from\_pretrained function is utilized to load a tokenizer from the Hugging Face model hub. In this instance, the tokenizer for "BERT_base_cased " is loaded with the add\_prefix\_space=True parameter, which handles situations where a leading space is required before each input example in the language model. For token classification tasks, the DataCollatorForTokenClassification class is employed to gather and preprocess the data. It takes the tokenizer as input to manage the tokenization and padding of input sequences. To load a pre-trained model for token classification, the AutoModelForTokenClassification.from\_pretrained method is used. In this scenario, the "bert-base-cased" model is loaded with the num\_labels=3 parameter, indicating the number of labels (classes) in the token classification task. Additionally, id2label and label2id mappings are required to convert between label indices and their corresponding string labels. The TrainingArguments class is responsible for defining the training arguments and hyperparameters. It encompasses various settings such as the output directory for saving the trained model, learning rate = 2e-5, per\_device\_train\_batch\_size and per\_device\_eval\_batch\_size are set to 4, weight\_decay is set to 0.01. The load\_best\_model\_at\_end parameter is set to True to load the best model at the end of training. The number of epochs, as specified in the TrainingArguments, is set to 2. This means that the training process will iterate over the entire training dataset twice.

## Experiments
For the LSTM model, we tried to use additional features like is_lower (if the token is lowercase), is_title (if only the first alphabet of the token is uppercase) etc., but these did not improve the results. This was our baseline model, we improved on this model by using BERT and Spacy models. The detailed results that we got with these models are listed below and the detailed hyper-parameters used for these models are mentioned above.


## Results
| **Model** | **Precision** | **Recall** | **F1 Score** |
| --- | --- | --- | --- |
| BERT | 0.93 | 0.94 | 0.93|
| Spacy_NER | 0.89 | 0.84 | 0.86  |
| LSTM | 0.70 | 0.72 | 0.71 |
