# Automated Prediction of Item Difficulty and Item Response Time

## Introduction
Accurately predicting item difficulty and response time is crucial for developing effective and personalized educational assessments. This repository presents a comparative analysis of various deep learning architectures for this task, focusing on their ability to capture semantic nuances within textual items. We base this paper on the BEA 2024 Shared Task, leveraging its provided dataset to benchmark the performance of traditional word embedding approaches with neural networks against more sophisticated models like Long Short-Term Memory (LSTM), LSTM with attention mechanism, Gated Recurrent Unit (GRU), word-embeddings with neural networks and CBOW with neural networks and state-of-the-art language models like BERT and ELECTRA. Each model’s prediction accuracy for both item difficulty and response time is rigorously evaluated and compared. Our findings, situated within the context of the BEA 2024 Shared Task, provide valuable insights into the strengths and limitations of different deep learning techniques for this critical task in educational data mining.

## Methodology
* Dataset:
  * Utilize a large-scale dataset of educational items, each labeled with its empirically determined difficulty level and average response time.
  * The dataset should encompass a diverse range of subjects and item formats to ensure generalizability of the findings
    
* Preprocessing:
  * Tokenization of items, removal of stop words, and potentially stemming or lemmatization will be performed.
  * Word embedding generation will be conducted using techniques like Word2Vec, CBOW, or pre-trained embeddings from BERT and ELECTRA.
 
* Model Implementation and Training:
  * Each model (Word Embeddings with NN, CBOW with NN, LSTM, LSTM with Attention, GRU, BERT, ELECTRA) will be implemented using a deep learning framework like TensorFlow or PyTorch.
  * Models will be trained on the preprocessed dataset, optimizing parameters to minimize prediction error for both item difficulty and response time.
 
* Evaluation:
  * The dataset will be split into training, validation, and test sets.
  * Performance will be evaluated using metrics such as Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) for both item difficulty and response time.
  * The strengths and limitations of each architecture will be analyzed based on their performance and computational efficiency.

## Proposed Approach

* <strong> Word Embeddings with Neural Networks: </strong>
  * This approach uses techniques like Word2Vec or GloVe to create numerical representations of words, feeding them into a feedforward neural network for prediction.
  * This method predicts item difficulty and response time in educational assessments using deep learning techniques. We address the challenge of leveraging the rich textual information inherent in assessment items, which traditional methods often overlook. Our contribution lies in developing a pipeline that effectively transforms textual data into meaningful numerical representations, enabling accurate prediction of item characteristics.

* <strong>CBOW: </strong>
  * This approach uses Continuous bag-of-words method to create numerical representations of words, feeding them into a feedforward neural network for prediction.
  * This method predicts item difficulty and response time in educational assessments using a Continuous Bagof-Words (CBOW) model. Our methodology leverages the power of deep learning to analyze textual data and extract meaningful representations of item characteristics, ultimately improving traditional psychometric analyses.

* <strong>LSTM: </strong>
  * LSTMs excel at processing sequential data like text, learning long-term dependencies within word sequences to understand the context and meaning of the item.
  * This model predicts item difficulty and response time in medical licensing examinations using Long Short Term Memory (LSTM) networks and sentence embedding techniques. The proposed model leverages the textual information embedded in both the question stem and answer choices to capture nuanced relationships and provide accurate predictions.

* <strong>LSTM with Attention Mechanism: </strong>
  * This enhanced LSTM architecture incorporates an attention layer to focus selectively on specific words within the item that contribute most significantly to difficulty and response
  * This method is for predicting item difficulty and response time in educational assessments using Long Short-Term Memory (LSTM) networks enhanced with an attention mechanism. Our proposed model leverages the power of contextualized word embeddings and sequence modeling to capture intricate relationships between item components and their psychometric properties.

* <strong>Gated Recurrent Unit (GRU): </strong>
  * As a slightly simplified alternative to LSTMs, GRUs offer comparable performance with reduced computational complexity.
  * This model predicts item difficulty and response time for medical board examination questions using deep learning techniques. We leverage the power of Gated Recurrent Units (GRUs) to capture the complex relationships between various item features and their psychometric properties.

* <strong>BERT: </strong>
  * This cutting-edge language models, pre-trained on vast text corpora, provide rich, contextualized word embeddings, capturing intricate semantic relationships for potentially superior performance.
  * The reason for using BERT is unlike traditional bag-of-words approaches, BERT captures contextual information within the question text, allowing for a more nuanced understanding of semantic meaning and difficulty.

* <strong>ELECTRA: </strong>
  * This cutting-edge language models, pre-trained on vast text corpora, provide rich, contextualized word embeddings, capturing intricate semantic relationships for potentially superior performance.
  * We use a fine-tuned ELECTRA model, specifically adapted for regression tasks, to predict item difficulty on a continuous scale. By training on a rich dataset of item stems and their corresponding human-annotated difficulty scores, our model learns to capture the subtle linguistic nuances and content complexities that contribute to item difficulty.

## Experiments and Results
We ran all seven methods and evaluated them using RMSE [Root Mean Squared Estimation]. The evaluation matric we use is the DummyRegressor Baseline Difficulty and DummyRegressor Baseline Response Time which are given by the organizers of BEA 2024:

![lmao](https://github.com/Harish-Balaji-B/Automated-Prediction-of-Item-Difficulty-and-Item-Response-Time/blob/main/Samples/dummy.png)<br>

The results we got out of our methods are as follows:

![lmao](https://github.com/Harish-Balaji-B/Automated-Prediction-of-Item-Difficulty-and-Item-Response-Time/blob/main/Samples/difficulty.png)<br>

This shows that LLM’s such as BERT and ELECTRA outperform other methods, and is also less than the DummyRegressor

Now for Response Time: <br>

![lmao](https://github.com/Harish-Balaji-B/Automated-Prediction-of-Item-Difficulty-and-Item-Response-Time/blob/main/Samples/response.png)<br>

Here too, the LLM’s dominate as they out-perform other methods and are less than the DummyRegressor.

## Discussions
Our experiments reveal several interesting findings.

* Firstly,
  * RT and ELECTRA significantly outperform the baseline Dummy Regressor in both difficulty and response time prediction.
  * This suggests that these transformer-based models can effectively capture intricate patterns within the data that contribute to these metrics.
  * Notably, BERT achieves a substantial reduction in response time compared to all other models.
 
* Secondly.
  * The difficulty scores across most models remain relatively close to the baseline.
  * This observation implies that the current features might not be sufficiently discriminative for accurately predicting difficulty levels.
  * Future work will explore advanced feature engineering techniques and data augmentation strategies to address this limitation.
 
* The variance in response time predictions across the different models indicates that specific model architectures capture different aspects of the data relevant to this task.

* BERT’s success highlights its ability to learn complex relationships between input features and response time.

![lmao](https://github.com/Harish-Balaji-B/Automated-Prediction-of-Item-Difficulty-and-Item-Response-Time/blob/main/Samples/comparisson.png)<br>

With respect to Difficulty Prediction, we can infer that,

* Simple NN, LSTM with Attention and LSTM have similar RMSE scores for Difficulty Prediction.
* Whereas CBOW and GRU have a larger RMSE which makes them ineffective.
* But BERT and ELECTRA have significantly low RMSE with BERT being the lowest which tells us that in this comparison model, BERT and ELECTRA are suited for Difficulty Prediction.

With respect to Response Time Prediction, we can infer that,

* BERT and CBOW have similar RMSE scores for Response Time,
* Whereas Simple NN, LSTM with Attention, LSTM, GRU have higher RMSE which makes them ineffective for Response Time prediction.
* BERT has the lowest REMSE score which tells that in this comparison model, BERT is suited for Response Time Prediction.

## Future Scope
Several promising avenues exist for extending this research:

* <strong>Feature Engineering and Data Augmentation: </strong> Exploring novel features, such as linguistic complexity metrics and task-specific attributes, could enhance difficulty prediction. Additionally, augmenting the dataset with carefully crafted synthetic examples, especially for challenging tasks, could improve model robustness and generalization ability.
* <strong>Hyperparameter Optimization and Model Finetuning: </strong> Systematically optimizing hyperparameters for top-performing models like BERT and ELECTRA, potentially through techniques like Bayesian Optimization, could further enhance their performance. Fine-tuning these pre-trained models on a larger, task-specific dataset is also crucial.
* <strong> Explainable AI: </strong> Employing techniques like attention visualization and feature importance analysis could provide insights into the decision-making process of these complex models, particularly for understanding the factors driving difficulty and response time predictions.
* <strong>Real-world Deployment and Evaluation: </strong> Ultimately, deploying these models in real-world applications and evaluating their impact on user experience would be crucial. This would involve collecting user feedback, analyzing performance under different conditions, and iteratively improving the models based on real-world insights.
