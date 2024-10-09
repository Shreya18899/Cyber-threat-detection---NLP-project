# Cyber-threat-detection---NLP-project
Detecting if information from emails and other sources are indicative of a threat

Dataset - https://www.kaggle.com/datasets/ramoliyafenil/text-based-cyber-threat-detection/code

Comprehensive Threat Analysis -
The dataset includes emails, information on senders, threats, and recipients, and shows how threats spread within a network.
19,940  observations with 21  classes
Attack Label: 
This categorizes the type of cyber threat identified, distinguishing among 21 distinct types of threats (malware, benign, threat actor)


DistilBERT -  Architecture of the Hierarchical model

1. Binary Classification (Threat vs. Benign): The first part of the model determines if the text is a threat.
2. Multiclass Classification (Type of Threat): If the text is determined to be a threat, the second part classifies the type of threat.
3. Linear Layer 1  - of dimension 768*1 with a sigmoid activation acting as a binary classifier
4. Linear Layer 2 of dimension 768 * num of threat classes with a softmax activation acting as a multiclass classifier
4. Hierarchical Loss - summing up the binary loss and the multiclass loss and backpropagating this loss through the network


