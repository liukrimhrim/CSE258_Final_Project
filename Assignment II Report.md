# A Collaborative Filtering Approach to Build a Restaurant Recommendation System Using Yelp Dataset



## Dataset

The dataset we use is from Yelp. Specifically, we use their data of users, businesses and reviews. The format is json, and one can refer to the official website for examples of the structure and content of each json entry. 

This is a gigantic dataset, the size of which is over 5 GB. We only use a small subset of it to do our project: we only consider data related to restaurants in Las Vegas. The reasons for picking this subset are as follows: 1. based on our exploration, most reviews are about businesses in Las Vegas; 2. most business being reviewed are restaurants; 3. we assume that businesses of the same type (restaurants in our case) in the same city should have latent relations to be discovered. 

We used the attributes 'city': 'Las Vegas' and 'state': 'NV' to filter out all the businesses in Las Vegas. Then, we picked out all the businesses that contains the words 'Restaurant' and 'Food' in the category tags. This subset has NUM reviews, and NUM restaurants and NUM users associated. The distribution of the ratings is as the bar plot.

We then shuffled the dataset, and divided it into training set, validation set and testing set roughly to the ratio 5:2:3. 

For the validation and testing set, since our predictive task is going to be predicting binary values, we preprocess the sets as such: we made a triple out of each entry of the validation and testing set. The first and the second element of the triple would be the user ID and business ID. As for the third element, it is either 0 or 1, depending on whether the rating by the user on the restaurants are higher than the average rating the user gave in all his or her reviews. The average ratings of users are provided in the json file regarding user profiles. One should be noted that the average rating is computed from **all** the reviews an user has submitted, which means some of the reviews might be out-sample. However, it is reasonable to assume that out in-sample average should be close to this overall average.

For the training set, we still found it too large (377 MB). Also, for some of our models (latent class and collaborative filtering), we preprocess the training data into a matrix, where the rows are the users, and the columns are the restaurants and each entry is either 0 (negative sample) or 1 (positive sample) or -1 (null entry). Then we found that the matrix is extremely large yet sparse, in the sense that most of the entries are null entry. This is because many of the users only make occasional reviews, and many of the businesses are barely reviewed by the users. Such training data would be unnecessarily hard to train and it is unlikely to get promising results from it (we can hardly learn anything from users who have made only one review, or businesses which have been reviewed by only a handful of users). Therefore, we decided to further shrink down our training set. We selected out the 800 most popular restaurants which have been reviewed the most, and the 5000 most active users who have submitted the most reviews. It happened that 10 of these 5000 most active users have never reviewed any of those 800 restaurants, so we just crossed out the 10 users. Eventually, we made a small trainning set with 800 restaurants, 4990 users, and NUM reviews made by one the 4990 users on one of the 800 restaurants. The density of this smaller matrix is NUM.       

## Pedictive Task

The predictive task we are performing is pretty straightforward: we are to predict whether an user would like a restaurant which he or she has never reviewed. The prediction shall have binary values: 1 represents the user would like the restaurant, and 0 represents the otherwise. We define "User A likes a restaurant B" as: if the predicted rating of A on B is higher than the average rating A gave in his or her previous reviews, then A likes B. As said, the average ratings of each user are provided in the dataset.

Our prediction is easy to assess: our prediction is either 0 or 1, and our preprocessed test set is also mapping a pair of user and restaurant to either 0 or 1, so we just see if the binary values match.

Predictions with high accuracy is trivial only if the dataset is highly imbalanced, but this is improbable for our case. We use the average rating of a user as the threshold, even though the average rating was not calculated merely from the reviews in our training set, our in-sample average is not supposed to deviate too far away from this overall average. Since the threshold is the average, we can expect that the amount of positive samples and the amount of negative samples shall be balanced. Therefore, the predictions of our models should be significant and not trivial.

## Previous Literature

There are dozens of works have been done on the Yelp dataset. The approaches could be divided into two general types: one is collaborative filtering, which does not take into consideration at all the textual content of users' reviews itself; and information filtering, which does pay attention to the content of the review texts. 

## Naive Baseline Model

Following Koren (2008), and Sawant (2013), we will use a naive baseline model to predict the rating of a new pair of user and restaurant: $r_{u,b}=\mu+d_u+d_b=\mbox{avg}_u+\mbox{avg}_b-\mu$, where $r_{u,b}$ is the predicted rating of user $u$ on restaurant $b$, $\mu$ is the average of all the ratings in our trainning set, $d_u$ denotes how much user $u$'s average rating deviates from $\mu$, and $d_b$ denotes how much the average rating of restaurant $b$ deviates from $\mu$, $\mbox{avg}_u$ is user $u$'s average rating, $\mbox{avg}_b$ is the average rating of restaurant $b$. Once we got the predicted rating, we just need to compare it to $\mbox{avg}_u$ to determine the binary value. Hence, in fact, we just need to check if $\mbox{avg}_b \geq \mu$. 

This model is naive in the sense that, given a dataset, we can immediately calculate the prediction analytically, so there is no learning process involved whatsoever. Also, this naive model has no extensibility at all to new users and restaurants. But anyway, as a baseline, this model suffices.

## Our Models

In our project, we experimented with following models: collaborative filtering based on Jaccard similarity, latent class mixture model optimized with EM algorithm, latent factor model and linear model. 

The drawbacks of this model are also apparent. First of all, it is incredibly computational heavy. Even for a small training set like ours, it will take a long time to compute the CPT. Even though we knew that had we used a larger trainning matrix or assumed more hidden states, we would have got much better results, we still chose not to do so, simply because we have very limited computational resources. Secondly, our model is not quite extensible to new businesses. When we take new businesses into consideration, we have redo the entire training process again to compute a new CPT, which, as we said, is incredibly computational heavy task.

[1]: https://www.yelp.com/dataset/documentation/json	"Yelp Dataset "

