---
title: Making a simple model with data generated from tweets
layout: post
---

This is a short demonstration of the text processing capabilities of R as well as a very basic example of how one can go from text based data to a simple model. It showcases a way of turning human-readable tweets into numerical data.

The data used in this example comes from the Kaggle competition called *Partly Sunny with a Chance of Hastags*. The dataset was contributed by the data library CrowdFlower and contains labelled tweets about the weather. I will use the training set provided for this competition and aims to build a model that predicts the negative sentiment of each tweet.

Let's get computing!

Looking at the data and building the toolbox
--------------------------------------------

Since we are going to do some data munging to get everything nice and orderly we use the tidyverse, readr to read data from the harddrive efficiently and tibble to store it in memory. GGplot2 creates nice visualizations. The real work horse in this example is the fantastic tm, or *text mining* package which supplies us with the option of transforming the tweets into numerical data. At last we are going build a LASSO regression model and keep the modelling data stored in a sparse matrix format that only keeps the non zero elements of the matrix.

``` r
library(tidyverse)
library(tm)
library(Matrix)
library(glmnet)
```

We read in the data but only keep two variables, the tweet itself and the column s2 that corresponds to the negative sentiment of the tweet. We also store the s2 values as a matrix for the later modelling.

Below we see some examples of negative tweets

``` r
#Read data from csv-file
tweetData <- read_csv(file = "train.csv", 
                      col_types = cols_only(tweet = col_character(),
                                            s2 = col_double()))

#Save negative sentiment in a vector that is to be predicted in later modelling
s2 <- as.matrix(tweetData$s2)

#Print some tweets ordered by negative sentiment
arrange(tweetData, desc(s2))
```

    ## # A tibble: 77,946 × 2
    ##                                                                          tweet
    ##                                                                          <chr>
    ## 1  I'm gonna do big things today: running during shit weather and writing my e
    ## 2                                                @mention cold cloudy rainy :(
    ## 3                                                             too hot out here
    ## 4  I'm SO happy we're getting this race in today but goodness... This humidity
    ## 5             My client said its not cold outside girl puhlezz its bur is shyt
    ## 6            Dang only 72degrees today, where is all the warm spring weather??
    ## 7  @mention I do have to agree with you, My lawn is getting about 12" tall, It
    ## 8  RT @mention: Sick of wearin a hoodie one day n the next day wearin flip flo
    ## 9  I hate when a shoot is cancelled & rescheduled due to weather then an hour 
    ## 10                                        My boxer is not liking this weather!
    ## # ... with 77,936 more rows, and 1 more variables: s2 <dbl>

We already get a few ideas on what might be good signals of negative sentiment in the tweets. For example, the first example contains a swear word, shit, which makes the tweet negative. Our goal today is to find a list of such words. The approach is pretty simple, we want to create a simple vectorisation of the tweets and then use those vectors as inputs to a machine learning model.

In this case we are satisfied with just treating the tweets as bag of words. That is, each position in the vector corresponds to a certain word and its value to the number of times the word occurs in the tweet. If we had two tweets that said:

1. *The weather is bad*
2. *Ugh, bad bad weather*

We could construct the vectors, v_1, v_2 where v = (n_the, n_\weather, n_\is, n_\bad, n_\ugh) as:

- v_1 = (1,1,1,1,0) 
- v_2 = (0,1,0,2,1)

We also need to process the tweets abit before we go ahead and create the vectors. We want to remove punctuation and stop words since these does not carry any interesting information about the sentiment. Below we see the code to perform this in R. First we create a data representation called corpus. We want to make sure that we are only working with a plain text source, so we transform our corpus using the *tm_map* function. This function is a wrapper to work with the corpus files in the tm package. The first argument it takes is the corpus and the rest are functions with additional arguments. 

After the punctuations and the stopwords are removed we also stem the document, this means that we will represent similar words that differ only in their endings as one word. The stemmed versions of *terrible* and *terribly* would be *terribl*. We then create the tweet vectors using the DocumentTermMatrix function. 

At last we pass the document term matrix through the removeSpareTerms function which removes words that are very rare in the corpus. In our example with the tweets above we could for example remove the words *the*, *Ugh* and *is* because they are only present in a single tweet or observation.

``` r
#Create a corpus representation of the tweets
corp <- Corpus(VectorSource(tweetData$tweet))
corp <- tm_map(corp, PlainTextDocument)
# Removes punctiation 
corp <- tm_map(corp, removePunctuation)
# Remove stopwords
corp <- tm_map(corp, removeWords, stopwords("en")) 
# Stem words in the tweets         
corp <- tm_map(corp, stemDocument, language = "english")
# Create the matrix and remove unuseful terms

dtm <- DocumentTermMatrix(corp)                             
dtm <- removeSparseTerms(dtm, 0.999)
```
So finally, we have our data processed and ready. Now, since our goal is to create a list of words that are associated with tweets about bad weather, we use a simple model that is easy to interpret. We use a regularized version of the linear regression which combines L1 and L2 regularization, also called elastic net logistic regression. 

We prepare our data using the built in spare.model.matrix function. To find the appropriate amount of regularization we use a cross-validation approach to find it. The plot below shows the error plotted against different values of the lambda parameter. We choose to use the value that minimizes the error.

``` r
data <- cbind(s2, data.frame(as.matrix(dtm))) 
# Create a sparse model matrix to be used as input in the model
mat <- sparse.model.matrix(s2~. ,data)

# Use cross-validation to optimize the regularization
model <- cv.glmnet(y = s2, x = mat, alpha = 0.5)            
```

![](assets/2017-09-19-fig1.png)

At last we refit our model to the data using the tuned lambda value. The model we are using is linear so each word has its own beta parameter. The value of the parameter can be intepreted as the marginal increase of expected negativity score if the word is in the tweet. So a higher parameter value indicates that the word has a negative sentiment. Below we see the list of the 10 words that has the highest parameter estimates.

``` r
finalModel <- glmnet(y = s2, x = mat, alpha = 0.5, 
                     lambda = model$lambda.min)
coef <- tibble(variable = row.names(finalModel$beta), 
               beta = as.numeric(finalModel$beta))
arrange(coef, desc(beta))
```

    ## # A tibble: 1,077 × 2
    ##    variable      beta
    ##       <chr>     <dbl>
    ## 1    crappi 0.5117718
    ## 2    shitti 0.4595968
    ## 3   horribl 0.4562549
    ## 4   depress 0.4292953
    ## 5      suck 0.4175831
    ## 6     nasti 0.4157189
    ## 7    dreari 0.4069617
    ## 8     miser 0.3783830
    ## 9      crap 0.3428576
    ## 10  terribl 0.3324477
    ## # ... with 1,067 more rows

As we see in the model, the words are stemmed as discussed before. It is obvious that our model managed to find negative words such as *crappi*, *shitti* or *depress*.
