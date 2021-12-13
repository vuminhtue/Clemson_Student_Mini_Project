### Mini Project (dmalmei@clemson.edu)

## Dataset retrieved from kaggle.com/datasets; 
# 1st the iris dataset was downloaded 
# 2nd the iris dataset was stored in the project directory ~/Desktop/Mini_project as Iris.csv

## The objective of this project is to classify the species of an Iris flower based on 4 properties (measurements) of this flower using a ML model. 

#Required packages #
install.packages("kernlab")

# Required Library #
library(caret) # Tool for machine learning projects in R.

## The chosen dataset contains 150 observations of iris flowers. 
## The first 4 columns in this mini project dataset are the predictor, which are the variables used to predict/classify a specific species.
## The predictor are measurements of the flowers in centimetres. 
## The last column (5th) is the species of the flower observed (there are three different species in this dataset).

#input<-dataset[,1:4] # predictor
#target<-dataset[ ,5] #predictand


##The type of ML model output for this dataset is classification, as we will use some labelled properties of a flower to classify its species.


##Load the dataset (Since the R platform provides this dataset, there is no need to read it from the .csv file using the function read.csv)

# attach the iris dataset to the environment
data(iris)

# rename the dataset
dataset <- iris

##The input data does not require to be cleaned or standardized

##Split data into training and testing

## create a list of 60% of the rows in the original dataset we can use for training
test_index <- createDataPartition(dataset$Species, p=0.60, list=FALSE)
## select 40% of the data for testing (data that we won't present to the network during training)
test <- dataset[-test_index,]
# use the remaining 60% of data to training the model
dataset <- dataset[test_index,]

#Dimension of Iris Dataset: 120 instances (80% 120/150) and 5 attributes
dim(dataset)

## list types for each attribute; (uncomment next line to check)
# sapply(dataset, class)

## first 5 rows of the data (uncomment next line to check)
#head(dataset)

## list the levels for the class (uncomment next line to check)
#levels(dataset$Species)

## summarize the class distribution (uncomment next line to check)
#percentage<-prop.table(table(dataset$Species)) * 100
#cbind(freq = table(dataset$Species), percentage = percentage)

## summarize attribute distributions (uncomment next line to check)
#summary(dataset)

## Run algorithms using 10-fold cross validation 
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

## Althought the data distribution seemed linear in some dimensions to choose a good fitting algorithm I applied the following 5: 
# linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
# nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
# advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)

## Since The LDA was the most accurate model I tested and evaluated the model using linear algorithms. 
## The confusion matrix is used to evaluate the model. It shows us how many of each species the model was able to classify correctly.  

predictions<-predict(fit.lda, test)
confusionMatrix(predictions, test$Species) # compare the classified/predicted by the ML model with the "corrects ones" that were previously labelled on the dataset. 

## The model correctly classified most of the testing inputs (40% of the whole dataset), which means it is a reliable model for this dataset.

