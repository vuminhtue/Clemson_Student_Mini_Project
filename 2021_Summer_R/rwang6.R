# 1. Objective: Predicting survival on the Titanic

# 2. For the data, there are following variables:
# plcass(ticket class), 
# sex, 
# age, 
# sibsp(# of siblings/spouses aboard the Titanic), 
# parch(# of parents aboard the Titanic), 
# ticket(ticket number), 
# fare(passenger fare),
# cabin(cabin #), 
# embarked(port of embarkation)

#3. Type of ML model output: Classification

# Load data: dounload from https://www.kaggle.com/c/titanic/data, and save at desktop
train<- read.csv("desktop/train.csv",stringsAsFactors = FALSE)
test<- read.csv("desktop/test.csv")
validation<-read.csv("desktop/gender_submission.csv")
str(train)

library(caret)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(class)

# train model using decision tree
set.seed(123)
fit <- rpart(Survived ~ Pclass + Sex + SibSp + Parch + Age,
             method="class", data=train)
fancyRpartPlot(fit)

# predict the output from testing set
library(rattle)
prediction<-predict(fit,test,type="class")

#evaluate the output use Confusion Matrix
results<- table(prediction, validation$Survived)
confusionMatrix(results)

# The accuracy of this ML model is 0.9665, and P-value is very small,
# so this model is good.
