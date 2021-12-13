## Mini Project Breast Cancer Data

##Data Source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

rm(list=ls())
dev.off()

setwd("c:/CLEMSON/CITI/Workshop/4.Machine Learning/MiniProject/2021_Summer_R/Mini Project(R-Bootcamp)/")
#read CSV

mydata <- read.csv("breast_cancer.csv")


# Look at first 6 rows
head(mydata)

str(mydata)
summary(mydata)


#Find Missing Value

sum(is.na(mydata)) #No missing values

##Target Variable

table(mydata$diagnosis)

#change target variable to factor

mydata$diagnosis<-factor(mydata$diagnosis, labels=c('B','M'))
prop.table(table(mydata$diagnosis))*100

##correlation
corr_mean<- cor(mydata[,c(3:12)],method="pearson")
corr_sec<- cor(mydata[,c(13:22)],method="pearson")
corr_worst<- cor(mydata[,c(23:32)],method="pearson")

#We see that there are extremely high correlations between some of the variables.

##Some more correlations plots

pairs(data=mydata,
      ~ radius_mean + radius_se + radius_worst,)


pairs(data=mydata,
      ~ perimeter_mean + perimeter_se + perimeter_worst,)



#Training and Testing dataset


dataset<-mydata
head(dataset)

set.seed(123)
smp_size <- floor(0.70 * nrow(dataset))
train_ind <- sample(seq_len(nrow(dataset)), size = smp_size)
train <- dataset[train_ind, ]
test <- dataset[-train_ind, ]

#check target variable

prop.table(table(train$diagnosis))*100
prop.table(table(test$diagnosis))*100

names(train)


##Decision Tree
ModFit_rpart <- train(diagnosis~.,data=train[,-1],method="rpart",
                      parms = list(split = "gini"))

#fancier plot
library(rattle)
fancyRpartPlot(ModFit_rpart$finalModel)


predict_rpart <- predict(ModFit_rpart,test)
confusionMatrix(predict_rpart, test$diagnosis)


##Random Forest

ModFit_rf <- train(diagnosis~.,data=train[,-1],method="rf", prox=TRUE)
predict_rf <- predict(ModFit_rf,test)
confusionMatrix(predict_rf, test$diagnosis)


##The positive class of the classification is Benin with accuracy score 94% with decision tree method and 97% with random forest method


