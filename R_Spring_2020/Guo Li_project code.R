# Guo Li_Mini project R code 
setwd("C:/Users/Guo/Desktop/project")
rm(list=ls())

library(rpart)    

# Read original data, I already integrated gender_submission.csv into test.csv 
train0 <- read.csv("C:\\Users\\Guo\\Desktop\\project\\train.csv", header = TRUE)
test0 <- read.csv("C:\\Users\\Guo\\Desktop\\project\\test.csv", header = TRUE)

# Data cleaning1: dropped columns "PassengerId,Name,Ticket,Cabin" because they are not very useful in this prediction
train1 <- subset(train0, select = -c(PassengerId,Name,Ticket,Cabin) )
test1 <- subset(test0, select = -c(PassengerId,Name,Ticket,Cabin) )

# Data cleaning2: dropped rows with NA entries or empty entries
train2 <- na.omit(train1)
training <- train2[-which(train2$Embarked == ""), ]
testing <- na.omit(test1)

# Build model with decision tree classification
model_dt<-rpart(Survived~., data=training,method = "class")
predictions<-predict(model_dt,newdata=testing,type="class")

# Prediction results (0.92145 accuracy rate)
table(testing$Survived,predictions)
mean(testing$Survived==predictions)



