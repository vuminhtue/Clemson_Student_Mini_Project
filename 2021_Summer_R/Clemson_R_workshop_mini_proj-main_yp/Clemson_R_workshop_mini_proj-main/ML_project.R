rm(list=ls())
dev.off()
library(data.table)
library(stringr)
library(randomForest)
library(caret)

setwd('c:/CLEMSON/CITI/Workshop/4.Machine Learning/MiniProject/2021_Summer_R/Clemson_R_workshop_mini_proj-main_yp/Clemson_R_workshop_mini_proj-main/')
#Read in data
data = read.delim("Normalized_transcriptome_counts.txt")

#Set the first column as row name 
rownames(data) <- data[,1] #Assigning row names from 1st column 
data[,1] <- NULL #Removing the first column


#Transpose data
df <- transpose(data)
rownames(df) <- colnames(data)
colnames(df) <- rownames(data)

#Create a vector that is factor for category/class (PE) 
category <- rownames(df)
df1 <- cbind(category, df)
PE <- substring(df1$category, 1, 1)
df1$category <- as.factor(PE)

#Create training and testing set
parsing <- createDataPartition(y=df1$category, p=0.6, list = FALSE)
training <- df1[parsing,]
testing <- df1[-parsing,]

#Separating out the predictor: category/class
train_class <- training$category
train_data <- training[,-1]
#install.packages("e1071")

#Building the model using caret, but took too long
#model <- train(train_data,train_class,method = 'rf',prox = TRUE)
#Building the model using Random Forest Pacakge
model <- randomForest(train_data, train_class, proximity = TRUE)


#Applying model to predict the output on testing data set
testing_class <- testing$category
testing_data <- testing[,-1]
predict_df <- predict(model, testing_data)


#Evaluating accuracy using Confusion Matrix
confusionMatrix(predict_df, testing_class)


