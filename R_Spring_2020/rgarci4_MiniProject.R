#Name: Ricardo Garcia Carcamo
rm(list = ls())
setwd("G:/My Drive/Clemson/R Scripts/R_Machine_Learning-master (Clemson ML Course)/MiniProject")
library(dplyr)
library(base)
library(caret)
library(rattle)
library(factoextra)
library(purrr)
library(ggplot2)
library(mclust)

#Read data set
test=read.csv("test.csv")
training=read.csv("train.csv")
results=read.csv("gender_submission.csv")

#Supervised method
print("First a summaty of the data is computed:")
dim(training)
names(training)
print("Check if the descriptive variables match:") 
rbind(names(training),names(training)%in%names(test))
rbind(names(test),names(test)%in%names(training))
summary(training)
print("The missing values per column are computed:")
Missings=colSums(is.na(training))
print(Missings)
print("Many missing values in Age were found")
#Compute a sub set of train data without NA
subtrain=na.omit(training)
print("To evaluate a possible correlation in the Age variable the correlation coefficient is calculated:")
cor(subtrain$Age,subtrain$PassengerId)
cor(subtrain$Age,as.numeric(subtrain$Ticket))
cor(subtrain$Age,as.numeric(subtrain$Fare))
print("The missing values are filled with an estimation of the age")
trainImputeBag = preProcess(training,method="bagImpute")
training2=predict(trainImputeBag,training)
training2=training2[,c(2,3,5,6,7,8,10,12)]
print("The meaningful variables are filtered")
training2$Survived=as.factor(training2$Survived)
test2=test
test2=test2[,names(training2[,-1])]


#Do the Decision Tree model
ModFit_DT=train(Survived~.,data=training2,method="rpart")
fancyRpartPlot(ModFit_DT$finalModel)
test3=merge(test,results,by = "PassengerId")
test3=test3[complete.cases(test3),]
predict_rpart = predict(ModFit_DT,test3)
confusionMatrix(predict_rpart,as.factor(test3$Survived))

#Filter desired Variables
#training3=na.omit(training2)
#training3=training3[c(1,2,3,4)]
#training3$Sex=as.numeric(training3$Sex)
#km <- kmeans(training3,3,iter.max = "15")
#fviz_cluster(km,data = training3[,c(2,4)])

#Unsupervised learning
#Change the directory
setwd("G:/My Drive/Clemson/R Scripts/R_Machine_Learning-master (Clemson ML Course)/MiniProject/data")
#Read data set
dataset1=read.csv("aniso.csv")
Missings=colSums(is.na(dataset1))
print(Missings)
print("Try kmeans method")
ggplot(data = dataset1,aes(V1,V2))+geom_point()
KmeansCluster = kmeans(dataset1,3,iter.max = "50")
ggplot(data = dataset1,aes(V1,V2,color=factor(KmeansCluster$cluster)))+geom_point()+labs(color="Cluster",title = "K means clustering")
print("Kmeans method is not appropiate")
print("Change to Gaussian Mixture Modelling")
gmm.mclust <- Mclust(dataset1, 3)
gmm.mclust$classification
ggplot(data = dataset1,aes(V1,V2,color=factor(gmm.mclust$classification)))+geom_point()+labs(color="Cluster",title = "Gaussian Mixture Modelling")
print("Gaussian Mixture Modelling is adecuate method")
