
#                                             SUPERVISED

#training and testing data sets combind 

test <- read.csv("~/Downloads/titanic (1)/test.csv")
View(test)

train <- read.csv("~/Downloads/titanic (1)/train.csv")
View(train)

train$IsTrainSet <- TRUE
test$IsTrainSet <- FALSE
test$Survived <- NA

titanic.full <- rbind(train,test)
View(titanic.full)
str(titanic.full)

#Data Cleaning
is.na(titanic.full)

#age
titanic.full.clean <- titanic.full[,-c(1,9,11)] #these might not be very useful 
impute <- preProcess(titanic.full.clean, method = "bagImpute")
titanic.full.clean <- predict(impute, titanic.full.clean)
titanic.full.clean$Survived[892:1309] <- NA
View(titanic.full.clean)

#Embark
table(titanic.full.clean$Embarked)
titanic.full.clean$Embarked<- as.character(titanic.full.clean$Embarked)
titanic.full.clean$Embarked[titanic.full.clean$Embarked == ""] <- "S"
table(titanic.full.clean$Embarked)


#Standardizing data

titanic.full.clean.standard <-titanic.full.clean

standard <- function(x)(x-min(x))/(max(x)-min(x))
titanic.full.clean.standard$Age <- standard(titanic.full.clean.standard$Age)
titanic.full.clean.standard$SibSp <- standard(titanic.full.clean.standard$SibSp)
titanic.full.clean.standard$Parch <- standard(titanic.full.clean.standard$Parch)
titanic.full.clean.standard$Fare <- standard(titanic.full.clean.standard$Fare)
head(titanic.full.clean.standard)
str(titanic.full.clean.standard)


# Building a model

titanic_train <- titanic.full.clean.standard[1:891,]
titanic_test <- titanic.full.clean.standard[892:1309,]

set.seed(1122)
TitanicModel_rf <- train(Survived ~., data = titanic_train, method="rf")
TitanicPredict <- predict(TitanicModel_rf, newdata = titanic_test)
confusionMatrix(TitanicPredict, titanic_train$Survived)





#                                           UNSUPERVISED



library(ggplot2)
library(factoextra)
library(purrr)
library(cluster)

#                                           aniso data

ggplot(aniso,aes(x=V1,y=V2))+
  geom_point()

set.seed(2211)


#Elbow approach
fviz_nbclust(aniso[,1:2], kmeans, method = "wss")

km <- kmeans(aniso[,1:2],3,nstart=20)
fviz_cluster(km,data=aniso[,1:2])


#Gap Statistics
gap_stat_aniso <- clusGap(aniso[,1:2], FUN = kmeans, nstart=200,K.max=10,B = 50)
print(gap_stat_aniso, method = "firstmax")
fviz_gap_stat(gap_stat_aniso)

# this one is suggesting k=2?, e.g:
km <- kmeans(aniso[,1:2],2,nstart=20)
fviz_cluster(km,data=aniso[,1:2])  #?



#                                           Noisy_moon data

ggplot(noisy_moons,aes(x=V1,y=V2))+
  geom_point()

#Elbow approach
fviz_nbclust(noisy_moons, kmeans, method = "wss")

set.seed(2311)
km <- kmeans(noisy_moons,2,nstart=20)
fviz_cluster(km,data=noisy_moons)



#Gap Statistics
gap_stat_moon <- clusGap(noisy_moons, FUN = kmeans, nstart=200,K.max=10,B = 50)
print(gap_stat_moon, method = "firstmax")
fviz_gap_stat(gap_stat_moon)


# this one is suggesting k=10?, e.g:
km <- kmeans(noisy_moons,10,nstart=20)
fviz_cluster(km,data=noisy_moons)  #?

#                                          No Structure


ggplot(no_structure,aes(x=V1,y=V2))+
  geom_point()

#Elbow approach
fviz_nbclust(no_structure, kmeans, method = "wss")

km <- kmeans(no_structure,4,nstart=20)
fviz_cluster(km,data=no_structure)

#Gap Statistics
gap_stat_nos <- clusGap(no_structure, FUN = kmeans, nstart=200,K.max=10,B = 50)
print(gap_stat_nos, method = "firstmax")
fviz_gap_stat(gap_stat_nos)

# this one is suggesting k=1?, e.g:
km <- kmeans(gap_stat_nos,1,nstart=20)
fviz_cluster(km,data=no_structure)  #?


#                                          Noisy circles
ggplot(noisy_circles,aes(x=V1,y=V2))+
  geom_point()

#Elbow approach
fviz_nbclust(noisy_circles, kmeans, method = "wss")

km <- kmeans(noisy_circles,4,nstart=20)
fviz_cluster(km,data=noisy_circles)


#Gap Statistics
gap_stat_circl <- clusGap(noisy_circles, FUN = kmeans, nstart=200,K.max=10,B = 50)
print(gap_stat_circl, method = "firstmax")
fviz_gap_stat(gap_stat_circl)

# this one is suggesting k=1?, e.g:
km <- kmeans(noisy_circles,1,nstart=20)
fviz_cluster(km,data=noisy_circles)  #?


#                                          Varied

ggplot(varied,aes(x=V1,y=V2))+
  geom_point()

#Elbow approach
fviz_nbclust(varied, kmeans, method = "wss")

km <- kmeans(varied,3,nstart=20)
fviz_cluster(km,data=varied)


#Gap Statistics
gap_stat_var <- clusGap(varied, FUN = kmeans, nstart=200,K.max=10,B = 50)
print(gap_stat_var, method = "firstmax")
fviz_gap_stat(gap_stat_var)

# this one is suggesting k=3?, e.g:
km <- kmeans(varied,3,nstart=20)
fviz_cluster(km,data=varied)  #?



#                                          Blobs

ggplot(blobs,aes(x=V1,y=V2))+
  geom_point()

#Elbow approach
fviz_nbclust(varied, kmeans, method = "wss")

km <- kmeans(blobs,3,nstart=20)
fviz_cluster(km,data=blobs)


#Gap Statistics
gap_stat_blob <- clusGap(blobs, FUN = kmeans, nstart=200,K.max=10,B = 50)
print(gap_stat_blob, method = "firstmax")
fviz_gap_stat(gap_stat_blob)

# this one is suggesting k=3?, e.g:
km <- kmeans(gap_stat_blob,3,nstart=20)
fviz_cluster(km,data=blobs)  #?

