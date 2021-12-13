rm(list=ls())
dev.off()
library(ggplot2)
library(Amelia)
library(ggthemes)
library(randomForest)
library(mice)
library(scales)
library(dplyr)
library(caret)

setwd('c:/CLEMSON/CITI/Workshop/4.Machine Learning/MiniProject/2021_Summer_R/changar/')
fig <- function(width, heigth){
     options(repr.plot.width = width, repr.plot.height = heigth)
}

titanic_test <- read.csv(file='test.csv')
titanic_train <- read.csv(file='train.csv')
comp <- bind_rows(titanic_train,titanic_test)
str(comp)

colSums(is.na(comp))

missmap(comp,col=c("red","black"))

table(comp$Survived)

prop.table(table(comp$Survived))

prop.table(table(comp$Sex, comp$Survived),1)

#Missing Embarkment
which (comp$Embarked=='')

fig(10,5)
# Get rid of our missing passenger IDs
embarkfare <- comp %>%
  filter(PassengerId != 62 & PassengerId != 830)

# Use ggplot2 to visualize embarkment, passenger class, & median fare
ggplot(embarkfare, aes(x = Embarked, y = Fare, fill = factor(Pclass))) +
  geom_boxplot() +
  geom_hline(aes(yintercept=80), 
    colour='gold', linetype='dashed', lwd=3) +
  scale_y_continuous(labels=dollar_format()) +
  theme_few()

#Higher fare passengers mostly embarked from Charbourg or "C", hence changing NA to "C"
comp$Embarked[comp$Embarked=='']="C"

#Missing Fare value
which (is.na(comp$Fare))

# Replacing missing fare value with median fare for class and embarkment
comp$Fare[1044] <- median(comp[comp$Pclass == '3' & comp$Embarked == 'S', ]$Fare, na.rm = TRUE)

colSums(comp=="")

#Total missing age values
sum(is.na(comp$Age))

vfactors <- c('PassengerId','Pclass','Sex','Embarked',
                 'Survived','Parch','Fare')

comp[vfactors] <- lapply(comp[vfactors], function(x) as.factor(x))

set.seed(250)

# Perform mice imputation, excluding certain less-than-useful variables:
micemod <- mice(comp[, !names(comp) %in% c('PassengerId','Name','Ticket','Cabin','Survived','Parch','Fare')], method='rf') 

mice_output <- complete(micemod)

# Plot age distributions
par(mfrow=c(1,2))
hist(comp$Age, freq=F, main='Age: Original Data', 
  col='palevioletred1', ylim=c(0,0.04),xlab='Original Age')
hist(mice_output$Age, freq=F, main='Age: MICE Output', 
  col='lightgreen', ylim=c(0,0.04),xlab='Age from MICE')

comp$Age <- mice_output$Age

colSums(is.na(comp))

md.pattern(comp)

sum(!is.na(comp$Survived))

# Relationship between Age and survival:
ggplot(comp[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram(binwidth=3) + 
  theme_few()

# Relationship between Sex and survival:
ggplot(data=comp[1:891,],aes(x=Sex,fill=Survived))+geom_bar()

# Relationship between Age and sex:
ggplot(comp[1:891,], aes(Age, fill = factor(Sex))) + 
  geom_histogram(binwidth=5) + 
  theme_few()

# Relationship between Embarked and survival:
ggplot(data = comp[1:891,],aes(x=Embarked,fill=Survived))+geom_bar(position="fill")+ylab("Frequency")

# Relationship between Survival and Pclass
ggplot(data = comp[1:891,],aes(x=Pclass,fill=Survived))+geom_bar(position="fill")+ylab("Frequency")

# Now, dividing the graph of Embarked by Pclass:
ggplot(data = comp[1:891,],aes(x=Embarked,fill=Survived))+geom_bar(position="fill")+facet_wrap(~Pclass)

# Relationship between sex and survival:
ggplot(comp[1:891,], aes(Age, fill = factor(Survived))) + 
  geom_histogram(binwidth=3) + 
  facet_grid(.~Sex) + 
  theme_few()

# Relationship between Sex and Survival by Pclass
ggplot(data = comp[1:891,],aes(x=Sex,fill=Survived))+geom_bar(position="fill")+facet_wrap(~Pclass)

# Relationship between Fare and survival:

ggplot(comp[1:891,], aes(Fare, fill = factor(Survived))) + 
  geom_bar(width=6) + 
  theme_few()

# Relationship between Pclass and survival:
ggplot(comp[1:891,], aes(Pclass, fill = factor(Survived))) + 
  geom_bar() + 
  theme_few()

indy <- createDataPartition(y=comp$Survived,
                               p=0.80, list=FALSE)
trainset <- comp[indy,]
testset <- comp[-indy,]

lrmodel <- glm(Survived ~ Pclass + Sex + Age + Embarked, data = trainset, family = 'binomial')
summary(lrmodel)

pred <- predict(lrmodel, testset, type = 'response')
pred[pred > 0.5 ] <- 1
pred[pred < 0.5 | is.na(pred) ] <- 0
pred <- as.factor(pred)
confusionMatrix(pred,testset$Survived)$overall[1]


