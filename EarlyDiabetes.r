##########################################################
# File Name: EarlyDiabetes.r
# File Type: R Programming Language
# Author: Daniel R. Alexander
# Created on: 25 May 2021
# Description: Data science project to read in early stage
#    diabetes risk prediction data set.  Dataset contains
#    the sign and symptom data of newly diabetic or
#    would-be diabetic patients. Goal is to find best 
#    predictors for early stage diabetes.

# Will use the following classification models with scores:
# Logistic Regression
# Random Forest Classifier

##########################################################

##########################################################
# INGEST THE DATA
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("vcd", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("ROCR", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(vcd) # For mosaic plots
library(randomForest)  # Classification Algorithm
library(ROCR) #AUC ROC

# Early Stage Diabetes Risk Prediction Data Set:
# https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv

dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/00529/diabetes_data_upload.csv", dl)

data_set <- read.csv(dl)

##########################################################
# EXPLORATORY DATA ANALYSIS (EDA)
##########################################################

# What are my variables?
glimpse(data_set) # Only Age is numeric (int).  All others are binary Yes/No
names(data_set)
head(data_set)

# See how many unique values are in the data set
sapply(data_set, function(x) length(unique(x)))

# See if there are any NA values
sapply(data_set, function(x) sum(is.na(x)))

# see if there are any empty values
colSums(data_set == '')

# Class is the variable I want to predict so...

# Change class to factor
class(data_set$class)
data_set$class <- factor(data_set$class)
class(data_set$class)

ggplot(data_set, aes(class)) + geom_bar()

# Explore age since it is the only numeric variable
summary(data_set$Age)

data_set %>%
  group_by(Age) %>%
  ggplot(aes(Age)) +
  geom_histogram(bins = 20, color = "black") +
  ggtitle("Age")

# What are the relationships between variables?
boxplot(data_set$Age~data_set$class, main="Age vs. Diabetes", xlab="class", ylab="Age")

# Produce contingency tables to see how certain symptoms impact outcome
x = c()
x = names(data_set)
x = x[-17] # Remove class
x = x[-1] # Remove Age
# Convert remaining variables to factors then print contingency tables
for (i in x){
  print(i)
  data_set[,i]=as.factor(data_set[,i])
  print(table(data_set[[i]], data_set$class))
}
# Looks like Polyureia, Polydipsia, Alopecia are strongly correlated
ggplot(data_set,aes(Polyuria,class)) + geom_jitter()
ggplot(data_set,aes(Polydipsia,class)) + geom_jitter()
ggplot(data_set,aes(Alopecia,class)) + geom_jitter()

### Create mosaicplots to visualize Polyuria and Polydipsia
mosaicplot(table(data_set$Polyuria, data_set$class), main = "Diabetic by Polyuria", shade = TRUE, legend = TRUE)
mosaicplot(table(data_set$Polydipsia, data_set$class), main = "Diabetic by Polydipsia", shade = TRUE, legend = TRUE)

##########################################################
# DO THE WORK
##########################################################

# Create train_set (80%), test_set (20%)
set.seed(1)
test_index <- createDataPartition(y = data_set$Age, times = 1, p = 0.2, list = FALSE)
test_set <- data_set[test_index,]
train_set <- data_set[-test_index,]

## LOGISTIC REGRESSION
# Create Model
LR_model <- glm(class ~.,family=binomial(link='logit'),data=train_set)
summary(LR_model)

# Analyze the table of deviance
anova(LR_model, test='Chisq')

# Predict test data
LR_result <- predict(LR_model, newdata = test_set, type='response')
LR_result <- ifelse(LR_result > 0.5,"Positive","Negative")
LR_result <- as.factor(LR_result)
confusionMatrix(data=LR_result, reference=test_set$class)

# Check the model's performance with AUC-ROC:
prediction1 <- predict(LR_model, newdata=test_set, type="response")

ROC_pred <- prediction(prediction1, test_set$class)
ROC_perf <- performance(ROC_pred, measure = "tpr", x.measure = "fpr")

plot(ROC_perf, colorize = TRUE, text.adj = c(-0.2,1.7), print.cutoffs.at = seq(0,1,0.1))
auc <- performance(ROC_pred, measure = "auc")
auc <- auc@y.values[[1]]
auc

## RANDOM FOREST
RF_model <- randomForest(factor(class) ~ Age + Gender + Polyuria + Polydipsia + sudden.weight.loss + weakness + Polyphagia + Genital.thrush + visual.blurring + Itching + Irritability + delayed.healing + partial.paresis + muscle.stiffness + Alopecia + Obesity, data = train_set)

# Plot the error rate of the model
plot(RF_model, ylim = c(0, 0.36))
legend("topright", colnames(RF_model$err.rate), col = 1:3, fill = 1:3)

# Rank the variables and plot on bar graph
importance <- importance(RF_model)

# Lower Gini means stronger factor in partitioning data into classes
varImportance <- data.frame(Variables = row.names(importance), Importance = round(importance[ ,'MeanDecreaseGini'],2))

rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

ggplot(rankImportance, aes(x = reorder(Variables, Importance), y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank), hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() + 
  theme_bw()

prediction2 <- predict(RF_model, test_set)
RF_predictions <- data.frame(test_set, class=prediction2)
table(RF_predictions$class)
table(test_set$class)
