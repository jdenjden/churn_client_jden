---
  title: "Project Capstone Churn "
author: "Johnny den Braber Lártiga"
date: "01-03-2022"
output: pdf_document
---
  
  $$INTRODUCTION$$
  
  On the following document we develop a predictive model for the customer churn on a telecommunication company using different machine learning algorithms. In order to determine if the client is ending or not its contract with the company we first need to understand and characterize the client profile and all of the different services that the company offers to the clients. This project use all of the skills and knowledge that we have acquired through this course. 

This report is obtained from an RMarkdown file and has five different sections that show all of the work involved on the development of the different models and the final selection of the most accurate model. Sections 2 and 3 are focused on the preparations and data exploration. Section 4 shows the development of all five models that we tested for this project and Section 5 shows our conclusions and final comments around the project.

We will star this work by uploading the data and the relevant libraries.


## DATA WRANGLING
```{r dependencies, message = FALSE, warning = FALSE}
library(tidyverse)
library(forcats)
library(stringr)
library(caTools)
library(ggthemes)
library(MASS)
library(party)


```


## DATA ASSESSMENT / VISUALIZATIONS
```{r}
library(DT)
library(data.table)
library(pander)
library(ggplot2)
library(scales)
library(grid)
library(gridExtra)
library(corrplot)
library(VIM) 
library(knitr)
library(vcd)
library(caret)

```


## MODEL
```{r}
library(xgboost)
library(MLmetrics)
library(randomForest) 
library(rpart)
library(rpart.plot)
library(car)
library(e1071)
library(vcd)
library(ROCR)
library(pROC)
library(VIM)
library(glmnet) 
library(plyr)
```


\newpage

### SECTION II: 
$$DATA\,\,\,PREPARATION$$
  The available data is organized on 7043 different rows, where each one represents each client, and 21 columns that represent each variable associated to the clients. The target variable for these models will be the Churn variable and the explanatory variables will be the remaining 20 variables, which are the following: 'CustomerID', 'Gender', 'SeniorCitizen', 'Partner', 'Dependents', 'Tenure', 'PhoneService', 'MultipleLines, 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges' and 'TotalCharges'.


First we will identify and treat the missing values with an special interest on the Tenure variable, where the minimum is 1 month and the maximum is 72 months, which were classified on five intervals of 12 moths each. We also changed the Senior Citizen from 0 and 1 to No and Yes. this process was also performed for the Internet Service and Phone Service variables.

The organization and cleaning process is a critical step on data science and the posterior development of models for analysis.


```{r}
churn <- read.csv('C:/Users/brabe/Documents/R_Edx_Capstone_A/churn_jden/TelcoCustomerChurn.csv')

```


```{r}
dim(churn)
str(churn)
```


```{r}
sapply(churn, function(x) sum(is.na(x)))
```

```{r}
churn <- churn[complete.cases(churn), ]
```

#####
```{r}
cols_recode1 <- c(10:15)
for(i in 1:ncol(churn[,cols_recode1])) {
  churn[,cols_recode1][,i] <- as.factor(mapvalues
                                        (churn[,cols_recode1][,i], from =c("No internet service"),to=c("No")))
}
```


```{r}
churn$MultipleLines <- as.factor(mapvalues(churn$MultipleLines, 
                                           from=c("No phone service"),
                                           to=c("No")))
```


####
```{r}
min(churn$tenure); max(churn$tenure)
```


```{r}
group_tenure <- function(tenure){
  if (tenure >= 0 & tenure <= 12){
    return('0-12 Month')
  }else if(tenure > 12 & tenure <= 24){
    return('12-24 Month')
  }else if (tenure > 24 & tenure <= 48){
    return('24-48 Month')
  }else if (tenure > 48 & tenure <=60){
    return('48-60 Month')
  }else if (tenure > 60){
    return('> 60 Month')
  }
}
```


```{r}
churn$tenure_group <- sapply(churn$tenure,group_tenure)
churn$tenure_group <- as.factor(churn$tenure_group)
```


```{r}
churn$SeniorCitizen <- as.factor(mapvalues(churn$SeniorCitizen,
                                           from=c("0","1"),
                                           to=c("No", "Yes")))
```


```{r}
churn$customerID <- NULL
churn$tenure <- NULL
```


## Exploratory data analysis and feature selection
```{r echo=FALSE}
numeric.var <- sapply(churn, is.numeric) ## Find numerical variables
corr.matrix <- cor(churn[,numeric.var])  ## Calculate the correlation matrix
kable(corr.matrix, 
      caption="Correlation Plot for Numeric Variables", 
      align = "cc",
      digits = 2)
```


```{r}
churn$TotalCharges <- NULL
```




\newpage
### SECTION III
$$DATA\,\,\,VISUALIZATION$$
  Here we add a set of graphs to get a global overview of the behavior and distribution of the variables that we will use to develop our models, such as Gender, Contract and Tenure Group.

## Bar plots of categorical variables.
```{r}
graf_1 <- ggplot(churn, aes(x=gender)) + 
  ggtitle("Gender") + xlab("Gender") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'blue' ,width = 0.5) + 
  ylab("Percentage") + 
  coord_flip() + 
  theme_minimal()
graf_2 <- ggplot(churn, aes(x=SeniorCitizen)) + 
  ggtitle("Senior Citizen") + xlab("Senior Citizen") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'red' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
graf_3 <- ggplot(churn, aes(x=Partner)) + 
  ggtitle("Partner") + xlab("Partner") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'blue' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
graf_4 <- ggplot(churn, aes(x=Dependents)) + 
  ggtitle("Dependents") + xlab("Dependents") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'red' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
grid.arrange(graf_1, graf_2, graf_3, graf_4, ncol=2)
```


\newpage
```{r}
graf_5 <- ggplot(churn, aes(x=PhoneService)) + 
  ggtitle("Phone Service") + xlab("Phone Service") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'blue' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
graf_6 <- ggplot(churn, aes(x=MultipleLines)) + 
  ggtitle("Multiple Lines") + xlab("Multiple Lines") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'red' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
graf_7 <- ggplot(churn, aes(x=InternetService)) + ggtitle("Internet Service") + 
  xlab("Internet Service") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'blue' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
graf_8 <- ggplot(churn, aes(x=OnlineSecurity)) + ggtitle("Online Security") + 
  xlab("Online Security") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'red' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
grid.arrange(graf_5, graf_6, graf_7, graf_8, ncol=2)
```


\newpage
```{r}
graf_9 <- ggplot(churn, aes(x=OnlineBackup)) + 
  ggtitle("Online Backup") + xlab("Online Backup") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'blue' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
graf_10 <- ggplot(churn, aes(x=DeviceProtection)) + 
  ggtitle("Device Protection") + xlab("Device Protection") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'red' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
graf_11 <- ggplot(churn, aes(x=TechSupport)) + 
  ggtitle("Tech Support") + xlab("Tech Support") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'blue'  ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
graf_12 <- ggplot(churn, aes(x=StreamingTV)) + 
  ggtitle("Streaming TV") + xlab("Streaming TV") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'red'  ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
grid.arrange(graf_9, graf_10, graf_11, graf_12, ncol=2)
```


\newpage
```{r}
graf_13 <- ggplot(churn, aes(x=StreamingMovies)) + 
  ggtitle("Streaming Movies") + xlab("Streaming Movies") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'blue' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
graf_14 <- ggplot(churn, aes(x=Contract)) + 
  ggtitle("Contract") + xlab("Contract") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'red' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
graf_15 <- ggplot(churn, aes(x=PaperlessBilling)) + 
  ggtitle("Paperless Billing") + xlab("Paperless Billing") + 
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'blue' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
grid.arrange(graf_13, graf_14, graf_15, ncol=2)
```

\newpage
```{r}
graf_16 <- ggplot(churn, aes(x=PaymentMethod)) + 
  ggtitle("Payment Method") + xlab("Payment Method") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'blue' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
graf_17 <- ggplot(churn, aes(x=tenure_group)) + 
  ggtitle("Tenure Group") + xlab("Tenure Group") +
  geom_bar(aes(y = 100*(..count..)/sum(..count..)), fill = 'red' ,width = 0.5) + 
  ylab("Percentage") + coord_flip() + 
  theme_minimal()
grid.arrange(graf_16, graf_17, ncol=2)
```

\newpage
```{r}
library(dplyr)
library(tidyverse)
```

## We show a table for the main variable.
```{r}
tabla <- churn %>%
  dplyr::group_by(Churn) %>%
  dplyr::summarize(count=n())
kable(tabla, caption = 'Observation to Churn', aling = 'cc')
```






\newpage

### SECTION IV
$$MODELS$$
  
  **Training Set** - In machine learning, a training set is a dataset used to train a model.  In training the model, specific features are picked out from the training set.  These features are then incorporated into the model. 

**Test Set** - The test set is a dataset used to measure how well the model performs at making predictions on that test set.

## Section 4 show the different models that we used to obtain the best possible prediction. The models used are the following: On first place we start using a decision tree model, next we used a random forest model, logistic regression model and support vector machine model, to finish with a radial support vector machine model. 

```{r, message=FALSE, warning=FALSE}
feauter_1<-churn[1:7032, c("gender", "InternetService","PhoneService","Contract","PaymentMethod","tenure_group")]

response <- as.factor(churn$Churn)

feauter_1$Churn=as.factor(churn$Churn)
```

Verifying data.
```{r, message=FALSE, warning=FALSE}

set.seed(500)
ind=createDataPartition(feauter_1$Churn,times=1,p=0.8,list=FALSE)
train_val=feauter_1[ind,]
test_val=feauter_1[-ind,]

```


#### Here we check the Churn rate in the orginal training data, current training data and test data.
```{r, message=FALSE, warning=FALSE}

round(prop.table(table(churn$Churn)*100),digits = 1)

round(prop.table(table(train_val$Churn)*100),digits = 1)

round(prop.table(table(test_val$Churn)*100),digits = 1)

```

\newpage
#####
### MODEL DECISION TREE (Predictive Analysis and Cross Validation {.tabset})
### Model Decision Tree {-}

```{r, message=FALSE, warning=FALSE}

set.seed(1234)
Model_DT=rpart(Churn~.,data=train_val,method="class")
rpart.plot(Model_DT,extra =  3,fallen.leaves = T)

```


```{r, message=FALSE, warning=FALSE}

PRE_TDT=predict(Model_DT,data=train_val,type="class")
confusionMatrix(PRE_TDT,train_val$Churn)

set.seed(1234)
cv.10 <- createMultiFolds(train_val$Churn, k = 10, times = 10)


### Control
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                     index = cv.10)
train_val <- as.data.frame(train_val)


###Train the data
Model_CDT <- train(x = train_val[,-7], y = train_val[,7], method = "rpart", tuneLength = 30,
                   trControl = ctrl)


rpart.plot(Model_CDT$finalModel,extra =  3,fallen.leaves = T)

PRE_VDTS=predict(Model_CDT$finalModel,newdata=test_val,type="class")
confusionMatrix(PRE_VDTS,test_val$Churn)


col_names <- names(train_val)
train_val[col_names] <- lapply(train_val[col_names] , factor)
test_val[col_names] <- lapply(test_val[col_names] , factor)

```


\newpage
## RANDOM FOREST MODEL
#### Random Forest {-}
```{r, message=FALSE, warning=FALSE}

set.seed(1234)

rf.1 <- randomForest(x = train_val[,-7],y=train_val[,7], importance = TRUE, ntree = 1000)
rf.1
varImpPlot(rf.1)


train_val1=train_val[,-4:-5]
test_val1=test_val[,-4:-5]


set.seed(1234)
rf.2 <- randomForest(x = train_val1[,-5],y=train_val1[,5], importance = TRUE, ntree = 1000)
rf.2
varImpPlot(rf.2)


set.seed(2348)
cv10_1 <- createMultiFolds(train_val1[,5], k = 10, times = 10)


ctrl_1 <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                       index = cv10_1)


set.seed(1234)
rf.5<- train(x = train_val1[,-5], y = train_val1[,5], method = "rf", tuneLength = 3,
             ntree = 1000, trControl =ctrl_1)
rf.5


pr.rf=predict(rf.5,newdata = test_val1)
confusionMatrix(pr.rf,test_val1$Churn)


```



\newpage
## LOGISTIC REGRESSION MODEL
#### Logistic Regression {-}
```{r, message=FALSE, warning=FALSE}

contrasts(train_val$gender)
contrasts(train_val$Contract)


log.mod <- glm(Churn ~ ., family = binomial(link=logit), 
               data = train_val1)


summary(log.mod)
confint(log.mod)

train.probs <- predict(log.mod, data=train_val1,type =  "response")#
table(train_val1$Churn,train.probs>0.5)


test.probs <- predict(log.mod, newdata=test_val1,type =  "response")
table(test_val1$Churn,test.probs>0.5)

```


\newpage
#### SUPPORT VECTOR MACHINE MODEL
#### Linear Support vector Machine {-}
```{r, message=FALSE, warning=FALSE}
set.seed(1274)

liner.tune=tune.svm(Churn~.,data=train_val,kernel="linear",cost=c(0.01,0.1,0.2,0.5,0.7,1,2,3,5,10,15,20,50,100))
liner.tune

best.linear=liner.tune$best.model

best.test=predict(best.linear,newdata=test_val,type="class")
confusionMatrix(best.test,test_val$Churn)

```

### RADIAL SUPPORT VECTOR MACHINE MODEL: (Non Linear Kernel give us a better accuracy) 
#### bRadial Support vector Machine {-}
```{r, message=FALSE, warning=FALSE}
set.seed(1274)
rd.poly=tune.svm(Churn~.,data=train_val1,kernel="radial",gamma=seq(0.1,5))

summary(rd.poly)
best.rd=rd.poly$best.model

pre.rd=predict(best.rd,newdata = test_val1)
confusionMatrix(pre.rd,test_val1$Churn)

```


\newpage
### SECTION V
$$CONCLUSIONS$$
  The different models that were developed show an acceptable accuracy level for the amount of variables used and the complexity of the model itself.

After the test of the models we can say that the Contract variable is the most relevant variable to predict the end of the service with the company. As such, when the contract of a client has lasted for a year or more, this client is more likely to end its contract rather than a client with a contract that has lasted less than a year, regardless of the gender, the kind of service or the payment method.

On the logistic regression model we can highlight that the selected variables are highly significant for the accuracy on the prediction of the model. On the case of the random forest model we obtained that a 'No' prediction is more accurate than a 'Yes' prediction for the same dataset. For the overall accuracy, all the models present an accuracy of around 0.79.


This next table synthesize the accuracy obtained for each model.
```{r}
models <- c("Decision Tree","Random Forest","Logic Regression", "Support Vector Machine", "Radial Support Vector Machine")
accur <- c(0.7917, 0.7957, 0.7941, 0.7943, 0.7955)
modelos <- data.frame(models,accur)
kable(modelos,caption="Accuracy by model",
      align="lc", col.names = c("Models","Accuracy"))
```






### SECTION VI
$$BIBLIOGRAPHY$$
  
  
  Garet James et.al 2021 An Introduction to Statistical Learning with Application in R 2da ed. edition.

Irizarry A. Rafael and Love I. Michael. Data Analysis for the life Sciences.








\newpage
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## R Markdown

This is an R Markdown document. Markdown is a simple formatting syntax for authoring HTML, PDF, and MS Word documents. For more details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated that includes both content as well as the output of any embedded R code chunks within the document. You can embed an R code chunk like this:
  
  ```{r cars}
summary(cars)
```

## Including Plots

You can also embed plots, for example:
  
  ```{r pressure, echo=FALSE}
plot(pressure)
```

Note that the `echo = FALSE` parameter was added to the code chunk to prevent printing of the R code that generated the plot.