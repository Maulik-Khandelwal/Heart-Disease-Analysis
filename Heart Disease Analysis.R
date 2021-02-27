# We have a data which classifies if patients have heart disease or not according to features in it. 
# We will try to use this data to create a model which tries predict if a patient has this disease or not. 
# We will use logistic regression (classification) algorithm, naive bayes classifier and decision tree.


#Features

# age
# sex
# chest pain type (4 values)
# resting blood pressure
# serum cholestoral in mg/dl
# fasting blood sugar > 120 mg/dl
# resting electrocardiographic results (values 0,1,2)
# maximum heart rate achieved
# exercise induced angina
# oldpeak = ST depression induced by exercise relative to rest
# the slope of the peak exercise ST segment
# number of major vessels (0-3) colored by flourosopy
# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

library(dplyr)
library(DataExplorer)
library(caret)
library(lattice)
library(GGally)
library(ggplot2)
library(tidyverse)
library(ggthemes)
library(caTools)
library(corrplot)
library(cowplot)
library(highcharter)
require(scales)

data <- read.csv("D:/DS-CP/heart.csv")

head(data)

data <- data %>% 
  rename(
    age=ï..age
  )

glimpse(data)

dim(data)

cat('dataset shape -> 303 rows 17 columns')

plot_missing(data)

summary(data)

df <- data %>% 
  mutate(sex = ifelse(sex == 1, "Male", "Female"),
         fbs = ifelse(fbs == 1, "blood_sugar_higher_than_120", "blood_sugar_lower_than_120"),
         exang = ifelse(exang == 1, "angina", "No-angina"),
         cp = if_else(cp == 0, "Medium",
                      if_else(cp == 1, "High", if_else(cp == 2, "Very High", "Agressive"))),
         restecg = if_else(restecg == 0, "Normal",
                           if_else(restecg == 1, "Abnormality", "Probable or Definite")),
         slope = as.factor(slope),
         ca = as.factor(ca),
         thal = as.factor(thal),
         target = ifelse(target == 0, "presence", "absence")
  ) %>% 
  mutate_if(is.character, as.factor) %>% 
  dplyr::select(target, sex, fbs, exang, cp, restecg, slope, ca, thal, everything())

glimpse(df)









cat('Number of target classes -> ', length(unique(df$target))) 

unique(df$target)

df %>%
  select(target)%>%
  group_by(target)%>%
  count(target)

# Bar plot for target (Heart disease) 
ggplot(df, aes(x=target, fill=target)) + 
  geom_bar() +
  xlab("Heart Disease") +
  ylab("Count") +
  ggtitle("Frequency of patients with Presence and Absence of Heart Disease")

prop.table(table(df$target))

# We can see that the distribution is quite balanced. Thanks to this it wouldn't be a bad idea using 
# accuracy to evaluate how well the models perform.






#EDA

#Sex
ggplot(df, aes(x=sex, fill=sex)) + 
  geom_bar() +
  xlab("Sex") +
  ylab("Count") +
  ggtitle("Gender distribution") +
  scale_fill_discrete(name = "Gender", labels = c("Female", "Male"))

prop.table(table(df$sex))

gender <- df %>% 
  count(sex, target) %>% 
  glimpse()

unique(df$sex)

unique(data$sex)

gender %>% 
  hchart("column", hcaes(x = sex, y = n, group = target)) %>% 
  hc_title(
    text = "Analysis based on gender",
    margin = 20,
    align = "center"
  ) %>%
  hc_xAxis(title = list(text = 'Sex')) %>%
  hc_yAxis(title = list(text = 'count')) %>%
  hc_caption(text = "Heart Disease",
             margin = 20,
             align = "center")


# There are approximately half the observation of women than men. We can also see that sex is a risk factor, 
# like some of the references indicate, men are more likely to have a heart disease than women.








#Age

#Patient age in years. 
data %>%
  select(age)%>%
  ggplot(aes(x = age, fill = '#b8cff5'))+
  geom_histogram(aes(y=..density..), color = 'black', fill='#b8cff5')+
  geom_density(aes(y=..density..),color = 'black',fill = 'grey',  alpha = 0.5,  kernel='gaussian')+
  theme_minimal()+
  scale_x_continuous(labels = comma)+
  scale_y_continuous(labels = comma)+
  xlab("Age") +
  ylab("Density") +
  ggtitle("Age Distribution")

boxplot(df$age,main ="boxplot of age for Age distribution",col ='#b8cff5')

ggplot(df,aes(age, fill=target)) +
  geom_histogram(aes(y=..density..),breaks=seq(0, 80, by=1), color="grey17") +
  geom_density(alpha=.1, fill="black")+
  facet_wrap(~target, ncol=1,scale="fixed") +
  scale_fill_manual(values=c("springgreen2","firebrick2"))+
  xlab("Age") +
  ylab("Density") +
  ggtitle("Age based analysis of Heart Disease")

ggplot(df, aes(age, fill=target)) + 
  geom_histogram(binwidth=1) +
  labs(fill="Disease", x="Age", y="Number of patients") +
  ggtitle("Age based analysis of Heart Disease") +
  scale_fill_manual(values=c("springgreen2","firebrick2"))

# In the data we can see, as expected, that age is a risk factor. 
# In other words, the higher the age, the more likely that the patient has a heart disease.









# Chest pain type
# 
# Value 0: asymptomatic
# Value 1: atypical angina
# Value 2: pain without relation to angina
# Value 3: typical angina

Chest_Pain <- df %>% 
  group_by(cp)%>%
  count(cp, target)

data.frame(Chest_Pain)

Chest_Pain <- Chest_Pain %>% 
  rename(
    count=n
  )

ggplot(df, aes(x=cp, fill=cp)) + 
  geom_bar() +
  xlab("Chest pain type") +
  ylab("Count") +
  ggtitle("Chest Pain Analysis")

Chest_Pain %>% 
  hchart("column", hcaes(x = cp, y = count, group = target)) %>% 
  hc_title(
    text = "Chest Pain based analysis of Heart Disease",
    margin = 20,
    align = "center"
  ) %>% 
  hc_xAxis(title = list(text = 'Chest Pain')) %>%
  hc_caption(text = "Heart Disease",
             margin = 20,
             align = "center")

unique(df$cp)

unique(data$cp)

# The description of the data doesn't provide information about how this classification of pain was made. 
# But we can see that it is very difficult to tell whether a patient has a heart disease attending just to 
# the symptoms of the patients.







hch <- function(a, b, c, d, e) {
  
  hcboxplot(x = a, var = b, var2 = c) %>%
    hc_plotOptions(boxplot = list(
      fillColor = "grey"), alpha = 0.7)%>%
    hc_chart(type = "column") %>%
    hc_yAxis(title = list(text = d),
             labels = list(format = "{value}")) %>%
    hc_xAxis(title = list(text = 'target'),
             labels = list(format = "{value}")) %>%
    hc_title(
      text = e,
      margin = 20,
      align = "center"
    )
}










#rest bp

#Resting blood pressure in millimeters of mercury (mm Hg) when the patient was admitted to the hospital.

boxplot(data$trestbps,main ="boxplot of age for Resting blood pressure distribution",col ='#b8cff5')

data %>%
  select(trestbps)%>%
  ggplot(aes(x = trestbps, fill = '#b8cff5'))+
  geom_histogram(aes(y=..density..), color = 'black', fill='#b8cff5')+
  geom_density(aes(y=..density..),color = 'black',fill = 'grey',  alpha = 0.5,  kernel='gaussian')+
  theme_minimal()+
  scale_x_continuous(labels = comma)+
  scale_y_continuous(labels = comma)+
  xlab("Resting blood pressure") +
  ylab("Density") +
  ggtitle("Resting blood pressure Distribution")

blood_pressure <- df %>%
  select(age, sex, trestbps, target) %>%
  group_by(age)%>%
  arrange(desc(trestbps))%>%
  head()

blood_pressure 

hch(df$trestbps, df$target, df$sex, 'resting blood pressure', 'resting blood pressure vs gender')

hch(df$trestbps, df$target, df$age, 'resting blood pressure', 'resting blood pressure vs age')

ggplot(df,aes(trestbps, fill=target)) +
  geom_histogram(aes(y=..density..),breaks=seq(90, 200, by=10), color="grey17") +
  geom_density(alpha=.1, fill="black")+
  facet_wrap(~target, ncol=1,scale="fixed") +
  scale_fill_manual(values=c("springgreen2","firebrick2")) +
  xlab("Resting Blood Pressure (in mm Hg on admission to the hospital") +
  ylab("Density") +
  ggtitle("Resting blood pressure based analysis of Heart Disease")

ggplot(df, aes(trestbps, fill=target)) +
  geom_histogram(binwidth=3) +
  labs(fill="Disease", x="Blood pressure (mm Hg)", y="Number of patients") +
  scale_fill_manual(values=c("springgreen2","firebrick2")) +
  ggtitle("Resting blood pressure based analysis of Heart Disease")

# By the different peaks, looks like most people tend to have a normal blood pressure inside certain groups 
# (could be healthy adults, adults that take medication, seniors...). It also looks like very high pressures 
# can indicate that there is a heart disease.









#chol

#Cholesterol level in mg/dl. 
boxplot(data$chol, main ="boxplot for Cholesterol level in mg/dl. distribution",col ='#b8cff5')

data %>%
  select(chol)%>%
  ggplot(aes(x = chol, fill = '#b8cff5'))+
  geom_histogram(aes(y=..density..), color = 'black', fill='#b8cff5')+
  geom_density(aes(y=..density..),color = 'black',fill = 'grey',  alpha = 0.5,  kernel='gaussian')+
  theme_minimal()+
  scale_x_continuous(labels = comma)+
  scale_y_continuous(labels = comma)+
  xlab("Cholesterol level in mg/dl.") +
  ylab("Density") +
  ggtitle("Cholesterol level in mg/dl. Distribution")

cholestoral <- df %>%
  select(age, sex, chol, target) %>%
  group_by(age)%>%
  arrange(desc(chol))%>%
  head()
cholestoral

hch(df$chol, df$target, df$sex, 'Cholesterol level in mg/dl.', 'Cholesterol level in mg/dl. vs gender')

hch(df$chol, df$target, df$age, 'Cholesterol level in mg/dl.', 'Cholesterol level in mg/dl. vs age')


ggplot(df,aes(chol, fill=target)) +
  geom_histogram(aes(y=..density..),breaks=seq(100, 600, by=25), color="grey17") +
  geom_density(alpha=.1, fill="black")+
  facet_wrap(~target, ncol=1,scale="fixed") +
  scale_fill_manual(values=c("springgreen2","firebrick2")) +
  xlab("Cholesterol level in mg/dl.") +
  ylab("Density") +
  ggtitle("Cholesterol level in mg/dl. based analysis of Heart Disease")

ggplot(df, aes(chol, fill=target)) +
  geom_histogram(binwidth=10) +
  labs(fill="Disease", x="Cholesterol (mg/dl)", y="Number of patients") +
  scale_fill_manual(values=c("springgreen2","firebrick2")) +
  ggtitle("Cholesterol level in mg/dl. based analysis of Heart Disease")

# This is a variable that we can control to prevent the disease. Looks like the majority of the people in 
# the dataset have high levels of cholesterol. It also looks like up to a certain level, the presence of a
# heart disease is slightly higher on higher cholesterol levels. Though the cases that have the 
# highest levels of cholesterol don't have a heart disease, it could be that these people weren't fasting 
# when the blood sample was taken.









#FBS

#Whether the level of sugar in the blood is higher than 120 mg/dl or not. 
ggplot(df, aes(x=fbs, fill=fbs)) + 
  geom_bar() +
  xlab("FBS") +
  ylab("Count") +
  ggtitle("Blood sugar level value count")

prop.table(table(df$fbs))

sugar <- df %>% 
  count(fbs, target) %>% 
  glimpse()

unique(df$fbs)

unique(data$fbs)

sugar %>% 
  hchart("column", hcaes(x = fbs, y = n, group = target)) %>% 
  hc_title(
    text = "FBS based analysis of Heart Disease",
    margin = 20,
    align = "center"
  ) %>% 
  hc_xAxis(title = list(text = 'FBS')) %>%
  hc_yAxis(title = list(text = 'count')) %>%
  hc_caption(text = "Heart Disease",
             margin = 20,
             align = "center")

# This is another variable that we can control. However, by itself it doesn't seem very useful to know if 
# a patient has a heart disease or not. Though we shouldn't drop it right now because it might be useful 
# combined with other variables.







# Hereon, variables are related to a nuclear stress test. That is, a stress test where a radioactive dye is 
# also injected to the patient to see the blood flow.







#restecg

ggplot(df, aes(x=restecg, fill=restecg)) + 
  geom_bar() +
  xlab("electrocardiogram result Types") +
  ylab("Count") +
  ggtitle("electrocardiogram result Analysis")

prop.table(table(df$restecg))

ecg <- df %>% 
  count(restecg, target) %>% 
  glimpse()

unique(df$restecg)

unique(data$restecg)

ecg %>% 
  hchart("column", hcaes(x = restecg, y = n, group = target)) %>% 
  hc_title(
    text = "electrocardiogram result based analysis of Heart Disease",
    margin = 20,
    align = "center"
  ) %>% 
  hc_xAxis(title = list(text = 'electrocardiogram result')) %>%
  hc_yAxis(title = list(text = 'count')) %>%
  hc_caption(text = "Heart Disease",
             margin = 20,
             align = "center")


# Results of the electrocardiogram on rest
# 
# Value 0: probable left ventricular hypertrophy
# Value 1: normal
# Value 2: abnormalities in the T wave or ST segment

#No major difference in Rest ECG for Healthy and Heart Disease patients








#thalach

#Maxium heart rate during the stress test
data %>%
  select(thalach)%>%
  ggplot(aes(x = thalach, fill = '#b8cff5'))+
  geom_histogram(aes(y=..density..), color = 'black', fill='#b8cff5')+
  geom_density(aes(y=..density..),color = 'black',fill = 'grey',  alpha = 0.5,  kernel='gaussian')+
  theme_minimal()+
  scale_x_continuous(labels = comma)+
  scale_y_continuous(labels = comma)+
  xlab("Maximum heart rate during exercise") +
  ylab("Density") +
  ggtitle("Maximum heart rate Distribution")

boxplot(df$thalach,col ='#b8cff5',main ="boxplot for Maximum heart rate during exercise")

max_heart_rate <- df %>%
  select(age, sex, thalach, target) %>%
  group_by(age)%>%
  arrange(desc(thalach))%>%
  head()
max_heart_rate 

hch(df$thalach, df$target, df$sex, 'maximum heart rate achieved', 'maximum heart rate vs gender')

hch(df$thalach, df$target, df$age, 'maximum heart rate achieved', 'maximum heart rate vs age')

ggplot(df,aes(thalach, fill=target)) +
  geom_histogram(aes(y=..density..),breaks=seq(70, 205, by=10), color="grey17") +
  geom_density(alpha=.1, fill="black")+
  facet_wrap(~target, ncol=1,scale="fixed") +
  scale_fill_manual(values=c("springgreen2","firebrick2"))+
  xlab("Maximum Heart Rate Achieved") +
  ylab("Density") +
  ggtitle("Max Heart Rate based analysis of Heart Disease")

ggplot(df, aes(thalach, fill=target)) +
  geom_histogram(binwidth=10) +
  labs(fill="Disease", x="Maximum heart rate during exercise", y="Number of patients") +
  scale_fill_manual(values=c("springgreen2","firebrick2"))+
  ggtitle("Max Heart Rate based analysis of Heart Disease")

# Maximum heart rate during the stress test

# At first sight it may seem weird to see that the higher the heart rate the lower the presence of a 
# heart disease and vice versa. However, it makes sense taking into account that the maximum healthy 
# heart rate depends on the age (220 - age). Thus, higher rates tend to be from younger people.







#exang

#Whether the patient had angina during exercise

ggplot(df, aes(x=exang, fill=exang)) + 
  geom_bar() +
  xlab("Presence of angina during exercise") +
  ylab("Count") +
  ggtitle("Presence of angina during exercise")

prop.table(table(df$exang))

angina <- df %>% 
  count(exang, target) %>% 
  glimpse()

unique(df$exang)

unique(data$exang)

angina %>% 
  hchart("column", hcaes(x = exang, y = n, group = target)) %>% 
  hc_title(
    text = "Angina based analysis",
    margin = 20,
    align = "center"
  ) %>% 
  hc_xAxis(title = list(text = 'Presence of angina during exercise')) %>%
  hc_yAxis(title = list(text = 'count')) %>%
  hc_caption(text = "Heart Disease",
             margin = 20,
             align = "center")


# We can see that this feature is a good indicator for the presence of heart disease. 
# However, we can also see that knowing what is angina and what not is not an easy task, 
# it can be confused with other pains or it can be atypical angina.









#oldpeak

ggplot(df, aes(oldpeak, fill=target)) +
  geom_histogram(binwidth=0.25) +
  labs(fill="Disease", x="Depression of the ST segment", y="Number of patients") +
  scale_fill_manual(values=c("springgreen2","firebrick2"))+
  ggtitle("Analysis of Decrease of the ST segment")

#Decrease of the ST segment during exercise according to the same one on rest.

# The ST segment is a part of the electrocardiogram of a heart beat that is usually found at a certain level
# in a normal heart beat. A significant displacement of this segment can indicate the presence of a 
# heart disease as we can see in the plot.










#slope

# Slope of the ST segment during the most demanding part of the exercise
# 
# Value 0: descending
# Value 1: flat
# Value 2: ascending

ggplot(df, aes(x=slope, fill=slope)) + 
  geom_bar() +
  xlab("Slope of the ST segment") +
  ylab("Count") +
  ggtitle("Slope")

prop.table(table(df$slope))

ggplot(df, aes(slope, fill=target)) +
  geom_bar() +
  labs(fill="Disease", x="Slope of the ST segment", y="Number of patients") +
  scale_fill_manual(values=c("springgreen2","firebrick2"))+
  ggtitle("Slope based analysis of Heart Disease")

ggplot(df, aes(x=slope, y=oldpeak, fill=target)) +
  geom_boxplot() +
  labs(fill="Disease", x="Slope of the ST segment", y="Depression of the ST segment")


# In the first graph we can see that the slope by itself can help determine whether there is a heart disease
# or not if it is flat or ascending. However, if the slope is descending doesn't seem to give much information.
# Because of this, in the second graph we add a third variable where we can see that, if the slope is 
# descending, the depression of the ST segment can help to determine if the patient has a heart disease.








#ca

ggplot(df, aes(x=ca, fill=ca)) + 
  geom_bar() +
  xlab("Number of main blood vessels coloured by the radioactive dye") +
  ylab("Count") +
  ggtitle("Number of main blood vessels coloured by the radioactive dye Analysis")

prop.table(table(df$ca))

vessels <- df %>%
  group_by(ca)%>%
  count(ca, target)%>%
  arrange(desc(ca))                                                                                                                

data.frame(vessels)

vessels %>% 
  hchart("column", hcaes(x = ca, y = n, group = target)) %>% 
  hc_title(
    text = "ca analysis",
    margin = 20,
    align = "center"
  ) %>% 
  hc_xAxis(title = list(text = 'ca')) %>%
  hc_caption(text = "Heart Disease",
             margin = 20,
             align = "center")

unique(df$ca)

unique(data$ca) 


# Number of main blood vessels coloured by the radioactive dye. The number varies between 0 to 4 

# This feature refers to the number of narrow blood vessels seen, this is why the higher the value of this 
# feature, the more likely it is to have a heart disease.










#thal

prop.table(table(df$thal))

ggplot(df, aes(thal, fill=target)) +
  geom_bar() +
  labs(fill="Disease", x="Results of the blood flow", y="Number of patients") +
  scale_fill_manual(values=c("springgreen2","firebrick2"))+
  ggtitle("Analysis of results of the blood flow observed via the radioactive dye")

# Results of the blood flow observed via the radioactive dye.

# Value 0: NULL (dropped from the dataset previously)
# Value 1: fixed defect (no blood flow in some part of the heart)
# Value 2: normal blood flow
# Value 3: reversible defect (a blood flow is observed but it is not normal)

# This feature and the next one are obtained through a very invasive process for the patients. 
#But, by themselves, they give a very good indication of the presence of a heart disease or not.
















#Correlations

cor_heart <- cor(data)
cor_heart

# GGally::ggcorr(data, geom = "circle")

corrplot(cor_heart, method = "ellipse", type="upper",)

ggpairs(data)

#There is no variable which has strong positive or negative correlation with target variable.




# select2 <- data %>%
#   dplyr::select(
#     slope,
#     thalach,
#     restecg,
#     cp,
#     target
#   )
# 
# cor_heart1 <- cor(select2)
# cor_heart1
# 
# ggcorr(select2, geom = "circle")
# 
# corrplot(cor_heart1, method = "ellipse", type="upper",)
# 
# ggpairs(select2)

# From the correlation study it seems that the parameters
# * cp
# * restecg
# * thalach
# * slope
# are the most usefull to predict the risk of heart disease
# 
# From the EDA anlysis it semms that
# * age
# * sex
# * cholesterol
# * restecg
# are also usefull
# 
# For prediction the following variables seems the most usefull
# * age
# * sex
# * cholesterol
# * restecg
# * cp
# * thalach
# * slope
# 
# d<-data[,c(2,3,9,10,12,14)]
# summary(d)
# 
# df_select <- data %>%
#   dplyr::select( #because of conflict between MASS and dplyr select need to use dplyr::select
#     target,
#     age,
#     sex,
#     chol,
#     restecg,
#     cp,
#     thalach,
#     slope
#   )






#Modelling


#Logistic Regression

set.seed(23)
split <- createDataPartition(data$target, time = 1, list = FALSE, p = 0.7)

heart_train <- data[split,]
heart_test <- data[-split,]

dim(data)
dim(heart_train)
dim(heart_test)

heart_test_x <- heart_test %>% dplyr::select(-target)
heart_test_y <- heart_test$target

head(heart_train)
head(heart_test_x)

heart_mod <- glm(target~., data = heart_train, family = "binomial")

heart_mod

prob_pred <- predict(heart_mod, type = "response", newdata = heart_test_x)

y_pred = ifelse(prob_pred > 0.5, 1, 0)

t<-table(y_pred, heart_test_y)
t.df<-as.data.frame(t)

ggplot(data = t.df, aes(x = heart_test_y, y = y_pred, label=Freq)) +
  geom_tile(aes(fill = Freq)) +
  scale_fill_gradient(low="firebrick2", high="springgreen2") +
  theme_economist() +
  xlab("Actual Heart Disease") +
  ylab("Predicted Heart Disease") +
  geom_text(size=8) +
  ggtitle("Logistic Regression")


y_pred <- as.factor(y_pred)
heart_test_y <- as.factor(heart_test_y)
library(caret)
library(e1071)
confusionMatrix(y_pred , heart_test_y)
summary(heart_mod)

library(ROCR)
ROCRpred=prediction(prob_pred, heart_test_y)
ROCRperf=performance(ROCRpred,'tpr','fpr')
plot(ROCRperf)
plot(ROCRperf,colorize=TRUE)
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1),
     text.adj=c(-0.2,1.7))


auc = as.numeric(performance(ROCRpred, 'auc')@y.values)
auc







#Improved Logistic Regression

heart_mod <- glm(target~., data = heart_train, family = "binomial")

prob_pred <- predict(heart_mod, type = "response", newdata = heart_test_x)

y_pred = ifelse(prob_pred > 0.65, 1, 0)

t<-table(y_pred, heart_test_y)
t.df<-as.data.frame(t)

ggplot(data = t.df, aes(x = heart_test_y, y = y_pred, label=Freq)) +
  geom_tile(aes(fill = Freq)) +
  scale_fill_gradient(low="firebrick2", high="springgreen2") +
  theme_economist() +
  xlab("Actual Heart Disease") +
  ylab("Predicted Heart Disease") +
  geom_text(size=8) +
  ggtitle("Logistic Regression")


y_pred <- as.factor(y_pred)
heart_test_y <- as.factor(heart_test_y)
library(caret)
library(e1071)
confusionMatrix(y_pred , heart_test_y)
summary(heart_mod)

library(ROCR)
ROCRpred=prediction(prob_pred, heart_test_y)
ROCRperf=performance(ROCRpred,'tpr','fpr')
plot(ROCRperf)
plot(ROCRperf,colorize=TRUE)
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1),
     text.adj=c(-0.2,1.7))


auc = as.numeric(performance(ROCRpred, 'auc')@y.values)
auc






#Decision Tree

library(rpart)
library(rpart.plot)
fit <- rpart(target~., data = heart_train, method = 'class')
fit

rpart.plot(fit, extra = 106)

predict_unseen <-predict(fit, heart_test_x, type = 'class')

t<-table(predict_unseen, heart_test_y)
t.df<-as.data.frame(t)

ggplot(data = t.df, aes(x = heart_test_y, y = predict_unseen, label=Freq)) +
  geom_tile(aes(fill = Freq)) +
  scale_fill_gradient(low="firebrick2", high="springgreen2") +
  theme_economist() +
  xlab("Actual Heart Disease") +
  ylab("Predicted Heart Disease") +
  geom_text(size=8) +
  ggtitle("Decision Tree")

predict_unseen <- as.factor(predict_unseen)
heart_test_y <- as.factor(heart_test_y)
library(caret)
library(e1071)
confusionMatrix(predict_unseen , heart_test_y)
summary(heart_mod)





#Naive Bayes Classifier

heart_test_y <- as.factor(heart_test_y)
heart_train$target <- as.factor(heart_train$target)


classifier_cl <- naiveBayes(target~., data = heart_train) 
classifier_cl 

prob_pred <- predict(classifier_cl, type = "raw", newdata = heart_test_x)
# prob_pred[,1]

y_pred <- predict(classifier_cl, heart_test_x)
# y_pred
t<-table(y_pred, heart_test_y)
t.df<-as.data.frame(t)

ggplot(data = t.df, aes(x = heart_test_y, y = y_pred, label=Freq)) +
  geom_tile(aes(fill = Freq)) +
  scale_fill_gradient(low="firebrick2", high="springgreen2") +
  theme_economist() +
  xlab("Actual Heart Disease") +
  ylab("Predicted Heart Disease") +
  geom_text(size=8) +
  ggtitle("Naive Bayes")


y_pred <- as.factor(y_pred)
heart_test_y <- as.factor(heart_test_y)
library(caret)
library(e1071)
confusionMatrix(y_pred , heart_test_y)
summary(heart_mod)

library(ROCR)
ROCRpred=prediction(1-prob_pred[,1], heart_test_y)
ROCRperf=performance(ROCRpred,'tpr','fpr')
plot(ROCRperf)
plot(ROCRperf,colorize=TRUE)
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1),
     text.adj=c(-0.2,1.7))


auc = as.numeric(performance(ROCRpred, 'auc')@y.values)
auc

