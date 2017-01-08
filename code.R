setwd("C:/Users/bjx_3/OneDrive/Documents/mushroom")
mushroom <- read.csv("mushrooms.csv")
head(mushroom)
mushroom <- mushroom[, -which(names(mushroom) == "veil.type")]

mushroom_numeric <- mushroom
for(i in colnames(mushroom)){
  mushroom_numeric[,i] <- as.numeric(mushroom[,i])
}
# poisonous is 1 and eatable is 0
mushroom_numeric[,1] <- mushroom_numeric[,1] - rep(1, 8124)
train_numeric <- mushroom_numeric[1:2708,]
test_numeric <- mushroom_numeric[2709:5417,]
test2_numeric <- mushroom_numeric[5418:8124,]

mushroom.pca <- prcomp(train_numeric[,c(2,3,4,5,6,8,9,10,11,12,13,14,15,16,19,20,21,22)],
                 center = TRUE,
                 scale. = TRUE)
summary(mushroom.pca) -> mushroom_pca_sum
plot(mushroom_pca_sum$importance[3,], type = "b",
    xlab = "Principle component", ylab = "Cumulative Proportion of Variance Explained")
abline(h = 0.9)
plot(mushroom.pca, type = "l")

#compute standard deviation of each principal component
std_dev <- mushroom.pca$sdev

#compute variance
pr_var <- std_dev^2

#proportion of variance explained
prop_varex <- pr_var/sum(pr_var)
#scree plot
plot(prop_varex, xlab = "Principal Component",
       ylab = "Proportion of Variance Explained",
       type = "b")
abline(h=0.9)
#add a training set with principal components
train.data <- data.frame(class = train_numeric$class, mushroom.pca$x)

#we are interested in first 12 PCAs
train.data <- train.data[,1:13]

##run a decision tree
#install.packages("rpart")
library(rpart)
rpart.model <- rpart(class ~ .,data = train.data, method = "class")
rpart.model
plot(rpart.model)
text(rpart.model, use.n = TRUE)

##run logistic regression
model <- glm(class ~ . ,family=binomial(link='logit'),data=train.data)
summary(model)
anova(model, test="Chisq")
#reduce logistic regression model
model <- glm(class ~ . ,family=binomial(link='logit'),data=train.data[, -which(names(train.data) %in% c("PC5", "PC6","PC12"))])
summary(model)
anova(model, test="Chisq")

#transform test into PCA
test.data <- predict(mushroom.pca, newdata = test_numeric[,c(2,3,4,5,6,8,9,10,11,12,13,14,15,16,18,19,20,21,22)])
test.data <- as.data.frame(test.data)

#select the first 30 components
test.data <- test.data[,1:13]

#make prediction on test data
rpart.prediction <- predict(rpart.model, test.data)
rpart.prediction[which(rpart.prediction<=0.5)] = 0
rpart.prediction[which(rpart.prediction>0.5)]=1
mean((rpart.prediction- test_numeric[,1])^2)
table(test_numeric[,1],rpart.prediction)

#make prediction on test data using logistic regression
model.prediction <- predict(model, test.data)
model.prediction[which(model.prediction<=0.5)] = 0
model.prediction[which(model.prediction>0.5)]=1
mean((model.prediction- test_numeric[,1])^2)
table(test_numeric[,1],model.prediction)
