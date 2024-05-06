# library
library(R.utils)
library(class)
library(caret)
library(psych)
library(factoextra)
library(ggplot2)
library(randomForest)
library(readr)

###########################loading Dataset###################################

load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('train-images-idx3-ubyte')
  test <<- load_image_file('t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('train-labels-idx1-ubyte')
  test$y <<- load_label_file('t10k-labels-idx1-ubyte')  
}

show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}
load_mnist()

# visualizing images
show_digit(train$x[1,])

summary(train$x)
summary(train$y)


train$x[1,]
show_digit(train$x[1,])


#missing value
anyNA(train)
anyNA(test)

#splitting data and normilizing
x_train<- train$x/255
x_test <- test$x/255
y_train <- train$y
y_test <- test$y

##########################   PCA  ############################################


# Proceed PCA 
data.pca <- prcomp(x_train, retx = TRUE, center = TRUE)


# Reduction to two dimensions
PC1 <- data.pca$x[,1]
PC2 <- data.pca$x[,2]
df <- data.frame(PC1 = PC1, PC2 = PC2, label = as.factor(y_train))

# Visualize 2 dimensional dataset
ggplot(df, aes(x = PC1, y = PC2, color = label)) +
  geom_point(alpha = 0.5) +
  scale_color_discrete(name = "Digit") +
  labs(title = "MNIST dataset visualized with PCA (2 dimensions)",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal()

# plotting PCA
fviz_eig(data.pca, ncp =100)
eig.val <- get_eigenvalue(data.pca)

# Plot cumulative variance percentage
plot(eig.val$cumulative.variance.percent, type = "b", col="blue",
     xlab = "Number of Principal Components", 
     ylab = "Cumulative Variance Explained",
     main = "Cumulative Variance Explained by Principal Components")
axis(1, at = seq(0, length(eig.val$cumulative.variance.percent), by = 50))

#calculating the related pca to 80%
desired_percentage <- 80
principal_component <- which.min(abs(eig.val$cumulative.variance.percent - desired_percentage))
principal_component

#transform data
pca_model <- prcomp(x_train, retx = TRUE, center = TRUE, rank. = 43)

# Transform data
x_train_pca <- predict(pca_model, x_train)
x_test_pca <- predict(pca_model, x_test)


#############################finding best K for KNN#############################
##create a subset of data
Find_k = createDataPartition(y_train, p=0.03, list=FALSE, times=1)
train_kx = x_train[Find_k,]
train_ky = y_train[Find_k]

#calculating train error
error_train_full <- replicate(0,43)
for(k in 1:43){
  predictions <-knn(train=train_kx, test=train_kx, cl=train_ky,k)
  error_train_full[k] <- 1-mean(predictions==train_ky)
}
error_train_full <- unlist(error_train_full, use.names=FALSE)

#calculating test error
error_test_full <- replicate(0,43)
for(k in 1:43){
  predictions <- knn(train=train_kx, test=x_test, cl=train_ky, k)
  error_test_full[k] <- 1-mean(predictions==y_test)
}
error_test_full <- unlist(error_test_full, use.names=FALSE)

#visualization of test and train error
png("43_values_knn_no_pca.png", height=800, width=1000)
plot(error_train_full, type="o", ylim=c(0,0.11), col="blue", xlab="K values", ylab="Misclassification errors", main="Test vs train error for varying k values without PCA")
lines(error_test_full, type="o", col="red")
legend("topright",legend=c("Training error", "Test error"), col=c("blue","red"), lty=1:1)
dev.off()

####################################  KNN  ##################################

#KNN(k=3)
KNN3_time <- system.time({
  knn_3 <- knn(train = x_train, test = x_test, cl = train$y, k=3)#1245
})
ACC_3 <- 100 * sum(y_test == knn_3)/NROW(y_test)


#########################best PCA in applying KNN#############################

#define a vector for different PCA
pca_values <- c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200)
accuracies <- numeric(length(pca_values))

#calculating accuracy for different PCA
for (i in seq_along(pca_values)) {
  pca_model <- prcomp(x_train, retx = TRUE, center = TRUE, rank. = pca_values[i])
  x_train_pca <- predict(pca_model, x_train)
  x_test_pca <- predict(pca_model, x_test)
  knn_model <- knn(train = x_train_pca , test = x_test_pca, cl = train$y, k = 3)
  accuracies[i] <- mean(knn_model == test$y)
}

# plotting the result
plot(pca_values, accuracies, type = "l", xlab = "Number of PCA Components", ylab = "Accuracy", main = "Accuracy vs. PCA Components")
result_table <- data.frame(PCA_Components = pca_values, Accuracy = accuracies)


############################KNN on PCA data####################################

#implementation of choosen PCA
pca_model_final <- prcomp(x_train, retx = TRUE, center = TRUE, rank. = 60)
x_train_pca_f <- predict(pca_model_final, x_train)
x_test_pca_f <- predict(pca_model_final, x_test)

#KNN (K=3) on after dimention reduction
KNN3_time_pca <- system.time({
  knn_3_pca <- knn(train = x_train_pca_f , test = x_test_pca_f, cl = train$y, k=3)
  
})
KNN3_time_pca

#accuracy
ACC_3_pca <- 100 * sum(y_test == knn_3_pca)/NROW(y_test)

############################finding best k on PCA data##########################
##create a subset of data
Find_k = createDataPartition(y_train, p=0.03, list=FALSE, times=1)
train_kx = x_train_pca_f[Find_k,]
train_ky = y_train[Find_k]

#calculating train error
error_train_full <- replicate(0,60)
for(k in 1:60){
  predictions <-knn(train=train_kx, test=train_kx, cl=train_ky,k)
  error_train_full[k] <- 1-mean(predictions==train_ky)
}
error_train_full <- unlist(error_train_full, use.names=FALSE)

#calculating test error
error_test_full <- replicate(0,60)
for(k in 1:60){
  predictions <- knn(train=train_kx, test=x_test_pca_f, cl=train_ky, k)
  error_test_full[k] <- 1-mean(predictions==y_test)
}
error_test_full <- unlist(error_test_full, use.names=FALSE)

#visualization of test and train error
png("60_values_knn_pca.png", height=1000, width=1000)
plot(error_train_full, type="o", ylim=c(0,0.20), col="blue", xlab="K values", ylab="Misclassification errors", main="Test vs train error for varying k values with PCA")
lines(error_test_full, type="o", col="red")
legend("topright",legend=c("Training error", "Test error"), col=c("blue","red"), lty=1:1)
dev.off()

############################Random Forest####################################


#initialization 
set.seed(132)
numTrain <- 40000
numTrees <- 100

# Train the Random Forest model
RF_time <- system.time({rf_model <- randomForest(x_train, as.factor(y_train), ntree = numTrees)})
rf_model
plot(rf_model)

# Extract error rates and plotting
err <- rf_model$err.rate

errbydigit <- data.frame(Label =0:9, Error = err[100, 2:11])

errordigitplot <- ggplot( data = errbydigit, aes(x=Label, y= Error))+
  geom_bar(stat = "identity")+
  labs(x = "Digit Label", y = "Error", title = "Error Rates by Digit")
errordigitplot

# Make predictions on the test set
predictions <- predict(rf_model,x_test)

# Create a confusion matrix
confusion <- confusionMatrix(predictions,as.factor(y_test) )

# Calculate accuracy
accuracy <- mean(predictions == y_test)
cat("Accuracy:", accuracy)

############################Comparing KNN and RF####################################
#1. Accuracy
model_comparison <- data.frame(
  Model = c("Random Forest", "kNN (k=3) without PCA", "kNN (k=3) with PCA"),
  Accuracy = c(accuracy*100, ACC_3, ACC_3_pca)
)
model_comparison

#2.Run Time illustration
model_names <- c("Random Forest", "kNN (k=3) without PCA", "kNN (k=3) with PCA")
running_times <- c(447.63, 1245.37, 37.39)

# Create a bar plot
barplot(running_times, names.arg = model_names, col = "skyblue", 
        main = "Model Running Time", xlab = "Model", ylab = "Running Time (seconds)")
text(x = 1:length(model_names), y = running_times, labels = running_times, pos = 3, cex = 0.8, col = "black")

############################Random Forest on PCA Data####################################
#initialization 
set.seed(132)
numTrain <- 40000
numTrees <- 100

# Train the Random Forest model on PCA(43)
RF_time_PCA <- system.time({rf_model_pca <- randomForest(x_train_pca, as.factor(y_train), ntree = numTrees)})

# Make predictions on the test set
predictions_pca <- predict(rf_model_pca,x_test_pca)

# Create a confusion matrix
 confusionMatrix(predictions_pca,as.factor(y_test) )


