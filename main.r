# load libs
require(neuralnet)
require(nnet)
# require(ggplot2)


# Set seed for reproducibility purposes
set.seed(10)

normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

dataset <- read.csv("yeast.csv")
dim(dataset)
names(dataset)
str(dataset)
# Encode as a one hot vector multilabel data
train <- cbind(dataset[, 2:9], class.ind(as.factor(dataset$label)))
# Set labels name
names(train) <- c(names(dataset)[2:9],"CYT","NUC","MIT",
                  "ME3","ME2","ME1","EXC","VAC","POX", "ERL")
# Scale data
train[, 2:9] <- as.data.frame(lapply(train[, 2:9], normalize))
train <- train[,2:19]
str(train)
# Set up formula
n <- names(train)
f <- as.formula(paste("CYT + NUC + MIT + ME3 + ME2 + ME1 +
                      EXC + VAC + POX + ERL ~",
                      paste(n[!n %in% c("CYT","NUC","MIT",
                                        "ME3","ME2","ME1",
                                        "EXC","VAC","POX",
                                        "ERL")],
                            collapse = " + ")))
# 10 fold cross validation
k <- 10
# Results from cv
outs <- NULL
# Train test split proportions
proportion <- 0.95 # Set to 0.995 for LOOCV

# Crossvalidate, go!
for(i in 1:k){
  print(paste("Iteración ", i))
  tryCatch({
    index <- sample(1:nrow(train), round(proportion*nrow(train)))
    train_cv <- train[index, ]
    test_cv <- train[-index, ]
    nn_cv <- neuralnet(f,
                       data = train_cv,
                       hidden = 2,
                       act.fct = "logistic",
                       threshold = 0.001,
                       algorithm = "rprop+",
                       learningrate.factor = list(minus = 0.5, plus = 1.2),
                       stepmax = 1e+06,
                       lifesign = "full",
                       linear.output = FALSE)
    print("Red neuronal")
    # Compute predictions
    pr.nn <- compute(nn_cv, test_cv[, 1:8])
    print("Clasificación realizada")
    # Extract results
    pr.nn_ <- pr.nn$net.result
    # Accuracy (test set)
    original_values <- max.col(test_cv[, 9:18])
    pr.nn_2 <- max.col(pr.nn_)
    outs[i] <- mean(pr.nn_2 == original_values)
    print(paste("Resultado ", outs[i]))
  })
}

mean(outs)