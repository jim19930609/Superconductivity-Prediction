library("glmnet")

# Readin Data into DataFrame
sem = read.csv('C:\\Users\\jim19\\Desktop\\Statistics_Project\\ML Models\\train.csv')

# Split into Traing & Test
smp_size <- floor(0.8 * nrow(sem))
train_ind <- sample(seq_len(nrow(sem)), size = smp_size)

sem_train = sem[train_ind, ]
sem_test = sem[-train_ind, ]

X_train = as.matrix(sem_train[-82])
Y_train = as.matrix(sem_train["critical_temp"])

X_test = as.matrix(sem_test[-82])
Y_test = as.matrix(sem_test["critical_temp"])

# CV to determine lambda
cvresult = cv.glmnet(x=X_train, 
                     y=Y_train, 
                     nfolds=20, 
                     alpha=1, 
                     type.measure="mse")

plot(cvresult)
best_lambda = cvresult$lambda.min

# Apply Naive Bayes
ptm = proc.time()
sem_lasso = glmnet(x=X_train, y=Y_train, alpha=1, lambda=best_lambda)
proc.time() - ptm

# Train Error
train_pred = predict(sem_lasso, X_train)
TrainRMSE = sqrt( sum( (train_pred - Y_train)^2 ) / length(train_pred) )
TrainRMSE

# Test Error
test_pred = predict(sem_lasso, X_test)
TestRMSE = sqrt( sum( (test_pred - Y_test)^2 ) / length(test_pred) )
TestRMSE

