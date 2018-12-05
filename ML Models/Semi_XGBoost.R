library("xgboost")

# Readin Data into DataFrame
sem = read.csv('C:\\Users\\jim19\\Desktop\\Statistics_Project\\train.csv')

# Split into Traing & Test
smp_size <- floor(0.75 * nrow(sem))
train_ind <- sample(seq_len(nrow(sem)), size = smp_size)

sem_train = sem[train_ind, ]
sem_test = sem[-train_ind, ]

X_train = as.matrix(sem_train[-82])
Y_train = as.matrix(sem_train["critical_temp"])

X_test = as.matrix(sem_test[-82])
Y_test = as.matrix(sem_test["critical_temp"])

# Apply Naive Bayes
model_xg = xgboost(data = X_train, label = Y_train, max_depth = 20, eta = 0.5, nthread = 2, nrounds = 5, objective = "reg:linear")

# Training MSE
pred_train = predict(model_xg, X_train)
TrainMSE = sqrt(sum((pred_train - Y_train)^2)/length(pred_train))
TrainMSE

# Test MSE
pred_test = predict(model_xg, X_test)
TestMSE = sqrt(sum((pred_test - Y_test)^2)/length(pred_test))
TestMSE